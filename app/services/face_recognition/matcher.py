import asyncio
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import faiss
import numpy as np
import redis.asyncio as redis
from core.config import settings
from services.database.mongodb import db
from services.face_recognition.processor import logger


class RedisFaceMatcher:
    def __init__(
        self,
        redis_host=settings.redis_config.host,
        redis_port=settings.redis_config.port,
        password=settings.redis_config.password,
    ):
        self.dimension = 512
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=password,
            decode_responses=False,
        )
        self.executor = ThreadPoolExecutor(max_workers=settings.WORKER_POOL_SIZE)

        # Константы для распределенных блокировок
        self.LOCK_EXPIRE = 10  # время жизни блокировки в секундах
        self.LOCK_TIMEOUT = 5  # время ожидания блокировки

    def _get_index_key(self, collection: str) -> str:
        """Генерирует уникальный ключ для индекса в Redis"""
        return f"faiss_index:{collection}"

    def _get_id_map_key(self, collection: str) -> str:
        """Генерирует уникальный ключ для маппинга ID в Redis"""
        return f"faiss_id_map:{collection}"

    def _get_lock_key(self, collection: str) -> str:
        """Ключ для распределенной блокировки"""
        return f"lock:{collection}"

    async def acquire_lock(self, lock_key: str) -> bool:
        """Получение распределенной блокировки"""
        lock_value = os.getpid()  # используем PID процесса как идентификатор
        timeout = self.LOCK_TIMEOUT
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            # Пытаемся установить блокировку с помощью SETNX
            if await self.redis.set(lock_key, lock_value, ex=self.LOCK_EXPIRE, nx=True):
                return True
            await asyncio.sleep(0.1)
        return False

    async def initialize(self):
        collections = await db.get_collections_names()
        for collection in collections:
            await self.create_index(collection)

    async def release_lock(self, lock_key: str):
        """Освобождение распределенной блокировки"""
        # Проверяем, что блокировка принадлежит текущему процессу
        lock_value = await self.redis.get(lock_key)
        if lock_value and int(lock_value) == os.getpid():
            await self.redis.delete(lock_key)

    async def create_index(self, collection: str):
        """Создание индекса с распределенной блокировкой"""
        lock_key = self._get_lock_key(collection)

        # Проверяем существование индекса без блокировки
        if await self.redis.exists(self._get_index_key(collection)):
            return
        # Пытаемся получить блокировку
        if not await self.acquire_lock(lock_key):
            logger.warning(f"Could not acquire lock for collection {collection}")
            return

        try:
            # Повторная проверка после получения блокировки
            if await self.redis.exists(self._get_index_key(collection)):
                return

            temp_index = faiss.IndexFlatIP(self.dimension)
            temp_id_map = []

            async for face in db.get_docs_from_collection(collection):
                try:
                    embedding = face.get("embedding")
                    person_id = face.get("person_id")

                    if embedding is None or person_id is None:
                        continue

                    vectors = np.array([embedding]).astype("float32")
                    if vectors.shape[1] != self.dimension:
                        continue

                    faiss.normalize_L2(vectors)
                    temp_index.add(vectors)
                    temp_id_map.append(person_id)

                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue

            if temp_index.ntotal == len(temp_id_map):
                # Атомарное сохранение обоих значений
                async with self.redis.pipeline() as pipe:
                    pipe.set(self._get_index_key(collection), pickle.dumps(temp_index))
                    pipe.set(
                        self._get_id_map_key(collection), pickle.dumps(temp_id_map)
                    )
                    await pipe.execute()

        finally:
            await self.release_lock(lock_key)

    async def search(self, collection: str, embedding: np.ndarray) -> Tuple[float, int]:
        """Поиск без блокировки, так как чтение безопасно"""
        try:
            # Атомарное получение данных
            async with self.redis.pipeline() as pipe:
                pipe.get(self._get_index_key(collection))
                pipe.get(self._get_id_map_key(collection))
                index_bytes, id_map_bytes = await pipe.execute()

            if not index_bytes or not id_map_bytes:
                return 0.0, 0

            current_index = pickle.loads(index_bytes)
            current_id_map = pickle.loads(id_map_bytes)

            query = np.array([embedding]).astype("float32")
            faiss.normalize_L2(query)

            loop = asyncio.get_event_loop()
            scores, indices = await loop.run_in_executor(
                self.executor, lambda: current_index.search(query, 1)
            )

            if len(indices[0]) == 0:
                return 0.0, 0

            return float(scores[0][0]), current_id_map[indices[0][0]]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return 0.0, 0

    async def get_index_stats(self, collection: str) -> Dict:
        """Получение статистики индекса"""
        index_key = self._get_index_key(collection)
        id_map_key = self._get_id_map_key(collection)

        # Проверяем существование индекса
        if not await self.redis.exists(index_key):
            return {"error": "Index not found"}
        # Загружаем индекс
        index_bytes = await self.redis.get(index_key)
        id_map_bytes = await self.redis.get(id_map_key)
        index = pickle.loads(index_bytes)
        id_map = pickle.loads(id_map_bytes)
        return {
            "total_vectors": index.ntotal,
            "id_map_length": len(id_map),
            "dimension": self.dimension,
        }

    async def add_face(self, collection: str, embedding: np.ndarray, person_id: int):
        """Добавление лица с распределенной блокировкой"""
        lock_key = self._get_lock_key(collection)

        if not await self.redis.exists(lock_key):
            await self.create_index(collection)
            return

        if not await self.acquire_lock(lock_key):
            raise Exception(f"Could not acquire lock for collection {collection}")

        try:
            # Атомарное получение данных
            async with self.redis.pipeline() as pipe:
                pipe.get(self._get_index_key(collection))
                pipe.get(self._get_id_map_key(collection))
                index_bytes, id_map_bytes = await pipe.execute()

            if not index_bytes or not id_map_bytes:
                await self.create_index(collection)
                return

            current_index = pickle.loads(index_bytes)
            current_id_map = pickle.loads(id_map_bytes)

            vectors = np.array([embedding]).astype("float32")
            faiss.normalize_L2(vectors)

            current_index.add(vectors)
            current_id_map.append(person_id)

            # Атомарное сохранение обновленных данных
            async with self.redis.pipeline() as pipe:
                pipe.set(self._get_index_key(collection), pickle.dumps(current_index))
                pipe.set(self._get_id_map_key(collection), pickle.dumps(current_id_map))
                await pipe.execute()

        except Exception as e:
            logger.error(f"Error adding face: {e}")
            raise

        finally:
            await self.release_lock(lock_key)


matcher = RedisFaceMatcher()
