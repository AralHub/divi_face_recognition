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
        self.lock = asyncio.Lock()

    def _get_index_key(self, collection: str) -> str:
        """Генерирует уникальный ключ для индекса в Redis"""
        return f"faiss_index:{collection}"

    def _get_id_map_key(self, collection: str) -> str:
        """Генерирует уникальный ключ для маппинга ID в Redis"""
        return f"faiss_id_map:{collection}"

    async def initialize(self):
        """Инициализация индексов для всех коллекций"""
        try:
            collections = await db.get_collections_names()
            initialization_tasks = [
                self.create_index(collection) for collection in collections
            ]
            await asyncio.gather(*initialization_tasks)
            print(
                f"Worker {os.getpid()}: Initialized FAISS indexes for collections {collections}"
            )
        except Exception as e:
            print(f"Worker {os.getpid()}: Initialization error: {e}")
            raise

    async def create_index(self, collection: str):
        """Создание индекса для коллекции"""
        # Проверяем существование индекса
        index_exists = await self.redis.exists(self._get_index_key(collection))
        if index_exists:
            return

        # Создаем временный индекс
        temp_index = faiss.IndexFlatIP(self.dimension)
        temp_id_map = []

        async for face in db.get_docs_from_collection(collection):
            try:
                embedding = face.get("embedding")
                person_id = face.get("person_id")

                if embedding is None or person_id is None:
                    continue

                vectors = np.array([embedding]).astype("float32")

                # Проверка размерности
                if vectors.shape[1] != self.dimension:
                    continue

                faiss.normalize_L2(vectors)
                temp_index.add(vectors)
                temp_id_map.append(person_id)

            except Exception as e:
                logger.error(f"Error creating face: {e}")
                continue

        # Сохраняем индекс и маппинг в Redis
        async with self.lock:
            if temp_index.ntotal == len(temp_id_map):
                # Сериализуем индекс
                index_bytes = pickle.dumps(temp_index)
                await self.redis.set(self._get_index_key(collection), index_bytes)

                # Сериализуем маппинг ID
                id_map_bytes = pickle.dumps(temp_id_map)
                await self.redis.set(self._get_id_map_key(collection), id_map_bytes)

    async def delete_collection_index(self, collection: str):
        """Удаление индекса коллекции"""
        async with self.lock:
            await self.redis.delete(self._get_index_key(collection))
            await self.redis.delete(self._get_id_map_key(collection))

    async def update_collection_index(self, collection: str):
        """Полное обновление индекса коллекции"""
        # Сначала удаляем существующий индекс
        await self.delete_collection_index(collection)

        # Создаем новый
        await self.create_index(collection)

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
        """Добавление нового лица в индекс"""
        # Если индекса нет, создаем
        index_key = self._get_index_key(collection)
        id_map_key = self._get_id_map_key(collection)

        if not await self.redis.exists(index_key):
            await self.create_index(collection)

        async with self.lock:
            try:
                # Загружаем текущий индекс и маппинг
                index_bytes = await self.redis.get(index_key)
                id_map_bytes = await self.redis.get(id_map_key)

                current_index = pickle.loads(index_bytes)
                current_id_map = pickle.loads(id_map_bytes)

                # Подготавливаем вектор
                vectors = np.array([embedding]).astype("float32")
                faiss.normalize_L2(vectors)

                current_index.add(vectors)
                current_id_map.append(person_id)

                updated_index_bytes = pickle.dumps(current_index)
                updated_id_map_bytes = pickle.dumps(current_id_map)

                await self.redis.set(index_key, updated_index_bytes)
                await self.redis.set(id_map_key, updated_id_map_bytes)

            except Exception as e:
                logger.error(f"Error adding face: {e}")
                raise

    async def search(self, collection: str, embedding: np.ndarray) -> Tuple[float, int]:
        """Поиск наиболее похожего лица"""
        index_key = self._get_index_key(collection)
        id_map_key = self._get_id_map_key(collection)

        # Проверяем существование индекса
        if not await self.redis.exists(index_key):
            return 0.0, 0

        try:
            # Подготавливаем запрос
            query = np.array([embedding]).astype("float32")
            faiss.normalize_L2(query)

            # Загружаем индекс и маппинг
            async with self.lock:
                index_bytes = await self.redis.get(index_key)
                id_map_bytes = await self.redis.get(id_map_key)

                current_index = pickle.loads(index_bytes)
                current_id_map = pickle.loads(id_map_bytes)

                # Выполняем поиск
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

    async def partial_update_index(self, collection: str, person_ids: List[int]):
        """Частичное обновление индекса"""
        # Создаем новый индекс
        temp_index = faiss.IndexFlatIP(self.dimension)
        temp_id_map = []

        # Получаем текущие данные
        current_faces = []
        async for face in db.get_docs_from_collection(collection):
            if face["person_id"] not in person_ids or face["person_id"] in person_ids:
                vectors = np.array([face["embedding"]]).astype("float32")
                faiss.normalize_L2(vectors)
                current_faces.append((vectors, face["person_id"]))

        # Создаем новый индекс
        for vectors, person_id in current_faces:
            temp_index.add(vectors)
            temp_id_map.append(person_id)

        # Сохраняем в Redis
        async with self.lock:
            index_bytes = pickle.dumps(temp_index)
            id_map_bytes = pickle.dumps(temp_id_map)

            await self.redis.set(self._get_index_key(collection), index_bytes)
            await self.redis.set(self._get_id_map_key(collection), id_map_bytes)

    async def delete_face(self, collection: str, person_id: int):
        """Удаление конкретного лица из индекса"""
        # Создаем новый индекс
        temp_index = faiss.IndexFlatIP(self.dimension)
        temp_id_map = []

        # Получаем все лица кроме удаляемого
        async for face in db.get_docs_from_collection(collection):
            if face["person_id"] != person_id:
                vectors = np.array([face["embedding"]]).astype("float32")
                faiss.normalize_L2(vectors)
                temp_index.add(vectors)
                temp_id_map.append(face["person_id"])

        # Сохраняем в Redis
        async with self.lock:
            index_bytes = pickle.dumps(temp_index)
            id_map_bytes = pickle.dumps(temp_id_map)

            await self.redis.set(self._get_index_key(collection), index_bytes)
            await self.redis.set(self._get_id_map_key(collection), id_map_bytes)


matcher = RedisFaceMatcher()
