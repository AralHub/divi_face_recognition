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
        lock_value = os.getpid()
        timeout = self.LOCK_TIMEOUT
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            if await self.redis.set(lock_key, lock_value, ex=self.LOCK_EXPIRE, nx=True):
                return True
            await asyncio.sleep(0.1)
        return False

    async def release_lock(self, lock_key: str):
        """Освобождение распределенной блокировки"""
        lock_value = await self.redis.get(lock_key)
        if lock_value and int(lock_value) == os.getpid():
            await self.redis.delete(lock_key)

    async def initialize(self):
        """Инициализация индексов для всех коллекций"""
        collections = await db.get_collections_names()
        for collection in collections:
            await self.create_index(collection)

    async def create_index(self, collection: str):
        """Создание индекса с усредненными эмбеддингами"""
        lock_key = self._get_lock_key(collection)

        if await self.redis.exists(self._get_index_key(collection)):
            return

        if not await self.acquire_lock(lock_key):
            logger.warning(f"Could not acquire lock for collection {collection}")
            return

        try:
            if await self.redis.exists(self._get_index_key(collection)):
                return

            # Получаем все лица из коллекции
            faces = []
            async for face in db.get_docs_from_collection(collection):
                embedding = face.get("embedding")
                person_id = face.get("person_id")
                if embedding is not None and person_id is not None:
                    faces.append((person_id, np.array(embedding).astype("float32")))

            if not faces:
                return

            # Группируем эмбеддинги по person_id
            embeddings_by_person = {}
            for person_id, embedding in faces:
                if person_id not in embeddings_by_person:
                    embeddings_by_person[person_id] = []
                embeddings_by_person[person_id].append(embedding)

            # Вычисляем средние эмбеддинги
            average_embeddings = []
            person_ids = []
            for person_id, embs in embeddings_by_person.items():
                avg_emb = np.mean(embs, axis=0)
                average_embeddings.append(avg_emb)
                person_ids.append(person_id)

            # Создаем FAISS индекс
            temp_index = faiss.IndexFlatIP(self.dimension)
            vectors = np.array(average_embeddings).astype("float32")
            faiss.normalize_L2(vectors)
            temp_index.add(vectors)

            # Сохраняем индекс и id_map
            async with self.redis.pipeline() as pipe:
                pipe.set(self._get_index_key(collection), pickle.dumps(temp_index))
                pipe.set(self._get_id_map_key(collection), pickle.dumps(person_ids))
                await pipe.execute()

        finally:
            await self.release_lock(lock_key)

    async def search(self, collection: str, embedding: np.ndarray) -> Tuple[float, int]:
        """Поиск ближайшего усредненного эмбеддинга"""
        try:
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

    async def add_face(self, collection: str, embedding: np.ndarray, person_id: int):
        """Добавление нового эмбеддинга и обновление среднего"""
        lock_key = self._get_lock_key(collection)

        if not await self.redis.exists(self._get_index_key(collection)):
            await self.create_index(collection)
            return

        if not await self.acquire_lock(lock_key):
            raise Exception(f"Could not acquire lock for collection {collection}")

        try:
            async with self.redis.pipeline() as pipe:
                pipe.get(self._get_index_key(collection))
                pipe.get(self._get_id_map_key(collection))
                index_bytes, id_map_bytes = await pipe.execute()

            if not index_bytes or not id_map_bytes:
                await self.create_index(collection)
                return

            current_index = pickle.loads(index_bytes)
            current_id_map = pickle.loads(id_map_bytes)

            # Получаем все эмбеддинги для person_id из MongoDB
            faces = []
            async for face in db.get_docs_from_collection_by_person_id(
                collection, person_id
            ):
                emb = face.get("embedding")
                if emb is not None:
                    faces.append(np.array(emb).astype("float32"))

            # Добавляем новый эмбеддинг
            new_emb = np.array(embedding).astype("float32")
            faces.append(new_emb)

            # Вычисляем новое среднее
            avg_emb = np.mean(faces, axis=0)

            # Обновляем или добавляем в индекс
            vectors = np.array([avg_emb]).astype("float32")
            faiss.normalize_L2(vectors)

            if person_id in current_id_map:
                idx = current_id_map.index(person_id)
                # FAISS не поддерживает прямое обновление, пересоздаем индекс
                new_index = faiss.IndexFlatIP(self.dimension)
                for i in range(current_index.ntotal):
                    if i == idx:
                        new_index.add(vectors)
                    else:
                        vec = current_index.reconstruct(i)
                        new_index.add(np.array([vec]))
                current_index = new_index
            else:
                current_index.add(vectors)
                current_id_map.append(person_id)

            # Сохраняем обновленные данные
            async with self.redis.pipeline() as pipe:
                pipe.set(self._get_index_key(collection), pickle.dumps(current_index))
                pipe.set(self._get_id_map_key(collection), pickle.dumps(current_id_map))
                await pipe.execute()

        except Exception as e:
            logger.error(f"Error adding face: {e}")
            raise

        finally:
            await self.release_lock(lock_key)

    async def delete_person(self, collection: str, person_id: int):
        """Удаление person_id из индекса"""
        lock_key = self._get_lock_key(collection)

        if not await self.redis.exists(self._get_index_key(collection)):
            return

        if not await self.acquire_lock(lock_key):
            raise Exception(f"Could not acquire lock for collection {collection}")

        try:
            async with self.redis.pipeline() as pipe:
                pipe.get(self._get_index_key(collection))
                pipe.get(self._get_id_map_key(collection))
                index_bytes, id_map_bytes = await pipe.execute()

            if not index_bytes or not id_map_bytes:
                return

            current_index = pickle.loads(index_bytes)
            current_id_map = pickle.loads(id_map_bytes)

            if person_id not in current_id_map:
                return

            idx_to_remove = current_id_map.index(person_id)

            # Создаем новый индекс без удаляемого person_id
            new_index = faiss.IndexFlatIP(self.dimension)
            new_id_map = []
            for i in range(current_index.ntotal):
                if i != idx_to_remove:
                    vector = current_index.reconstruct(i)
                    new_index.add(np.array([vector]))
                    new_id_map.append(current_id_map[i])

            # Сохраняем обновленные данные
            async with self.redis.pipeline() as pipe:
                pipe.set(self._get_index_key(collection), pickle.dumps(new_index))
                pipe.set(self._get_id_map_key(collection), pickle.dumps(new_id_map))
                await pipe.execute()

        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            raise

        finally:
            await self.release_lock(lock_key)

    async def delete_face(self, collection: str, person_id: int):
        """Удаление конкретного лица и обновление среднего эмбеддинга"""
        lock_key = self._get_lock_key(collection)

        if not await self.redis.exists(self._get_index_key(collection)):
            return

        if not await self.acquire_lock(lock_key):
            raise Exception(
                f"Не удалось получить блокировку для коллекции {collection}"
            )

        try:

            # Получаем все оставшиеся эмбеддинги для person_id
            remaining_faces = []
            async for face in db.get_docs_from_collection_by_person_id(
                collection, person_id
            ):
                emb = face.get("embedding")
                if emb is not None:
                    remaining_faces.append(np.array(emb).astype("float32"))

            # Получаем текущий индекс и id_map
            async with self.redis.pipeline() as pipe:
                pipe.get(self._get_index_key(collection))
                pipe.get(self._get_id_map_key(collection))
                index_bytes, id_map_bytes = await pipe.execute()

            if not index_bytes or not id_map_bytes:
                return

            current_index = pickle.loads(index_bytes)
            current_id_map = pickle.loads(id_map_bytes)

            # Если у персоны не осталось лиц, удаляем из индекса
            if not remaining_faces:
                if person_id in current_id_map:
                    idx_to_remove = current_id_map.index(person_id)

                    # Создаем новый индекс без удаляемого person_id
                    new_index = faiss.IndexFlatIP(self.dimension)
                    new_id_map = []
                    for i in range(current_index.ntotal):
                        if i != idx_to_remove:
                            vector = current_index.reconstruct(i)
                            new_index.add(np.array([vector]))
                            new_id_map.append(current_id_map[i])

                    # Сохраняем обновленные данные
                    async with self.redis.pipeline() as pipe:
                        pipe.set(
                            self._get_index_key(collection), pickle.dumps(new_index)
                        )
                        pipe.set(
                            self._get_id_map_key(collection), pickle.dumps(new_id_map)
                        )
                        await pipe.execute()
            else:
                # Вычисляем новое среднее и обновляем индекс
                avg_emb = np.mean(remaining_faces, axis=0)
                vectors = np.array([avg_emb]).astype("float32")
                faiss.normalize_L2(vectors)

                if person_id in current_id_map:
                    idx = current_id_map.index(person_id)
                    # FAISS не поддерживает прямое обновление, пересоздаем индекс
                    new_index = faiss.IndexFlatIP(self.dimension)
                    for i in range(current_index.ntotal):
                        if i == idx:
                            new_index.add(vectors)
                        else:
                            vec = current_index.reconstruct(i)
                            new_index.add(np.array([vec]))

                    # Сохраняем обновленные данные
                    async with self.redis.pipeline() as pipe:
                        pipe.set(
                            self._get_index_key(collection), pickle.dumps(new_index)
                        )
                        pipe.set(
                            self._get_id_map_key(collection),
                            pickle.dumps(current_id_map),
                        )
                        await pipe.execute()

        except Exception as e:
            logger.error(f"Ошибка при удалении лица: {e}")
            raise

        finally:
            await self.release_lock(lock_key)

    async def get_index_stats(self, collection: str) -> Dict:
        """Получение статистики индекса"""
        index_key = self._get_index_key(collection)
        id_map_key = self._get_id_map_key(collection)

        if not await self.redis.exists(index_key):
            return {"error": "Index not found"}

        index_bytes = await self.redis.get(index_key)
        id_map_bytes = await self.redis.get(id_map_key)
        index = pickle.loads(index_bytes)
        id_map = pickle.loads(id_map_bytes)
        return {
            "total_persons": index.ntotal,  # Теперь это количество person_id
            "id_map_length": len(id_map),
            "dimension": self.dimension,
        }


matcher = RedisFaceMatcher()
