import asyncio
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import faiss
import numpy as np
import redis.asyncio as redis
from aioredlock import Aioredlock

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
        self.lock_manager = Aioredlock(
            [{"host": redis_host, "port": redis_port, "password": password}]
        )

    def _get_index_key(self, collection: str) -> str:
        """Генерирует уникальный ключ для индекса в Redis"""
        return f"faiss_index:{collection}"

    def _get_id_map_key(self, collection: str) -> str:
        """Генерирует уникальный ключ для маппинга ID в Redis"""
        return f"faiss_id_map:{collection}"

    def _get_lock_key(self, collection: str) -> str:
        """Генерирует уникальный ключ для блокировки коллекции"""
        return f"lock:{collection}"

    async def initialize(self):
        """Инициализация индексов для всех коллекций"""
        try:
            collections = await db.get_collections_names()

            for collection in collections:
                await self.create_index(collection)
        except Exception as e:
            logger.error(f"Error initializing indexes: {e}")
            raise

    async def create_index(self, collection: str):
        """Создание индекса для коллекции"""
        lock_key = self._get_lock_key(collection)

        async with self.lock_manager.lock(lock_key):
            # Проверяем существование индекса
            index_exists = await self.redis.exists(self._get_index_key(collection))
            if index_exists:
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
                    logger.error(f"Error creating face: {e}")
                    continue

            if temp_index.ntotal == len(temp_id_map):
                index_bytes = pickle.dumps(temp_index)
                id_map_bytes = pickle.dumps(temp_id_map)
                await self.redis.set(self._get_index_key(collection), index_bytes)
                await self.redis.set(self._get_id_map_key(collection), id_map_bytes)

    async def delete_collection_index(self, collection: str):
        """Удаление индекса коллекции"""
        lock_key = self._get_lock_key(collection)

        async with self.lock_manager.lock(lock_key):
            await self.redis.delete(self._get_index_key(collection))
            await self.redis.delete(self._get_id_map_key(collection))

    async def add_face(self, collection: str, embedding: np.ndarray, person_id: int):
        """Добавление нового лица в индекс"""
        lock_key = self._get_lock_key(collection)

        async with self.lock_manager.lock(lock_key):
            index_key = self._get_index_key(collection)
            id_map_key = self._get_id_map_key(collection)

            if not await self.redis.exists(index_key):
                await self.create_index(collection)

            try:
                index_bytes = await self.redis.get(index_key)
                id_map_bytes = await self.redis.get(id_map_key)

                current_index = pickle.loads(index_bytes)
                current_id_map = pickle.loads(id_map_bytes)

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

    async def delete_face(self, collection: str, person_id: int):
        """Удаление конкретного лица из индекса"""
        lock_key = self._get_lock_key(collection)

        async with self.lock_manager.lock(lock_key):
            temp_index = faiss.IndexFlatIP(self.dimension)
            temp_id_map = []

            async for face in db.get_docs_from_collection(collection):
                if face["person_id"] != person_id:
                    vectors = np.array([face["embedding"]]).astype("float32")
                    faiss.normalize_L2(vectors)
                    temp_index.add(vectors)
                    temp_id_map.append(face["person_id"])

            index_bytes = pickle.dumps(temp_index)
            id_map_bytes = pickle.dumps(temp_id_map)

            await self.redis.set(self._get_index_key(collection), index_bytes)
            await self.redis.set(self._get_id_map_key(collection), id_map_bytes)


matcher = RedisFaceMatcher()
