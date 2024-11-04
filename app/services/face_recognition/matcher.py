import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import faiss
import numpy as np

from core.config import settings
from services.database.mongodb import db


class AsyncFaceMatcher:
    def __init__(self):
        self.dimension = 512
        self.indexes: Dict[str, faiss.Index] = {}
        self.id_maps: Dict[str, List[int]] = {}
        self.executor = ThreadPoolExecutor(max_workers=settings.WORKER_POOL_SIZE)
        self.lock = asyncio.Lock()

    async def initialize(self):
        try:
            collections = await db.get_collections_names()
            initialization_tasks = [
                self.create_index(collection) for collection in collections
            ]
            await asyncio.gather(*initialization_tasks)
            print(f"Initialization tasks for collection {collections}")
        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    async def create_index(self, collection: str):
        """Create a new index for the collection with logging and validation"""

        if collection in self.indexes:
            return

        temp_index = faiss.IndexFlatIP(self.dimension)
        temp_id_map = []

        # Счетчик для отладки
        face_count = 0

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
                face_count += 1

            except Exception as e:
                continue

        async with self.lock:
            # Проверяем, что индексы соответствуют друг другу
            if temp_index.ntotal != len(temp_id_map):
                return

            self.indexes[collection] = temp_index
            self.id_maps[collection] = temp_id_map

    async def delete_collection_index(self, collection: str):
        """Delete index and mapping for the specified collection with validation"""

        async with self.lock:
            if collection in self.indexes:
                # Очищаем память явно
                self.indexes[collection].reset()
                del self.indexes[collection]
                del self.id_maps[collection]

    async def update_collection_index(self, collection: str):
        """Full index update for collection with validation"""

        # Сохраняем старые данные временно
        old_index = self.indexes.get(collection)
        old_map = self.id_maps.get(collection)

        try:
            await self.delete_collection_index(collection)
            await self.create_index(collection)

            # Проверяем, что обновление прошло успешно
            if collection not in self.indexes:
                raise Exception("Failed to create new index")

        except Exception as e:
            # Восстанавливаем старые данные в случае ошибки
            if old_index is not None and old_map is not None:
                self.indexes[collection] = old_index
                self.id_maps[collection] = old_map
            raise

    async def get_index_stats(self, collection: str) -> Dict:
        """Get current index statistics for debugging"""
        if collection not in self.indexes:
            return {"error": "Index not found"}

        return {
            "total_vectors": self.indexes[collection].ntotal,
            "id_map_length": len(self.id_maps[collection]),
            "dimension": self.dimension,
        }

    async def partial_update_index(self, collection: str, person_ids: List[int]):
        """Частичное обновление индекса только для указанных person_ids"""
        if collection not in self.indexes:
            await self.create_index(collection)
            return

        temp_index = faiss.IndexFlatIP(self.dimension)
        temp_id_map = []

        # Получаем текущие данные
        current_faces = []
        async for face in db.get_docs_from_collection(collection):
            if face["person_id"] not in person_ids:
                # Сохраняем существующие лица, не затронутые обновлением
                vectors = np.array([face["embedding"]]).astype("float32")
                faiss.normalize_L2(vectors)
                current_faces.append((vectors, face["person_id"]))

        # Получаем обновленные данные
        async for face in db.get_docs_from_collection(collection):
            if face["person_id"] in person_ids:
                vectors = np.array([face["embedding"]]).astype("float32")
                faiss.normalize_L2(vectors)
                current_faces.append((vectors, face["person_id"]))

        # Создаем новый индекс
        for vectors, person_id in current_faces:
            temp_index.add(vectors)
            temp_id_map.append(person_id)

        async with self.lock:
            self.indexes[collection] = temp_index
            self.id_maps[collection] = temp_id_map

    async def delete_face(self, collection: str, person_id: int):
        """Удаляет конкретное лицо из индекса"""
        if collection not in self.indexes:
            return

        temp_index = faiss.IndexFlatIP(self.dimension)
        temp_id_map = []

        # Получаем все лица кроме удаляемого
        async for face in db.get_docs_from_collection(collection):
            if face["person_id"] != person_id:
                vectors = np.array([face["embedding"]]).astype("float32")
                faiss.normalize_L2(vectors)
                temp_index.add(vectors)
                temp_id_map.append(face["person_id"])

        async with self.lock:
            self.indexes[collection] = temp_index
            self.id_maps[collection] = temp_id_map

    async def add_face(self, collection: str, embedding: np.ndarray, person_id: int):
        """Добавляет новое лицо в индекс"""
        if collection not in self.indexes:
            await self.create_index(collection)

        vectors = np.array([embedding]).astype("float32")
        faiss.normalize_L2(vectors)

        async with self.lock:
            try:
                current_index = self.indexes[collection]
                current_id_map = self.id_maps[collection]

                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self.executor, lambda: current_index.add(vectors)
                )

                current_id_map.append(person_id)

            except Exception as e:
                print(f"Error adding face: {e}")
                raise

    async def search(self, collection: str, embedding: np.ndarray) -> Tuple[float, int]:
        """Ищет наиболее похожее лицо в коллекции"""
        if collection not in self.indexes:
            return 0.0, 0

        try:
            query = np.array([embedding]).astype("float32")
            faiss.normalize_L2(query)

            async with self.lock:
                current_index = self.indexes[collection]
                current_id_map = self.id_maps[collection]

                loop = asyncio.get_event_loop()
                scores, indices = await loop.run_in_executor(
                    self.executor, lambda: current_index.search(query, 1)
                )

                if len(indices[0]) == 0:
                    return 0.0, 0

                return float(scores[0][0]), current_id_map[indices[0][0]]

        except Exception as e:
            print(f"Search error: {e}")
            return 0.0, 0


# Создание синглтона
matcher = AsyncFaceMatcher()

"""
# Удаление индекса коллекции
await matcher.delete_collection_index("collection_name")

# Полное обновление индекса
await matcher.update_collection_index("collection_name")

# Частичное обновление для конкретных person_ids
await matcher.partial_update_index("collection_name", [1, 2, 3])

# Удаление конкретного лица
await matcher.delete_face("collection_name", person_id=1)
"""
