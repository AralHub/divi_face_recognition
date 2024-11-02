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
        collections = await db.get_collections_names()
        for collection in collections:
            await self.create_index(collection)

    async def create_index(self, collection: str):
        async with self.lock:
            self.indexes[collection] = faiss.IndexFlatIP(self.dimension)
            self.id_maps[collection] = []

            async for face in db.get_docs_from_collection(collection):
                await self.add_face(
                    collection, np.array(face["embedding"]), face["person_id"]
                )

    async def add_face(self, collection: str, embedding: np.ndarray, person_id: int):
        if collection not in self.indexes:
            await self.create_index(collection)

        loop = asyncio.get_event_loop()
        vectors = np.array([embedding]).astype("float32")

        async with self.lock:
            await loop.run_in_executor(
                self.executor,
                lambda: self._add_to_index(collection, vectors, person_id),
            )

    def _add_to_index(self, collection: str, vectors: np.ndarray, person_id: int):
        faiss.normalize_L2(vectors)
        self.indexes[collection].add(vectors)
        self.id_maps[collection].append(person_id)

    async def search(self, collection: str, embedding: np.ndarray) -> Tuple[float, int]:
        if collection not in self.indexes:
            return 0.0, 0

        loop = asyncio.get_event_loop()
        query = np.array([embedding]).astype("float32")
        faiss.normalize_L2(query)

        async with self.lock:
            scores, ids = await loop.run_in_executor(
                self.executor, lambda: self.indexes[collection].search(query, 1)
            )

        if len(ids[0]) == 0:
            return 0.0, 0

        return float(scores[0][0]), self.id_maps[collection][ids[0][0]]


matcher = AsyncFaceMatcher()
