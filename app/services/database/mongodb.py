from typing import List, AsyncIterator

from motor.motor_asyncio import AsyncIOMotorClient

from core.config import settings


class AsyncMongoDB:
    def __init__(self):
        self.client = AsyncIOMotorClient(settings.MONGODB_URL)
        self.db = self.client[settings.DATABASE_NAME]

    async def get_collections_names(self) -> List[str]:
        collections = await self.db.list_collection_names()
        return [c for c in collections if c != "system.indexes"]

    async def get_docs_from_collection(self, collection: str) -> AsyncIterator[dict]:
        async for doc in self.db[collection].find():
            yield doc

    async def add_face_to_collection(self, collection: str, face_data: dict) -> int:
        result = await self.db[collection].insert_one(face_data)
        return result.inserted_id

    async def delete_face(self, collection: str, face_id: int) -> bool:
        result = await self.db[collection].delete_one({"_id": face_id})
        return result.deleted_count > 0


db = AsyncMongoDB()
