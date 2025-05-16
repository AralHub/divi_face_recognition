from typing import List, AsyncIterator

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from core.config import settings


class AsyncMongoDB:
    def __init__(self):
        self.client = AsyncIOMotorClient(settings.MONGODB_URL)
        self.db = self.client[settings.DATABASE_NAME]

    async def get_collections_names(self) -> List[str]:
        collections = await self.db.list_collection_names()
        return [c for c in collections if c != "system.indexes"]

    async def get_docs_from_collection(self, collection: str) -> List[dict]:
        cursor = self.db[collection].find()
        return await cursor.to_list(length=None)

    async def add_face_to_collection(
        self, collection: str, face_data: dict
    ) -> ObjectId:
        result = await self.db[collection].insert_one(face_data)
        return result.inserted_id

    async def delete_face(self, collection: str, face_id: int) -> bool:
        result = await self.db[collection].delete_one({"_id": face_id})
        return result.deleted_count > 0

    async def delete_person(self, collection: str, person_id: int) -> bool:
        result = await self.db[collection].delete_many({"person_id": person_id})
        return result.deleted_count > 0

    async def delete_collection(self, collection: str) -> bool:
        result = await self.db.drop_collection(collection)
        return True

    async def get_documents(self):
        documents = self.db.documents.find()
        return await documents.to_list(None)  # Загружаем все документы в список

    async def count_documents(self, collection: str) -> int:
        return await self.db[collection].count_documents({})

    async def get_documents_limit(self, collection: str, limit: int) -> List[dict]:
        documents = self.db[collection].find().limit(limit)
        return await documents.to_list(None)

db = AsyncMongoDB()
