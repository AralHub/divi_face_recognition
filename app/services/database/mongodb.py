from typing import List, AsyncIterator

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient

from core.config import settings
from schemas.data import DataRead


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

    async def delete_image_by_key(self, collection: str, key: str) -> bool:
        result = await self.db[collection].delete_one({"key": key})
        return result.deleted_count > 0

    async def get_image_by_key(self, collection: str, key: str):
        face = await self.db[collection].find_one({"key": key})
        return face

    async def save_to_db(self, document: DataRead):
        result = await self.db.documents.insert_one(document)
        return result.inserted_id

    async def get_documents(self):
        documents = self.db.documents.find()
        return await documents.to_list(None)  # Загружаем все документы в список

    async def get_images_by_person(self, collection: str, person_id: int):
        images = self.db[collection].find({"person_id": person_id})
        return await images.to_list(None)  # Загружаем все изображения в список


db = AsyncMongoDB()
