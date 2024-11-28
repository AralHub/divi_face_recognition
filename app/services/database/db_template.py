from motor.motor_asyncio import AsyncIOMotorClient

from core.config import settings
from schemas.face_meta import TemplateFaceData


class TemplateEmbedding:
    def __init__(self):
        self.client = AsyncIOMotorClient(settings.MONGODB_URL)
        self.db = self.client[settings.DATABASE_NAME]
        self.temp_db = self.db['template']

    async def get_face_data(self, key: str):
        doc = await self.temp_db.find_one({"key": key})
        if doc:
            doc = dict(doc)
            doc.pop("_id", None)
            return TemplateFaceData(**doc)
        return None

    async def add_face(self, new_face: TemplateFaceData):
        await self.temp_db.insert_one(new_face.model_dump())
        return

    async def delete_face(self, key: str):
        await self.temp_db.delete_one({"key": key})
        return


template_db = TemplateEmbedding()
