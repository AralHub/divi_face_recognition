import redis.asyncio as redis
from datetime import datetime
from pydantic import BaseModel
from core.config import settings
from schemas.face_meta import TemplateFaceData


class TemplateEmbedding:
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
        self.expiration_time = 3600  # Время жизни данных в секундах (1 час)

    async def get_face_data(self, key: str):
        """Получает данные лица по ключу."""
        data = await self.redis.get(key)
        if data:
            return TemplateFaceData.parse_raw(data)
        return None

    async def add_face(self, new_face: TemplateFaceData):
        """Добавляет данные лица с автоудалением через 1 час."""
        document = new_face.model_dump_json()
        await self.redis.setex(new_face.key, self.expiration_time, document)

    async def delete_face(self, key: str):
        """Удаляет данные лица по ключу."""
        await self.redis.delete(key)

    async def close(self):
        """Закрывает подключение к Redis."""
        await self.redis.close()


template_db = TemplateEmbedding()
