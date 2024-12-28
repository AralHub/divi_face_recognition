from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI

from api.routes.databases import router as database_router
from api.routes.face_recognition import router as face_recognition_router
from core.config import settings
from services.face_recognition.matcher import matcher
from services.face_recognition.processor import processor

redis_client = redis.StrictRedis(
    host=settings.redis_config.host,
    port=settings.redis_config.port,
    password=settings.redis_config.password,
    db=0,
    decode_responses=True,
)


async def initialize_once():
    # Проверяем, был ли уже выполнен запуск
    if not redis_client.get("initialization_done"):
        await matcher.initialize()
        await processor.initialize_model()
        # Устанавливаем флаг в Redis, чтобы другие воркеры знали, что инициализация уже выполнена
        redis_client.set("initialization_done", "true")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_once()
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(database_router)


app.include_router(face_recognition_router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
