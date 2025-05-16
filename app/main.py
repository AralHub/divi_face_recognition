from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI

from api.routes.databases import router as database_router
from api.routes.face_recognition import router as face_recognition_router
from services.face_recognition.divi_service import divi_service
from services.face_recognition.divi_processor import processor
from services.face_recognition.divi_matcher import matcher
from core.config import settings
redis_client = redis.StrictRedis(
    host=settings.redis_config.host,
    port=settings.redis_config.port,
    password=settings.redis_config.password,
    db=0,
    decode_responses=True,
)




@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(database_router)


app.include_router(face_recognition_router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
