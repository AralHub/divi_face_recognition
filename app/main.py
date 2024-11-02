import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes.databases import router as database_router
from app.api.routes.face_recognition import router as face_recognition_router
from app.core.config import settings
from services.face_recognition.matcher import matcher

app = FastAPI()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
# Монтируем директорию с загруженными файлами как статические файлы
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

app.include_router(database_router)
app.include_router(face_recognition_router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    matcher.initialize()
    yield


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, port=1234, host="localhost")
