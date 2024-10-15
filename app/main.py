import os

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import router
from app.config import settings
from app.services.face_recognition import face_recognition

app = FastAPI()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
# Монтируем директорию с загруженными файлами как статические файлы
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

app.include_router(router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.on_event("startup")
async def startup_event():
    face_recognition.load_indices()


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
