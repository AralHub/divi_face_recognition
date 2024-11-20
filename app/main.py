import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes.databases import router as database_router
from api.routes.face_recognition import router as face_recognition_router
from api.routes.save_to_db import router as save_to_db_router
from core.config import settings
from services.face_recognition.matcher import matcher


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up application...")
    await matcher.initialize()
    yield
    print("Shutting down application...")


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Create upload directory
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Mount static files directory
app.mount("/media", StaticFiles(directory=settings.UPLOAD_DIR), name="media")

# Include routers
app.include_router(database_router)
app.include_router(face_recognition_router)

app.include_router(save_to_db_router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True, port=1234, host="localhost")
