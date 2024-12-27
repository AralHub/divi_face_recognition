import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.databases import router as database_router
from api.routes.face_recognition import router as face_recognition_router
from services.face_recognition.matcher import matcher
from services.face_recognition.processor import processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("RUN_MAIN") == "true":
        await matcher.initialize()
        await processor.initialize_model()
    yield
    print("Shutting down application...")


# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

app.include_router(database_router)


app.include_router(face_recognition_router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
