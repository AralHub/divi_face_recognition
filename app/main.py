from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.databases import router as database_router
from api.routes.face_recognition import router as face_recognition_router
from services.database.localdb import db    
from services.face_recognition.divi_service import divi_service
from services.face_recognition.divi_processor import processor
from services.face_recognition.divi_matcher import matcher




@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(database_router)


app.include_router(face_recognition_router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
