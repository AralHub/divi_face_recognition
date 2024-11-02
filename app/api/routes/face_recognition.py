from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from services.database.mongodb import db
from services.face_recognition.matcher import matcher
from services.face_recognition.processor import processor
from services.file_storage.async_storage import storage

router = APIRouter(prefix="/face", tags=["face_recognition"])


@router.post("/recognize")
async def recognize_face(file: UploadFile = File(...), database: str = Form(...)):
    if database not in await db.get_collections_names():
        raise HTTPException(status_code=400, detail="Invalid database")

    contents = await file.read()

    # Обработка изображения
    face_data = await processor.process_image(contents)
    if face_data is None:
        raise HTTPException(status_code=400, detail="No face detected")

    embedding, metadata = face_data

    # Поиск совпадений
    score, person_id = await matcher.search(database, embedding)

    return {
        "person_id": person_id,
        "similarity": float(score * 100),
        "metadata": metadata,
    }


@router.post("/add")
async def add_face(
    file: UploadFile = File(...), database: str = Form(...), person_id: int = Form(...)
):
    # if database not in await db.get_collections_names():
    #     raise HTTPException(status_code=400, detail="Invalid database")

    contents = await file.read()

    # Обработка изображения
    face_data = await processor.process_image(contents)
    if face_data is None:
        raise HTTPException(status_code=400, detail="No face detected")

    embedding, metadata = face_data

    # Сохранение файла
    filepath = await storage.save_file(contents, database, person_id)

    # Добавление в базу данных
    face_doc = {
        "person_id": person_id,
        "embedding": embedding.tolist(),
        "metadata": metadata,
        "image_path": filepath,
    }

    face_id = await db.add_face_to_collection(database, face_doc)

    # Добавление в индекс
    await matcher.add_face(database, embedding, person_id)

    return {
        "face_id": face_id,
        "image_path": filepath,
        "person_id": person_id,
        "metadata": metadata,
    }
