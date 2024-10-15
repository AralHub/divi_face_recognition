import io
import os
from datetime import datetime

from PIL import Image
from fastapi import APIRouter, File, UploadFile, Form, HTTPException

from app.config import settings
from app.services.database import db
from app.services.embedding import model
from app.services.face_recognition import face_recognition

router = APIRouter()


@router.post("/recognize", tags=["face_recognition"])
async def recognize_face(file: UploadFile = File(...), database: str = Form(...)):
    if database not in db.get_collections_names():
        raise HTTPException(status_code=400, detail="Invalid database name")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    face_data = model.get_faces(image)
    if face_data is None:
        return {"error": "No face detected in the image"}

    score, person_id = face_recognition.search_face(face_data, database)

    return {"match": person_id, "database": database, "similarity": score}


@router.post("/add_face", tags=["face_recognition"])
async def add_face(
    file: UploadFile = File(...), database: str = Form(...), person_id: str = Form(...)
):
    if database not in db.get_collections_names():
        raise HTTPException(status_code=400, detail="Invalid database name")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    face_data = model.get_faces(image)
    if face_data is None:
        raise HTTPException(status_code=401, detail="No face detected in the image")

    score, person_id = face_recognition.search_face(face_data, database)
    if score > 99:
        return {"message": "face already exists", "score": score, "id": person_id}

    # Генерируем уникальное имя файла
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_extension = os.path.splitext(file.filename)[1]
    image_filename = f"{person_id}_{timestamp}{file_extension}"

    # Создаем директорию для базы данных, если она не существует
    database_dir = os.path.join(settings.UPLOAD_DIR, database)
    os.makedirs(database_dir, exist_ok=True)

    # Полный путь к файлу
    file_path = os.path.join(database_dir, image_filename)

    # Сохраняем изображение
    with open(file_path, "wb") as image_file:
        image_file.write(contents)
    score, person_id = face_recognition.search_face(face_data, database)
    if score > 99:
        return {"message": "face already exists", "score": score, "id": person_id}
    image_id = db.add_new_face_to_collection(face_data, file_path, person_id, database)

    image_url = f"/uploads/{database}/{image_filename}"

    face_recognition.add_to_index(face_data.embedding, database, person_id)

    return {
        "message": "Face added successfully",
        "id": str(image_id),
        "image_url": image_url,
    }


@router.post("/add_database", tags=["database"])
async def add_database(database: str):
    if database in db.get_collections_names():
        raise HTTPException(status_code=401, detail="such a base already exists ")

    face_recognition.create_index(database)
    return {"message": f"Database {database} added successfully"}


@router.get("/get_database_names", tags=["database"])
async def get_database_names():
    return {"database_names": db.get_collections_names()}


@router.post("/delete_database", tags=["database"])
async def delete_database(database: str):
    if database not in db.get_collections_names():
        raise HTTPException(status_code=404, detail="database not found")

    face_recognition.delete_index(database)
    db.delete_collection(database)
    return {"message": f"Database {database} deleted successfully"}


@router.get("update_indexes", tags=["database"])
async def update_index():
    face_recognition.update_indexes()
    return {"message": "Indexes updated successfully"}


@router.post("/delete_face", tags=["face_recognition"])
async def delete_face(database: str, face_id: str):
    if database not in db.get_collections_names():
        raise HTTPException(status_code=404, detail="database not found")

    if face_recognition.delete_face_from_index(database, face_id):
        db.delete_face_from_collection(database, face_id)
        return {"message": f"Face with id {face_id} deleted successfully"}
    else:
        return {"error": f"Face with id {face_id} not found"}


@router.post("/background_photo", tags=["face_recognition"])
async def background_photo(
    background_file: UploadFile = File(...),
    snap_file: UploadFile = File(...),
    score: str = Form(...),
):
    background_contents = await background_file.read()
    background_image = Image.open(io.BytesIO(background_contents))
    background_image_filename = background_file.filename
    image_path = os.path.join("images", "background_images", background_image_filename)
    upload_file = os.path.join(
        "uploads", "background_images", background_image_filename
    )
    snap_contents = await snap_file.read()
    snap_image = Image.open(io.BytesIO(snap_contents))

    face_data = model.get_faces(snap_image)
    if face_data is None:
        return {"error": "No face detected in the image"}

    image_path = model.send_background(
        background_image, face_data.embedding, score, image_path
    )

    return {"image_path": upload_file}


@router.get("/get_faces", tags=["database"])
async def get_faces(database: str):
    docs = db.get_docs_from_collection(database)
    results = [(doc["person_id"], doc["image_path"]) for doc in docs]
    return results
