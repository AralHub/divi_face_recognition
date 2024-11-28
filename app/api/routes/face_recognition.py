from urllib.parse import urljoin

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status

from core.config import settings
from services.database.mongodb import db
from services.face_recognition.matcher import matcher
from services.face_recognition.processor import processor

router = APIRouter(prefix="/face", tags=["face_recognition"])


@router.post("/recognize", status_code=status.HTTP_200_OK)
async def recognize_face(photo_key: str, database: str = Form(...)):

    face_data = await processor.process_image(photo_key)
    if face_data is None:
        raise HTTPException(status_code=400, detail="No face detected")

    embedding, metadata = face_data

    if database not in await db.get_collections_names():
        return {
            "person_id": 0,
            "similarity": 0,
            "metadata": metadata,
        }

    score, person_id = await matcher.search(database, embedding)

    return {
        "person_id": person_id,
        "similarity": float(score * 100),
        "metadata": metadata,
    }


@router.post("/add", status_code=status.HTTP_201_CREATED)
async def add_face(
    photo_key: str, database: str = Form(...), person_id: int = Form(...)
):

    face_data = await processor.process_image(photo_key)
    if face_data is None:
        raise HTTPException(status_code=400, detail="No face detected")

    embedding, metadata = face_data

    # Добавление в базу данных
    face_doc = {
        "person_id": person_id,
        "embedding": embedding.tolist(),
        "metadata": metadata,
        "image_path": filepath,
        "image_url": image_url,
    }

    face_id = await db.add_face_to_collection(database, face_doc)

    await matcher.add_face(database, embedding, person_id)

    return {
        "face_id": str(face_id),
        "image_path": filepath,
        "image_url": urljoin(
            f"{settings.MEDIA_URL}/", image_url
        ),  # Return URL in response
        "person_id": person_id,
        "metadata": metadata,
    }


@router.post(
    "/get_background_image",
)
async def get_background_image(
    background_image: UploadFile = File(...), snap_image: UploadFile = File(...)
):
    if background_image.filename is None or snap_image.filename is None:
        raise HTTPException(
            status_code=400, detail="No background or snap image provided"
        )

    # Обработка изображений
    background_contents = await background_image.read()
    snap_contents = await snap_image.read()

    data = await processor.background_image(
        snap_contents, background_contents, background_image.filename
    )
    return {
        "background_image_path": data["background_image_path"],
        "background_image_url": data["background_image_url"],
    }


@router.post("/delete_person", status_code=status.HTTP_200_OK)
async def delete_person(database: str = Form(...), person_id: int = Form(...)):
    if database not in await db.get_collections_names():
        raise HTTPException(status_code=400, detail="Invalid database")

    # Удаление из базы данных
    result = await db.delete_person(database, person_id)
    if not result:
        raise HTTPException(status_code=404, detail="Person not found")

    await matcher.delete_face(database, person_id)

    return {"message": f"Person with ID {person_id} deleted successfully"}


@router.delete("/delete_image", status_code=status.HTTP_200_OK)
async def delete_image(database: str = Form(...), image_url: str = Form()):
    if database not in await db.get_collections_names():
        raise HTTPException(status_code=400, detail="Invalid database")
    image_data = await db.get_image_by_url(database, image_url)
    if image_data is None:
        raise HTTPException(status_code=404, detail="Image not found")
    await storage.delete_file(image_data.get("image_path"))
    result = await db.delete_image_by_url(database, image_url)
    if not result:
        raise HTTPException(status_code=404, detail="Image not found")

    await matcher.update_collection_index(database)
    return {"message": f"Image with URL {image_url} deleted successfully"}
