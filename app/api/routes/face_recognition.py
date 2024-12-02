from fastapi import APIRouter, Form, HTTPException, status

from core.exceptions import InvalidDatabase
from schemas.db import SaveToDB
from schemas.face_meta import Recognize, PersonDelete, ImageDelete
from schemas.face_meta import ResponseRecognize, AddToDB
from services.database.db_template import template_db
from services.database.mongodb import db
from services.face_recognition.matcher import matcher
from services.face_recognition.processor import processor
from services.file_storage.async_s3_manager import s3_manager

router = APIRouter(prefix="/face", tags=["face_recognition"])


@router.post("/recognize", status_code=status.HTTP_200_OK)
async def recognize_face(recognize: Recognize):
    if face_data := await template_db.get_face_data(recognize.photo_key):
        embedding, metadata = face_data.embedding, face_data.metadata
    else:
        face_data = await processor.process_image(recognize.photo_key)
        embedding, metadata = face_data.embedding, face_data.metadata

    if recognize.database not in await db.get_collections_names():
        return ResponseRecognize(person_id=0, similarity=0, metadata=metadata)

    score, person_id = await matcher.search(recognize.database, embedding)

    return ResponseRecognize(person_id=person_id, similarity=score, metadata=metadata)


@router.post("/add", status_code=status.HTTP_201_CREATED)
async def add_face(new_face: AddToDB):
    if face_data := await template_db.get_face_data(new_face.photo_key):
        embedding, metadata = face_data.embedding, face_data.metadata
    else:
        face_data = await processor.process_image(new_face.photo_key)
        embedding, metadata = face_data.embedding, face_data.metadata

    # Добавление в базу данных
    face_doc = SaveToDB(
        person_id=new_face.person_id,
        key=new_face.photo_key,
        embedding=embedding,
        metadata=metadata,
    )

    await db.add_face_to_collection(new_face.database, face_doc.model_dump())

    await matcher.add_face(new_face.database, embedding, new_face.person_id)

    return face_doc.model_dump(exclude={"embedding"})


@router.post("/get_background_image")
async def get_background_image(background_image: str, snap_image: str):

    data = await processor.process_background_image(
        snap_key=snap_image, background_key=background_image
    )
    return data


@router.post("/delete_person", status_code=status.HTTP_200_OK)
async def delete_person(data: PersonDelete):
    if data.database not in await db.get_collections_names():
        raise InvalidDatabase

    # Удаление из базы данных
    result = await db.delete_person(data.database, data.person_id)
    if not result:
        raise HTTPException(status_code=404, detail="Person not found")

    await matcher.delete_face(data.database, data.person_id)

    return {"message": f"Person with ID {data.person_id} deleted successfully"}


@router.delete("/delete_image", status_code=status.HTTP_200_OK)
async def delete_image(data: ImageDelete):
    if data.database not in await db.get_collections_names():
        raise InvalidDatabase
    image_data = await db.get_image_by_key(data.database, data.image_key)
    if image_data is None:
        raise HTTPException(status_code=404, detail="Image not found")
    result = await db.delete_image_by_key(data.database, data.image_key)
    if not result:
        raise HTTPException(status_code=404, detail="Image not found")

    await matcher.update_collection_index(data.database)
    return {"message": f"Image with URL {data.image_key} deleted successfully"}
