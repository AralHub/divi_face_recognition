from fastapi import APIRouter, HTTPException, status
from typing import Tuple, Dict, Any
import asyncio

from core.exceptions import InvalidDatabase
from schemas.db import SaveToDB
from schemas.face_meta import (
    Recognize,
    PersonDelete,
    ImageDelete,
    ResponseRecognize,
    AddToDB,
    MovePerson,
)
from services.database.db_template import template_db
from services.database.mongodb import db
from services.face_recognition.matcher import matcher
from services.face_recognition.processor import processor
from services.file_storage.async_s3_manager import s3_manager

router = APIRouter(prefix="/face", tags=["face_recognition"])


async def get_face_data(photo_key: str) -> Tuple[list, Dict[str, Any]]:
    """Получение данных лица из кэша или обработка нового изображения."""
    if face_data := await template_db.get_face_data(photo_key):
        return face_data.embedding, face_data.metadata

    face_data = await processor.process_image(photo_key)
    return face_data.embedding, face_data.metadata


async def validate_database(database: str) -> None:
    """Проверка существования базы данных."""
    if database not in await db.get_collections_names():
        raise InvalidDatabase


@router.post("/recognize", status_code=status.HTTP_200_OK)
async def recognize_face(recognize: Recognize) -> ResponseRecognize:
    embedding, metadata = await get_face_data(recognize.photo_key)

    try:
        await validate_database(recognize.database)
    except Exception:
        return ResponseRecognize(person_id=0, similarity=0, metadata=metadata)

    score, person_id = await matcher.search(recognize.database, embedding)
    return ResponseRecognize(person_id=person_id, similarity=score, metadata=metadata)


@router.post("/add", status_code=status.HTTP_201_CREATED)
async def add_face(new_face: AddToDB):
    embedding, metadata = await get_face_data(new_face.photo_key)

    # await validate_database(new_face.database)

    face_doc = SaveToDB(
        person_id=new_face.person_id,
        key=new_face.photo_key,
        embedding=embedding,
        metadata=metadata,
    )

    await db.add_face_to_collection(new_face.database, face_doc.model_dump())
    await matcher.add_face(new_face.database, embedding, new_face.person_id)

    return face_doc.model_dump(exclude={"embedding"})


@router.post("/delete_person", status_code=status.HTTP_200_OK)
async def delete_person(data: PersonDelete):
    await validate_database(data.database)

    images = await db.get_images_by_person(data.database, data.person_id)
    if not images:
        raise HTTPException(status_code=404, detail="Person not found")

    # Удаление файлов и записей параллельно
    delete_tasks = [s3_manager.delete_file(image["key"]) for image in images]
    await asyncio.gather(*delete_tasks)

    await db.delete_person(data.database, data.person_id)
    await matcher.delete_person(data.database, data.person_id)

    return {"message": f"Person with ID {data.person_id} deleted successfully"}


@router.delete("/delete_image", status_code=status.HTTP_200_OK)
async def delete_image(data: ImageDelete):
    await validate_database(data.database)
    image_data = await db.get_image_by_key(data.database, data.image_key)
    if image_data is None:
        raise HTTPException(status_code=404, detail="Image not found")

    await asyncio.gather(
        db.delete_image_by_key(data.database, data.image_key),
        s3_manager.delete_file(key=data.image_key),
    )

    await matcher.delete_face(data.database, image_data["person_id"])
    return {"message": f"Image with URL {data.image_key} deleted successfully "}


@router.post("/move_person", status_code=status.HTTP_200_OK)
async def move_person(data: MovePerson):
    # Проверяем существование обеих коллекций
    await validate_database(data.source_database)
    await validate_database(data.target_database)

    # Получаем все изображения пользователя из исходной коллекции
    images = await db.get_images_by_person(data.source_database, data.person_id)
    if not images:
        raise HTTPException(status_code=404, detail="Человек не найден в исходной базе")

    # Добавляем лица в целевую коллекцию и обновляем индекс
    for image in images:
        face_doc = SaveToDB(
            person_id=data.person_id,
            key=image["key"],
            embedding=image["embedding"],
            metadata=image.get("metadata", {}),
        )

        # Добавляем в новую коллекцию
        await db.add_face_to_collection(data.target_database, face_doc.model_dump())

        # Обновляем FAISS индекс для целевой коллекции
        await matcher.add_face(data.target_database, image["embedding"], data.person_id)

    # Удаляем из исходного индекса
    await matcher.delete_person(data.source_database, data.person_id)

    # Удаляем из исходной коллекции
    await db.delete_person(data.source_database, data.person_id)

    return {
        "message": f"Человек с ID {data.person_id} успешно перемещен из {data.source_database} в {data.target_database}",
        "moved_images": len(images),
    }
