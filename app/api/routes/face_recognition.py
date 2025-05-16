from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Form
from typing import Dict, List, Any
import json
import time
import logging
from services.database.localdb import db
from services.face_recognition.divi_processor import processor
from services.face_recognition.divi_matcher import matcher
from schemas.face_meta import (
    Recognize,
    ResponseRecognize,
    FaceMeta,
    PersonDelete,
    AddToDB,
)

router = APIRouter(tags=["face_recognition"])

logger = logging.getLogger(__name__)


@router.post("/recognize", response_model=ResponseRecognize)
async def recognize_face(recognize: Recognize):
    """
    Распознавание лица по изображению в указанной базе данных
    """
    total_start = time.time()

    # Проверка существования базы данных
    db_start = time.time()
    # collections = await db.get_collections_names()
    # if recognize.database not in collections:
    #     raise HTTPException(status_code=404, detail="База данных не найдена")
    db_check_time = time.time() - db_start
    print(f"Проверка базы данных: {db_check_time:.4f} секунд")

    # Обработка изображения
    process_start = time.time()
    template_base64, face_meta = await processor.process_image(recognize.image_path)
    process_time = time.time() - process_start
    print(f"Обработка изображения: {process_time:.4f} секунд")

    # Получаем все документы из указанной коллекции
    db_get_start = time.time()
    documents = await db.get_documents_limit(recognize.database, recognize.limit)
    db_get_time = time.time() - db_get_start
    print(f"Получение документов из БД: {db_get_time:.4f} секунд")

    if not documents:
        raise HTTPException(status_code=404, detail="В базе нет зарегистрированных лиц")

    print(f"Количество документов в базе: {len(documents)}")

    # Подготавливаем шаблоны и идентификаторы для добавления в индекс
    prepare_start = time.time()
    templates = []
    uuids = []
    doc_map = {}  # Для сопоставления uuid с документами

    for doc in documents:
        uuid = str(doc["_id"])
        templates.append(doc["face_template"])
        uuids.append(uuid)
        doc_map[uuid] = doc
    prepare_time = time.time() - prepare_start
    print(f"Подготовка данных: {prepare_time:.4f} секунд")

    # Сбрасываем индекс перед новым поиском
    reset_start = time.time()
    await matcher.reset_index(uuids=uuids)  # Убираем параметр uuids
    reset_time = time.time() - reset_start
    print(f"Сброс индекса: {reset_time:.4f} секунд")

    # Добавляем шаблоны в индекс
    add_start = time.time()


    await matcher.add_templates(templates, uuids)
  

    add_time = time.time() - add_start
    print(f"Добавление шаблонов в индекс: {add_time:.4f} секунд")

    # Выполняем поиск
    search_start = time.time()
    search_results = await matcher.search_face(template_base64)
    search_time = time.time() - search_start
    print(f"Поиск лица: {search_time:.4f} секунд")

    if not search_results:
        raise HTTPException(status_code=404, detail="Лицо не найдено в базе")

    # Формируем ответ
    result_start = time.time()
    # Получаем лучшее совпадение
    best_match = search_results[0]
    doc = doc_map[best_match["uuid"]]

    # Формируем ответ
    result = ResponseRecognize(
        person_id=doc["person_id"],
        image_path=recognize.image_path,
        template_data=template_base64,
        metadata=FaceMeta.model_validate(face_meta),
        similarity=best_match["score"],
    )
    result_time = time.time() - result_start
    print(f"Формирование ответа: {result_time:.4f} секунд")

    total_time = time.time() - total_start
    print(f"Общее время распознавания: {total_time:.4f} секунд")

    return result


@router.post("/add_person", response_model=AddToDB)
async def add_person(person_data: AddToDB):
    """
    Добавление человека в базу данных
    """
    # Проверка существования базы данных
    collections = await db.get_collections_names()
    if person_data.database not in collections:
        # Создаем новую коллекцию, если она не существует
        await db.db.create_collection(person_data.database)

    # Создаем документ для добавления в БД
    doc = {
        "person_id": person_data.person_id,
        "face_template": person_data.template_data,
    }

    # Добавляем лицо в коллекцию
    await db.add_face_to_collection(person_data.database, doc)

    return person_data


@router.delete("/delete_person")
async def delete_person(person_data: PersonDelete):
    """
    Удаление человека из базы данных
    """
    # Проверка существования базы данных
    collections = await db.get_collections_names()
    if person_data.database not in collections:
        raise HTTPException(status_code=404, detail="База данных не найдена")

    # Удаляем все записи для указанного person_id
    result = await db.delete_person(person_data.database, person_data.person_id)

    if not result:
        raise HTTPException(status_code=404, detail="Человек не найден в базе")

    return {"success": True, "message": "Человек успешно удален из базы"}
