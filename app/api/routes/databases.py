from fastapi import APIRouter, HTTPException

from app.services.database.mongodb import db
from app.services.face_recognition.matcher import matcher

router = APIRouter(tags=["database"])


@router.get("/get_database_names")
async def get_database_names():
    # Await the asynchronous function to resolve the coroutine
    collections = await db.get_collections_names()
    return {"database_names": collections}


@router.post("/delete_database")
async def delete_database(database: str):
    collections_names = await db.get_collections_names()
    if database not in collections_names:
        raise HTTPException(status_code=404, detail="database not found")

    matcher.delete_index(database)
    await db.delete_collection(name=database)
    return {"message": f"Database {database} deleted successfully"}


@router.get("/update_indexes")
async def update_index():
    matcher.update_index()
    return {"message": "Indexes updated successfully"}


@router.get("/get_faces")
async def get_faces(database: str):
    docs = await db.get_docs_from_collection(database)
    results = [(doc["person_id"], doc["image_path"]) for doc in docs]
    return results
