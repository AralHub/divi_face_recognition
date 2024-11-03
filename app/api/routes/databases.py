from fastapi import APIRouter, HTTPException

from models.face import FaceInfo
from services.database.mongodb import db
from services.face_recognition.matcher import matcher

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

    matcher.delete_collection_index(database)
    await db.delete_collection(database)
    return {"message": f"Database {database} deleted successfully"}


@router.post("/update_indexes")
async def update_index(database: str):
    matcher.update_collection_index(database)
    return {"message": "Indexes updated successfully"}


@router.get("/get_faces")
async def get_faces(database: str):
    results = []
    try:
        async for doc in db.get_docs_from_collection(database):
            results.append(FaceInfo(
                face_id=str(doc["_id"]),
                person_id=doc["person_id"],
                image_path=doc["image_path"],
                metadata=doc.get("metadata")
            ))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting faces: {str(e)}")
