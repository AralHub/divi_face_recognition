from urllib.parse import urljoin

from fastapi import APIRouter, HTTPException

from services.database.mongodb import db
from services.face_recognition.divi_matcher import matcher

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
    await db.delete_collection(database)
    return {"message": f"Database {database} deleted successfully"}



@router.get("/get_faces")
async def get_faces(database: str):
    try:
        # Получаем количество документов в коллекции
        count = await db.count_documents(database)
        return {"count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error counting faces: {str(e)}")
