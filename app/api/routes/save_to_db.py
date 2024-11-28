from datetime import datetime

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status

from core.config import settings
from schemas.data import Data, DataRead
from services.database.mongodb import db


router = APIRouter(prefix="/face", tags=["save_data"])


@router.post("", status_code=201)
async def save_data(data: Data):
    """
    Save data to MongoDB.
    """
    try:
        # Create complete document with timestamp
        document = DataRead(
            name=data.name,
            surname=data.surname,
            address=data.address,
            phone=data.phone,
            created_at=datetime.utcnow(),
        )

        await db.save_to_db(document.model_dump())  # Convert to dict for MongoDB

        return {"message": "Data saved successfully."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving data: {str(e)}",
        )


@router.get(
    "",
)
async def get_data():
    """
    Get all data from MongoDB.
    """
    try:
        data = await db.get_documents()
        return [
            {
                "name": doc["name"],
                "surname": doc["surname"],
                "address": doc["address"],
                "phone": doc["phone"],
            }
            for doc in data
        ]


    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting data: {str(e)}",
        )
