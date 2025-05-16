
from fastapi import APIRouter

router = APIRouter(tags=["crud_person"])

@router.get("/")
async def get_person():
    return {"message": "Hello World"}