from pydantic import BaseModel

from schemas.face_meta import FaceMetadata


class SaveToDB(BaseModel):
    person_id: int
    key: str
    embedding: list[float]
    metadata: FaceMetadata


class GetImages(BaseModel):
    database: str
