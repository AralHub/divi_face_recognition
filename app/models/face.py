from typing import List

from pydantic import BaseModel


class FaceMetadata(BaseModel):
    embedding: str
    age: int
    gender: str
    pose: List[float]


class FaceEmbedding(BaseModel):
    metadata: FaceMetadata
    person_id: int
    image_path: str


class FaceSearchResult(BaseModel):
    person_id: int
    similarity: float
    database: str
