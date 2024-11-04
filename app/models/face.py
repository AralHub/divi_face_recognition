from typing import List

from pydantic import BaseModel


class FaceMeta(BaseModel):
    age: int
    gender: str
    pose: List[float]
    det_score: float


class FaceMetadata(FaceMeta):
    embedding: str


class FaceEmbedding(BaseModel):
    metadata: FaceMetadata
    person_id: int
    image_path: str


class FaceInfo(BaseModel):
    face_id: str
    person_id: int
    image_path: str
    metadata: FaceMeta


class FaceSearchResult(BaseModel):
    person_id: int
    similarity: float
    database: str
