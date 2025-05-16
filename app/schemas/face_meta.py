from enum import Enum
from typing import Dict

from pydantic import BaseModel


class Emotions(Enum):
    ANGRY = "ANGRY"
    DISGUSTED = "DISGUSTED"
    SCARED = "SCARED"
    HAPPY = "HAPPY"
    NEUTRAL = "NEUTRAL"
    SAD = "SAD"
    SURPRISED = "SURPRISED"

class Gender(Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"


class FaceMeta(BaseModel):
    age: int | None = None
    gender: Gender | None = None
    quality_score: float | None = None
    rotation: float | None = None
    eyes_distance: int | None = None
    emotions: Dict[Emotions, float] | None = None


class FaceResponse(BaseModel):
    person_id: int
    image_path: str
    template_data: str
    metadata: FaceMeta


class ResponseRecognize(FaceResponse):
    similarity: float


class Recognize(BaseModel):
    image_path: str
    database: str
    limit: int = 100

class AddToDB(BaseModel):
    person_id: int
    template_data: str
    database: str


class PersonDelete(BaseModel):
    database: str
    person_id: int

