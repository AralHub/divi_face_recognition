from pydantic import BaseModel


class FaceMetadata(BaseModel):
    age: int
    gender: str
    pose: list[float]
    det_score: float


class FaceResponse(FaceMetadata):
    face_id: int
    image_path: str
    person_id: int


class ResponseRecognize(BaseModel):
    person_id: int
    similarity: float
    metadata: FaceMetadata


class Recognize(BaseModel):
    photo_key: str
    database: str


class AddToDB(Recognize):
    person_id: int


class TemplateFaceData(BaseModel):
    key: str
    embedding: list[float]
    metadata: FaceMetadata


class PersonDelete(BaseModel):
    database: str
    person_id: int


class ImageDelete(BaseModel):
    database: str
    image_key: str
