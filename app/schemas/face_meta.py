from pydantic import BaseModel


class FaceMetadata(BaseModel):
    age: int
    gender: str
    pose: list[float]


class FaceResponse(FaceMetadata):
    face_id: int
    image_path: str
    person_id: int
