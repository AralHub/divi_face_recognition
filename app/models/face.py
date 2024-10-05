from pydantic import BaseModel
from typing import List

class Face(BaseModel):
    name: str
    embedding: List[float]
