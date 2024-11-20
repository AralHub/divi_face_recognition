from datetime import datetime

from pydantic import BaseModel
from sympy.strategies import canon


class Data(BaseModel):
    name: str
    surname: str
    address: str
    phone: str


class DataRead(Data):
    created_at: datetime
