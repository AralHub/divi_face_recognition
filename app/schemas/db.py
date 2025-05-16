from typing import Dict, List, Optional, Any

from pydantic import BaseModel


class SaveToDB(BaseModel):

    person_id: int
    template_data: str  # base64 шаблон для 3DiVi

