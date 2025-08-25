from pydantic import BaseModel
from typing import Optional


class RequestModel(BaseModel):
    lot_number: Optional[str] = None
    sale_order: str

class ResponseModel(BaseModel):
    recommendation: str
