from pydantic import BaseModel
class RequestModel(BaseModel):
    lot_number: str
    sale_order: str

class ResponseModel(BaseModel):
    recommendation: str
