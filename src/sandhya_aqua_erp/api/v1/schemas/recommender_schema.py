from pydantic import BaseModel
class RequestModel(BaseModel):
    lot_number: str


class ResponseModel(BaseModel):
    recommendation: str
