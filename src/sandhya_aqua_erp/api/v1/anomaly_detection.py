from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("")
async def get_anomaly():
    return JSONResponse(content={"message": "Soaking yield anomaly detected properly."})
