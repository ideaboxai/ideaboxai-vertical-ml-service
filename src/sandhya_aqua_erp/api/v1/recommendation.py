from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("")
async def get_recommendation():
    return JSONResponse(content={"message": "Recommendation API is working properly"})
