from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from src.azgems.Services.SupplierRanking import SupplierRanking
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/supplier-ranking")
async def supplier_ranking(
    customer_name: str = Query("Walmart", description="Customer name"),
) -> JSONResponse:
    try:
        supplier_ranking = SupplierRanking()
        ranked = supplier_ranking.calculate_topsis_score()

        return JSONResponse(status_code=status.HTTP_200_OK, content=ranked)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
