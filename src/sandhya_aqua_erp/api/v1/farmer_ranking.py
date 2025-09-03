from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse

from typing import Optional, Literal

from src.sandhya_aqua_erp.farmer_ranking.pipeline_topsis import (
    run_topsis_for_farmer_ranking,
)
from src.sandhya_aqua_erp.utils.param_input_validator import param_input_validator

from fastapi import HTTPException

router = APIRouter()


@router.get("/trigger")
async def trigger_farmer_ranking(
    time_format: Literal["EXACT", "INTERVAL"] = "INTERVAL",
    time_value: Optional[str] = Query(
        None,
        description="Exact date-time in 'YYYY-MM-DD HH:MM:SS' format if time_format is 'EXACT', else interval like '1 MONTH' if time_format is 'INTERVAL'",
    ),
):
    try:
        if time_format == "EXACT":
            if not time_value:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="time_value must be provided when time_format is 'EXACT'.",
                )
            param_input_validator(time_value)
            run_topsis_for_farmer_ranking(exact_date_time=time_value)
        else:
            interval = time_value or "24 MONTH"
            param_input_validator(time_value)
            run_success = run_topsis_for_farmer_ranking(interval=interval)
            if not run_success:
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={
                        "message": "Failed to rank farmers and insert into database."
                    },
                )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Farmer ranking process completed successfully."},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred: {str(e)}",
        )
