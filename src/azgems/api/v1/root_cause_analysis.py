from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from src.azgems.Services.RootCauseAnalysis import RootCauseAnalysisService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/recommendations")
async def get_alert_analysis(
    anomaly_id: str = Query(..., description="Anomaly ID to analyze"),
) -> JSONResponse:
    try:
        rca_service = RootCauseAnalysisService()
        data = rca_service.get_the_data(anomaly_id=anomaly_id)
        if not data:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "message": f"No data found for anomaly ID {anomaly_id}",
                    "analysis": {},
                },
            )
        analysis = await rca_service.perform_root_cause_analysis(data)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "analysis": analysis,
                "message": "Analysis generated successfully.",
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"message": str(e), "analysis": {}}
        )
