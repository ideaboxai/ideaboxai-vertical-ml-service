from fastapi import FastAPI
from src.sandhya_aqua_erp.api.v1.anomaly_detection import (
    router as anomaly_detection_router,
)
from src.sandhya_aqua_erp.api.v1.recommender import app as recommendation_router


def setup_routes(app: FastAPI):
    app.include_router(
        anomaly_detection_router,
        prefix="/api/v1/sandhya/anomaly-detection",
        tags=["Sandhya Anomaly Detection"],
    )
    app.include_router(
        recommendation_router,
        prefix="/api/v1/sandhya/recommendation",
        tags=["Sandhya Recommendation"],
    )
