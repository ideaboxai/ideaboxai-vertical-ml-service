from fastapi import FastAPI

from src.sandhya_aqua_erp.api.v1.anomaly_detection import (
    router as anomaly_detection_router,
)
from src.sandhya_aqua_erp.api.v1.recommender import app as recommendation_router
from src.sandhya_aqua_erp.api.v1.root_cause_analysis import (
    router as root_cause_analysis_router,
)
from src.sandhya_aqua_erp.api.v1.farmer_ranking import router as farmer_ranking_router
from src.azgems.api.v1.shipment_classification import (
    router as shipment_classification_router,
)
from src.azgems.api.v1.trigger_training import (
    router as trigger_training_router,
)
from src.azgems.api.v1.supplier_ranking import (
    router as supplier_ranking_router,
)


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
    app.include_router(
        root_cause_analysis_router,
        prefix="/api/v1/sandhya/root-cause-analysis",
        tags=["Sandhya Recommendation"],
    )
    app.include_router(
        farmer_ranking_router,
        prefix="/api/v1/sandhya/farmer-ranking",
        tags=["Sandhya Farmer Ranking"],
    )

    app.include_router(
        shipment_classification_router,
        prefix="/api/v1/azgems",
        tags=["Shipment Classification"],
    )
    app.include_router(
        trigger_training_router,
        prefix="/api/v1/azgems",
        tags=["Trigger Training"],
    )
    app.include_router(
        supplier_ranking_router,
        prefix="/api/v1/azgems",
        tags=["Rank Suppliers"],
    )
