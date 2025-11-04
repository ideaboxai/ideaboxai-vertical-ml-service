from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from src.azgems.Services.TrainModel import TrainModel
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/trigger-training")
def trigger_training(
    customer_name: str = Query("Walmart", description="Customer name"),
) -> JSONResponse:
    try:
        model_path = os.path.join(
            "models", "azgems", customer_name, "ShipmentClassificationModel.joblib"
        )

        # ✅ If model exists, remove it before retraining
        if os.path.exists(model_path):
            logger.info(
                f"Existing model found for {customer_name}. Removing old model..."
            )
            os.remove(model_path)

        # ✅ Train and save new model
        logger.info(f"Training new model for customer: {customer_name}")
        trainer = TrainModel(customer_name=customer_name)
        trainer.run_full_pipeline()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "customer_name": customer_name,
                "message": "Model retrained and saved successfully",
            },
        )

    except Exception as e:
        logger.exception("Error during model retraining")
        raise HTTPException(status_code=500, detail=str(e))
