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
    po_comitted: str = Query("Direct Sale", description="PO committed type"),
) -> JSONResponse:
    try:
        model_path = os.path.join("models", customer_name, "model.pkl")

        # ✅ If model exists, remove it before retraining
        if os.path.exists(model_path):
            logger.info(
                f"Existing model found for {customer_name}. Removing old model..."
            )
            os.remove(model_path)

        # ✅ Train and save new model
        logger.info(f"Training new model for customer: {customer_name}")
        trainer = TrainModel(customer_name=customer_name, po_comitted=po_comitted)
        trainer.train_and_save_model()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "customer_name": customer_name,
                "po_comitted": po_comitted,
                "message": "Model retrained and saved successfully",
            },
        )

    except Exception as e:
        logger.exception("Error during model retraining")
        raise HTTPException(status_code=500, detail=str(e))
