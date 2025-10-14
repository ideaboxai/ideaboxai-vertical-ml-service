from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from src.azgems.Services.TrainModel import TrainModel
from src.azgems.Services.Inference import ModelInference as Inference
from datetime import date
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/get-all-shipment-classifications")
def get_all_shipment_classifications(
    customer_name: str = Query("Walmart", description="Customer name"),
    start_timestamps: date = Query(
        ..., description="Start timestamp(s) in the format of 2025-11-12"
    ),
    end_timestamps: date = Query(
        ..., description="End timestamp(s) in the format of 2025-11-12"
    ),
    po_comitted: str = Query("Direct Sale", description="PO committed type"),
) -> JSONResponse:
    try:
        folder_path_to_check = os.path.join(
            "models", customer_name, "logistic_regression.pkl"
        )

        if not os.path.exists(folder_path_to_check):
            logger.info("Weights Not Found...Training the Model for the customers")
            trainer = TrainModel(customer_name=customer_name, po_comitted=po_comitted)
            trainer.train_and_save_model()

        inference_engine = Inference(
            customer_name=customer_name,
            start_timestamp=start_timestamps,
            end_timestamp=end_timestamps,
        )
        predictions = inference_engine.get_data_for_inference_from_cube()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "customer_name": customer_name,
                "po_comitted": po_comitted,
                "predictions": predictions,
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
