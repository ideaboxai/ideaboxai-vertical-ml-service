from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse
from src.azgems.Services.TrainModel import TrainModel
from src.azgems.Services.Inference import Inference
from datetime import date
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/get-all-shipment-classifications")
async def get_all_shipment_classifications(
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
        if start_timestamps > end_timestamps:
            raise HTTPException(
                status_code=400,
                detail="start_timestamps must be earlier than or equal to end_timestamps",
            )
            
        folder_path_to_check = os.path.join(
            "models", "azgems", customer_name, "ShipmentClassificationModel.joblib"
        )

        if not os.path.exists(folder_path_to_check):
            logger.info("Weights Not Found...Training the Model for the customers")
            print("Weights Not Found...Training the Model for the customers")
            trainer = TrainModel(customer_name=customer_name)
            trainer.run_full_pipeline()

        inference_engine = Inference(
            customer_name=customer_name,
            start_timestamp=start_timestamps,
            end_timestamp=end_timestamps,
        )
        predictions = await inference_engine.run_batch_and_format_json()

        return JSONResponse(status_code=status.HTTP_200_OK, content=predictions)

    except Exception as e:
        err_msg = str(e)
        truncated_msg = err_msg[:100] + "..." if len(err_msg) > 100 else err_msg
        raise HTTPException(status_code=500, detail=truncated_msg)
