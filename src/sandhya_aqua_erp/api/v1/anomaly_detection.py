from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse
import numpy as np
from src.sandhya_aqua_erp.utils.param_input_validator import param_input_validator
from src.sandhya_aqua_erp.repositories.yield_repo import YieldRepository
import ast

from src.sandhya_aqua_erp.anomaly_detection.supply_chain.pipeline.predict.predict_cross_stage import (
    predict,
)
from src.sandhya_aqua_erp.api.v1.schemas.anomaly_detection_schema import (
    AlertProcessing,
    AlertStage,
)

router = APIRouter()


@router.get("/yield-data")
async def get_anomaly(
    # interval: str = "1 DAY",
    # exact_date_time: str = None,
    # cross_stage_names: Optional[List[CrossStageName]] = Query(
    #     default=["grn_grading_yield_data", "soaking_yield_data", "packing_yield_data"],
    #     description="List of cross stage names to filter the yield data",
    # ),
    params: AlertProcessing = Query(...),
):
    """
    Endpoint to trigger yield anomaly detection.
    """
    # Validate input parameters
    try:
        print("Received params:", params)
        print("Received alert stage:", params.alert_stage)

        # Ensure timestamps is always a list
        timestamps = params.timestamps
        if not isinstance(timestamps, list):
            timestamps = ast.literal_eval(timestamps)
        else:
            raise ValueError("Timestamps must be a string or a list of strings.")

        timestamp1 = timestamps[0]
        timestamp2 = timestamps[1] if len(timestamps) > 1 else None

        print("Received timestamps:", timestamps)
        print("Received operator:", params.operator)

        param_input_validator(timestamp1)
        if timestamp2:
            param_input_validator(timestamp2)

        yield_repo = YieldRepository()

        stage_method_map = {
            "grn_grading_yield_data": yield_repo.get_grn_grading_yield_data,
            "soaking_yield_data": yield_repo.get_soaking_yield_data,
            "packing_yield_data": yield_repo.get_packing_yield_data,
        }

        selected_stages = (
            [
                stage.value if isinstance(stage, AlertStage) else stage
                for stage in params.alert_stage
            ]
            if params.alert_stage is not None
            else list(stage_method_map.keys())
        )

        yield_data_mapper = {
            stage: stage_method_map[stage](
                timestamp1=timestamp1,
                timestamp2=timestamp2,
                interval=None,
                operator=params.operator,
            )
            for stage in selected_stages
        }

        response = {}

        for cross_stage_name, queried_data in yield_data_mapper.items():
            df = queried_data
            predictions = predict(cross_stage_name=cross_stage_name, data=df)

            for col in predictions.columns:
                if predictions[
                    col
                ].dtypes.__class__.__name__ == "DatetimeTZDtype" or str(
                    predictions[col].dtypes
                ).startswith("datetime"):
                    predictions[col] = predictions[col].astype(str)

            predictions = predictions.replace([np.nan, np.inf, -np.inf], None)

            response[cross_stage_name] = predictions.to_dict(orient="records")
            print(
                f"Processed {cross_stage_name} with {len(response[cross_stage_name])} records."
            )

        return JSONResponse(status_code=status.HTTP_200_OK, content=response)

    except ValueError as ve:
        return JSONResponse(
            status_code=400, content={"error": f"Invalid input parameter: {str(ve)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"An error occurred: {str(e)}"}
        )
