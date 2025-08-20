from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import numpy as np
from enum import Enum
from typing import List, Optional

from src.sandhya_aqua_erp.repositories.yield_repo import YieldRepository

from src.sandhya_aqua_erp.anomaly_detection.supply_chain.pipeline.predict.predict_cross_stage import (
    predict,
)

router = APIRouter()


class CrossStageName(str, Enum):
    GRN_GRADING = "grn_grading_yield_data"
    SOAKING = "soaking_yield_data"
    PACKING = "packing_yield_data"


@router.get("/yield-data")
async def get_anomaly(
    interval: str = "1 DAY",
    exact_date_time: str = None,
    cross_stage_names: Optional[List[CrossStageName]] = Query(
        default=["grn_grading_yield_data", "soaking_yield_data", "packing_yield_data"],
        description="List of cross stage names to filter the yield data",
    ),
):
    """
    Endpoint to trigger yield anomaly detection.
    """
    yield_repo = YieldRepository()

    stage_method_map = {
        "grn_grading_yield_data": yield_repo.get_grn_grading_yield_data,
        "soaking_yield_data": yield_repo.get_soaking_yield_data,
        "packing_yield_data": yield_repo.get_packing_yield_data,
    }

    selected_stages = (
        [
            stage.value if isinstance(stage, CrossStageName) else stage
            for stage in cross_stage_names
        ]
        if cross_stage_names is not None
        else list(stage_method_map.keys())
    )

    yield_data_mapper = {
        stage: stage_method_map[stage](
            interval=interval, exact_date_time=exact_date_time
        )
        for stage in selected_stages
        if stage in stage_method_map
    }

    response = {}

    for cross_stage_name, queried_data in yield_data_mapper.items():
        df = queried_data
        predictions = predict(cross_stage_name=cross_stage_name, data=df)

        for col in predictions.columns:
            if predictions[col].dtypes.__class__.__name__ == "DatetimeTZDtype" or str(
                predictions[col].dtypes
            ).startswith("datetime"):
                predictions[col] = predictions[col].astype(str)

        predictions = predictions.replace([np.nan, np.inf, -np.inf], None)

        response[cross_stage_name] = predictions.to_dict(orient="records")
        print(
            f"Processed {cross_stage_name} with {len(response[cross_stage_name])} records."
        )

    return JSONResponse(content=response)
