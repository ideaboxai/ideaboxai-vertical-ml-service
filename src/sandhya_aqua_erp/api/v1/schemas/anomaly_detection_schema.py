from enum import Enum
from typing import List
from pydantic import BaseModel, Field


# class CrossStageName(str, Enum):
#     GRN_GRADING = "grn_grading_yield_data"
#     SOAKING = "soaking_yield_data"
#     PACKING = "packing_yield_data"


class AlertStage(str, Enum):
    GRADING_STAGE = "grn_grading_yield_data"
    SOAKING_STAGE = "soaking_yield_data"
    COOKING_STAGE = "cooking_yield_data"
    PACKING_STAGE = "packing_yield_data"


class DateOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "notEquals"
    BEFORE_DATE = "beforeDate"
    BEFORE_OR_ON_DATE = "beforeOrOnDate"
    AFTER_DATE = "afterDate"
    AFTER_OR_ON_DATE = "afterOrOnDate"
    IN_DATE_RANGE = "inDateRange"
    NOT_IN_DATE_RANGE = "notInDateRange"


class AlertProcessing(BaseModel):
    alert_stage: List[AlertStage] = Field(
        ..., min_length=1, max_length=255, description="Current alert stage"
    )
    timestamps: str = Field(
        ...,
        description="Timestamps of the alert like ['2023-10-01', '2023-10-05'] or ['2023-10-01']",
    )
    operator: DateOperator = Field(
        ...,
        description="Operator like equal afterDate, afterOrOnDate, notInDateRange, inDateRange, notEquals, equals,beforeDate, beforeOrOnDate",
    )
