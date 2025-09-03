from fastapi import APIRouter
from fastapi.responses import JSONResponse
from src.sandhya_aqua_erp.services.llm_recommender_service import (
    OpenAIRecommender,
)
from pydantic import BaseModel

router = APIRouter()


class RootCauseAnalysisRequest(BaseModel):
    data: dict


@router.post("/root-cause-analysis")
async def get_recommendation(request: RootCauseAnalysisRequest):
    llm_recommender = OpenAIRecommender()
    root_cause_analysis = await llm_recommender.get_root_cause_analysis(
        user_input=request.data
    )
    return JSONResponse(content={"root_cause": root_cause_analysis})
