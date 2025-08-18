from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from sandhya_aqua.services.llm_recommender import OpenAIRecommender
from sandhya_aqua.services.query_database import run_predefined_query

_ = load_dotenv()

app = APIRouter()


class RequestModel(BaseModel):
    lot_number: str
    user_prompt: Optional[str] = None
    chat_history: Optional[List[str]] = []


class ResponseModel(BaseModel):
    reply: str


recommender = OpenAIRecommender(api_key=os.getenv("OPENAI_API_KEY"))


@app.post("/recommend", response_model=ResponseModel)
def recommend(request: RequestModel):
    lot_number = request.lot_number
    user_prompt = request.user_prompt or "Give Recommendation for the lot number"
    chat_history = request.chat_history or []

    structured_input = f"User Query: {user_prompt}" if user_prompt else "User Query:"

    parameters = {
        "grn_process_parameters": run_predefined_query(
            "grn_process_query", params=(lot_number,)
        ).to_string(index=False),
        "grading_process_parameters": run_predefined_query(
            "grading_process_query", params=(lot_number,)
        ).to_string(index=False),
        "soaking_process_parameters": run_predefined_query(
            "soaking_process_query", params=(lot_number, lot_number)
        ).to_string(index=False),
        "cooking_process_parameters": run_predefined_query(
            "cooking_process_query", params=(lot_number, lot_number)
        ).to_string(index=False),
        "grading_yield_parameters": run_predefined_query(
            "yield_calculation_query", params=(lot_number,)
        ).to_string(index=False),
        "packing_yield_parameters": run_predefined_query(
            "packing_yield_query", params=(lot_number,)
        ).to_string(index=False),
    }

    reply = recommender.get_recommendation(
        structured_input,
        chat_history,
        parameters=parameters,
    )

    return {"reply": reply}
