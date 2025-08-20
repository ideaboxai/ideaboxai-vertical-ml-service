from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from sandhya_aqua.services.llm_recommender import OpenAIRecommender
from sandhya_aqua.services.query_database import run_predefined_query

_ = load_dotenv()

app = APIRouter()


class RequestModel(BaseModel):
    lot_number: str
    # stage_in_which_anomaly_is_detected: str   # These data must be taken from the database after the anomaly is detected and saved in database
    # anomaly_type: str
    # anomaly_context: str
    # user_prompt: Optional[str] = None


# anomaly_context = {"user_threshold": 10, "statistical_threshold": 15,"anomaly_value": 20}


class ResponseModel(BaseModel):
    recommendation: str


recommender = OpenAIRecommender(api_key=os.getenv("OPENAI_API_KEY"))


@app.post("/recommend")
async def recommend(request: RequestModel):
    lot_number = request.lot_number
    user_prompt = "Give Recommendation for the lot number"
    chat_history = []

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

    return StreamingResponse(
        recommender.get_recommendation_stream(
            structured_input,
            chat_history,
            parameters=parameters,
        ),
        media_type="text/event-stream",
    )
