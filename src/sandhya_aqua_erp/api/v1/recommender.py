from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from src.sandhya_aqua_erp.services.llm_recommender import OpenAIRecommender
from src.sandhya_aqua_erp.repositories.recommendation_repo import run_predefined_query
from src.sandhya_aqua_erp.api.v1.schemas.recommender_schema import RequestModel

_ = load_dotenv()

app = APIRouter()



@app.post("/recommend")
async def recommend(request: RequestModel):
    lot_number = request.lot_number
    user_prompt = "Give Recommendation for the lot number"
    chat_history = []

    structured_input = f"User Query: {user_prompt}" if user_prompt else "User Query:"

    parameters = {
        "grn_process_parameters": (
            await run_predefined_query("grn_process_query", params=(lot_number,))
        ).to_string(index=False),
        "grading_process_parameters": (
            await run_predefined_query("grading_process_query", params=(lot_number,))
        ).to_string(index=False),
        "soaking_process_parameters": (
            await run_predefined_query(
                "soaking_process_query", params=(lot_number, lot_number)
            )
        ).to_string(index=False),
        "cooking_process_parameters": (
            await run_predefined_query(
                "cooking_process_query", params=(lot_number, lot_number)
            )
        ).to_string(index=False),
        "grading_yield_parameters": (
            await run_predefined_query("yield_calculation_query", params=(lot_number,))
        ).to_string(index=False),
        "packing_yield_parameters": (
            await run_predefined_query("packing_yield_query", params=(lot_number,))
        ).to_string(index=False),
    }
    recommender = OpenAIRecommender(api_key=os.getenv("OPENAI_API_KEY"))
    return StreamingResponse(
        recommender.get_recommendation_stream(
            structured_input,
            chat_history,
            parameters=parameters,
        ),
        media_type="text/event-stream",
    )
