import json
import redis
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.sandhya_aqua_erp.services.llm_recommender_service import OpenAIRecommender
from src.sandhya_aqua_erp.api.v1.schemas.recommender_schema import RequestModel
from src.sandhya_aqua_erp.services.cube_query_service import get_data_from_cube_query
import asyncio

app = APIRouter()

# Redis client
redis_client = redis.Redis(host="ml-service-redis", port=6379, decode_responses=True)
# redis_client = redis.Redis(host="0.0.0.0", port=6379, decode_responses=True)


def ensure_sse_format(chunk: str) -> str:
    """Ensure the chunk is SSE formatted."""
    if not chunk.startswith("data:"):
        return f"{chunk}"
    return chunk


@app.post("/recommend")
async def recommend(request: RequestModel):
    lot_number = request.lot_number
    sale_order = request.sale_order

    cache_key = f"recommend:{lot_number}:{sale_order}"
    cached_data = redis_client.get(cache_key)

    mode = "normal"  # or "stream"
    recommender = OpenAIRecommender(mode=mode)

    if cached_data:
        if mode == "stream":

            async def replay_cached():
                for chunk in json.loads(cached_data):
                    yield ensure_sse_format(chunk)

            return StreamingResponse(replay_cached(), media_type="text/event-stream")
        else:
            return {"recommendation": json.loads(cached_data)}

    user_prompt = "Give Recommendation for the lot number"
    chat_history = []
    structured_input = f"User Query: {user_prompt}" if user_prompt else "User Query:"

    (
        grn_process,
        grading_process,
        soaking_process,
        cooking_process,
        yield_process,
    ) = await asyncio.gather(
        get_data_from_cube_query("grn_process_query", lot_number, sale_order),
        get_data_from_cube_query("grading_process_query", lot_number, sale_order),
        get_data_from_cube_query("soaking_process_query", lot_number, sale_order),
        get_data_from_cube_query("cooking_process_query", lot_number, sale_order),
        get_data_from_cube_query("yield_calculation_query", lot_number, sale_order),
    )

    parameters = {
        "grn_process_parameters": grn_process,
        "grading_process_parameters": grading_process,
        "soaking_process_parameters": soaking_process,
        "cooking_process_parameters": cooking_process,
        "grading_yield_parameters": yield_process,
    }

    if mode == "stream":
        # Await the coroutine to get the async generator
        response_stream = await recommender.get_recommendation(
            structured_input, chat_history, parameters=parameters
        )

        async def caching_stream():
            buffer = []
            async for chunk in response_stream:  # iterate over the async generator
                formatted_chunk = ensure_sse_format(chunk)
                buffer.append(formatted_chunk)
                yield formatted_chunk
            redis_client.setex(cache_key, 24 * 60 * 60, json.dumps(buffer))

        return StreamingResponse(caching_stream(), media_type="text/event-stream")

    else:
        result = await recommender.get_recommendation(
            structured_input, chat_history, parameters=parameters
        )
        redis_client.setex(cache_key, 24 * 60 * 60, json.dumps(result))
        return {"recommendation": result}
