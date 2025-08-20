# services/llm_recommender.py
from typing import AsyncGenerator, List, Dict, Any
import os
from openai import AsyncOpenAI

from sandhya_aqua_erp.utils.prompt import (
    Recommendation_system_prompt,
    Recommendation_user_prompt,
)
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()


langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


class OpenAIRecommender:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        # Use the ASYNC client
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def get_recommendation_stream(
        self,
        user_input: str,
        history: List[Dict[str, Any]],
        parameters: Dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:

        parameters = parameters or {}
        finalized_prompt = Recommendation_user_prompt.format(
            **parameters, user_query=user_input
        )

        try:
            # Async streaming with the Responses API
            with langfuse.start_as_current_span(
                name="Sandhya Aqua Recommender",
            ) as span:
                span.update(
                    input=finalized_prompt,
                )
                async with self.client.responses.stream(
                    model=self.model,
                    input=finalized_prompt,
                    instructions=Recommendation_system_prompt,
                    metadata={"model": self.model},
                ) as stream:
                    async for event in stream:
                        if event.type == "response.output_text.delta":
                            yield f"data: {event.delta}\n\n"

                final = await stream.get_final_response()
                metadata: Dict[str, Any] = {}
                if final.usage:
                    metadata.update(
                        {
                            "prompt_tokens": final.usage.input_tokens,
                            "completion_tokens": final.usage.output_tokens,
                            "total_tokens": final.usage.total_tokens,
                        }
                    )
                span.update(
                    output=final.output_text,
                    metadata=metadata,
                )
                langfuse.update_current_generation(
                    usage_details={
                        "input": final.usage.input_tokens,
                        "output": final.usage.output_tokens,
                        "total": final.usage.total_tokens,
                        "prompt_tokens": final.usage.input_tokens,
                        "completion_tokens": final.usage.output_tokens,
                        "total_tokens": final.usage.total_tokens,
                    },
                    model=self.model,
                )
        except Exception as e:
            yield f"Error: {str(e)}"
