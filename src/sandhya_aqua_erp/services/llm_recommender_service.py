# services/llm_recommender.py
from typing import AsyncGenerator, List, Dict, Any, Literal, Union
import os
from openai import AsyncOpenAI, OpenAI
from langfuse import Langfuse
from dotenv import load_dotenv

from src.sandhya_aqua_erp.utils.prompt import (
    Recommendation_system_prompt,
    Recommendation_user_prompt,
    Root_cause_analysis_system_prompt,
    Root_cause_analysis_user_prompt,
)

load_dotenv()

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


class OpenAIRecommender:
    def __init__(
        self,
        api_key: str = os.getenv("OPENAI_API_KEY"),
        model: str = "gpt-4o-mini",
        mode: Literal["stream", "normal"] = "normal",
    ):
        self.mode = mode
        self.client = AsyncOpenAI(api_key=api_key) if mode == "stream" else OpenAI(api_key=api_key)
        self.model = model

    async def get_recommendation(
        self,
        user_input: str,
        history: List[Dict[str, Any]],
        parameters: Dict[str, Any] | None = None,
    ) -> Union[AsyncGenerator[str, None], str]:
        """
        - mode == 'stream': returns async generator
        - mode == 'normal': returns full string
        """
        parameters = parameters or {}
        finalized_prompt = Recommendation_user_prompt.format(
            **parameters, user_query=user_input
        )

        try:
            with langfuse.start_as_current_span(name="Sandhya Aqua Recommender") as span:
                span.update(input=Recommendation_system_prompt + "\n\n" + finalized_prompt)

                if self.mode == "stream":
                    async def stream_generator():
                        async with self.client.responses.stream(
                            model=self.model,
                            input=finalized_prompt,
                            instructions=Recommendation_system_prompt,
                            metadata={"model": self.model},
                        ) as stream:
                            async for event in stream:
                                if event.type == "response.output_text.delta":
                                    yield event.delta
                            final = await stream.get_final_response()
                            self._log_usage(span, final)
                    return stream_generator()
                else:
                    response = self.client.responses.create(
                        model=self.model,
                        input=finalized_prompt,
                        instructions=Recommendation_system_prompt,
                        metadata={"model": self.model},
                    )
                    self._log_usage(span, response)
                    return response.output_text

        except Exception as e:
            return f"Error: {str(e)}"

    async def get_root_cause_analysis(
        self,
        user_input: str,
    ) -> str:
        finalized_prompt = Root_cause_analysis_user_prompt.format(user_query=user_input)

        try:
            with langfuse.start_as_current_span(name="Sandhya Aqua Root Cause Analysis") as span:
                span.update(input=Root_cause_analysis_system_prompt + "\n\n" + finalized_prompt)

                if self.mode == "stream":
                    async with self.client.responses.stream(
                        model=self.model,
                        input=finalized_prompt,
                        instructions=Root_cause_analysis_system_prompt,
                        metadata={"model": self.model},
                    ) as stream:
                        final = await stream.get_final_response()
                        self._log_usage(span, final)
                        return final.output_text
                else:
                    response = self.client.responses.create(
                        model=self.model,
                        input=finalized_prompt,
                        instructions=Root_cause_analysis_system_prompt,
                        metadata={"model": self.model},
                    )
                    self._log_usage(span, response)
                    return response.output_text
        except Exception as e:
            return f"Error: {str(e)}"

    def _log_usage(self, span, response):
        metadata: Dict[str, Any] = {}
        if response.usage:
            metadata.update(
                {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )
        span.update(output=response.output_text, metadata=metadata)
        langfuse.update_current_generation(
            usage_details={
                "input": response.usage.input_tokens if response.usage else 0,
                "output": response.usage.output_tokens if response.usage else 0,
                "total": response.usage.total_tokens if response.usage else 0,
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            model=self.model,
        )
