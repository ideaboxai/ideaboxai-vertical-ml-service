import os
import logging
from openai import AsyncOpenAI
from typing import Any, List, Optional, Dict
from langfuse import Langfuse

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Generic OpenAI client for chat and embedding operations."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided or found in environment.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model
        self.langfuse = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST"),
        )

    def _log_usage(self, span, response):
        metadata: Dict[str, Any] = {}
        if response.usage:
            metadata.update(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )
        span.update(output=response.choices[0].message.content, metadata=metadata)
        self.langfuse.update_current_generation(
            usage_details={
                "input": response.usage.prompt_tokens if response.usage else 0,
                "output": response.usage.completion_tokens if response.usage else 0,
                "total": response.usage.total_tokens if response.usage else 0,
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": (
                    response.usage.completion_tokens if response.usage else 0
                ),
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            model=self.model,
        )

    async def generate_response(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.7
    ) -> str | None:
        """
        Generate a chat completion using OpenAI Chat models.
        :param messages: list of dicts (e.g. [{"role": "user", "content": "Hello!"}])
        :param temperature: controls randomness (0.0–1.0)
        :return: model response text
        """
        try:
            with self.langfuse.start_as_current_span(name="Azgems Recommender") as span:
                span.update(input=system_prompt + "\n\n" + user_prompt)
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                self._log_usage(span, response)
                return response.choices[0].message.content

        except Exception as e:
            logger.exception("Error generating OpenAI response")
            raise RuntimeError(f"OpenAI API Error: {str(e)}")

    async def generate_formatted_response(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format,
        temperature: float = 0.7,
    ) -> str | None:
        try:
            with self.langfuse.start_as_current_span(name="Azgems Recommender") as span:
                span.update(input=system_prompt + "\n\n" + user_prompt)
                response = await self.client.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    response_format=response_format,
                )
                self._log_usage(span, response)
                return response.choices[0].message.parsed

        except Exception as e:
            logger.exception("Error generating OpenAI response")
            raise RuntimeError(f"OpenAI API Error: {str(e)}")


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    import asyncio

    load_dotenv()

    client = OpenAIClient()
    response = asyncio.run(
        client.generate_response(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is the capital of France?",
        )
    )
    print(response)
