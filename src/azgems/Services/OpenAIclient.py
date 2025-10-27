import os
import logging
from openai import AsyncOpenAI
from typing import List, Optional

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Generic OpenAI client for chat and embedding operations."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided or found in environment.")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model

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
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
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
            response = await self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format=response_format,
            )
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
