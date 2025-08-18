from openai import OpenAI
from sandhya_aqua.utils.prompt import Recommendation_system_prompt, Recommendation_user_prompt


class OpenAIRecommender:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def get_recommendation(
        self, user_input: str, history: list, parameters: dict = None
    ):
        messages = []
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})
        finalized_prompt = Recommendation_user_prompt.format(
            **parameters, user_query=user_input
        )

        try:
            response = self.client.responses.create(
                model=self.model,
                input=finalized_prompt,
                instructions=Recommendation_system_prompt,
            )
            reply = response.output_text
            return reply
        except Exception as e:
            return f"Error: {str(e)}"
