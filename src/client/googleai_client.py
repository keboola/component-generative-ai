import google.api_core.exceptions
import google.generativeai as genai
from typing import Optional, Tuple

from .base import AIClientException


class GoogleAIClient:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = None

    async def infer(self, model_name: str, prompt: str, **model_options) \
            -> Tuple[Optional[str], Optional[int]]:
        if not self.model:
            self.model = genai.GenerativeModel(model_name=model_name)

        try:
            response = await self.model.generate_content_async(prompt)
        except google.api_core.exceptions.FailedPrecondition as e:
            raise AIClientException(f"FailedPrecondition: {e}")

        content = str(response)

        token_usage = await self.get_total_tokens(model_name, prompt, content)

        return content, token_usage

    @staticmethod
    async def get_total_tokens(model: str, prompt: str, response: str = ""):
        prompt_tokens = google.ai.generativelanguage.CountTextTokensRequest(model, prompt)
        response_tokens = 0  # TODO

        return prompt_tokens + response_tokens

    @staticmethod
    async def list_models() -> list:
        models_with_generate_content = [
            m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods
        ]
        return models_with_generate_content
