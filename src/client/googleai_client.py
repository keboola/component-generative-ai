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

        try:
            content = str(response.text)
        except IndexError as e:
            content = f"Failed to process prompt: {prompt}, reason: {e}"

        token_usage = await self.get_total_tokens(model_name, prompt, content)

        return content, token_usage

    @staticmethod
    async def get_total_tokens(model: str, prompt: str, response: str = "") -> int:
        # todo implement https://ai.google.dev/api/python/google/ai/generativelanguage/CountTextTokensRequest
        return 0

    @staticmethod
    async def list_models() -> list:
        models_with_generate_content = [
            m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods
        ]
        return models_with_generate_content
