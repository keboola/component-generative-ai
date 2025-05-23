import backoff
from typing import Optional, Tuple
import google.api_core.exceptions
import google.generativeai as genai
from google.generativeai.types import AsyncGenerateContentResponse

from .base import CommonClient, AIClientException


class GoogleAIClient(CommonClient):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = None

    @backoff.on_exception(backoff.expo, google.api_core.exceptions.ResourceExhausted, max_time=60)
    async def infer(self, model_name: str, prompt: str, **model_options) \
            -> Tuple[Optional[str], Optional[int]]:
        if not self.model:
            self.model = genai.GenerativeModel(model_name=model_name)

        try:
            response = await self.model.generate_content_async(prompt)
        except google.api_core.exceptions.FailedPrecondition as e:
            raise AIClientException(f"FailedPrecondition: {e}")
        except google.auth.exceptions.GoogleAuthError as e:
            raise AIClientException(f"GoogleAuthError: {e}")

        try:
            content = str(response.text)
        except (IndexError, ValueError) as e:
            feedback = str(response.prompt_feedback) if response.prompt_feedback else ""
            content = f"Failed to process prompt: {prompt}, feedback: {feedback}, reason: {e}"

        token_usage = await self.get_total_tokens(response)

        return content, token_usage

    @staticmethod
    async def get_total_tokens(response: AsyncGenerateContentResponse) -> int:
        usage_metadata = getattr(response, "usage_metadata", 0)
        total_tokens = getattr(usage_metadata, "total_token_count", 0)
        return total_tokens

    async def list_models(self) -> list:
        models_with_generate_content = [
            m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods
        ]
        return models_with_generate_content
