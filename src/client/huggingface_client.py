import backoff
import logging
from typing import Optional, Tuple
from .base import CommonClient, AIClientException
from httpx import HTTPStatusError


from keboola.http_client import AsyncHttpClient

SUPPORTED_MODELS = {
    "mistral-7b-instruct-v0-1-fep": "https://qwiarbdt0vaeh0cb.us-east-1.aws.endpoints.huggingface.cloud",
    "gemma-7b-it-voa": "https://yi9ctuuxjnmgtjkc.us-east-1.aws.endpoints.huggingface.cloud",
    "meta-llama-3-8b-noz": "https://nbd6y80gxtows9b3.us-east-1.aws.endpoints.huggingface.cloud",
    "custom": "custom"
}


class IsSleepingException(Exception):
    pass


class HuggingfaceClient(CommonClient):
    def __init__(self, api_key):
        self.client = None
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

    async def infer(self, model_name: str, prompt: str, **model_options) -> Tuple[Optional[str], Optional[int]]:
        if model_name not in SUPPORTED_MODELS:
            raise AIClientException(f"Model {model_name} is not supported. "
                                    f"Supported models: {list(SUPPORTED_MODELS.keys())}")

        if not self.client:
            self.client = AsyncHttpClient(base_url=SUPPORTED_MODELS[model_name], default_headers=self.headers,
                                          max_requests_per_second=1)

        max_new_tokens = model_options.get("max_tokens")
        model_options = {"max_new_tokens": max_new_tokens} if max_new_tokens else {}

        logging.debug(f"Prompt: {prompt}")
        logging.debug(f"Using model options: {model_options}")

        content = await self.generate_text(prompt=prompt, model_name=model_name, **model_options)
        logging.debug(f"Response: {content}")

        token_usage = await self.get_total_tokens(model_name, prompt, content)

        return content, token_usage

    @backoff.on_exception(backoff.expo, IsSleepingException, max_time=500)
    async def generate_text(self, prompt: str, model_name: str, **model_options) -> str:
        endpoint = SUPPORTED_MODELS.get(model_name)

        data = {
            'inputs': prompt,
            'parameters': model_options
        }

        try:
            result = await self.client.post(endpoint, json=data)
        except HTTPStatusError as e:
            if self._is_asleep(e.response.status_code):
                logging.warning("The model is currently sleeping. The component will now wait "
                                "for the model to wake up.")
                raise IsSleepingException()
            else:
                raise AIClientException(f"HTTP error occurred: {e.response.status_code} {e.response.reason_phrase}")

        return result[0]['generated_text']

    @staticmethod
    def _is_asleep(status_code: int) -> bool:
        return status_code == 503

    @staticmethod
    async def get_total_tokens(model: str, prompt: str, response: str = "") -> int:
        return 0

    async def list_models(self) -> list:
        return list(SUPPORTED_MODELS.keys())
