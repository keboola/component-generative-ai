import aiohttp
import backoff
import logging
from typing import Optional, Tuple
import asyncio
from .base import CommonClient, AIClientException

SUPPORTED_MODELS = ['meta-llama/Meta-Llama-3-8B', 'mistralai/Mixtral-8x7B-Instruct-v0.1', 'microsoft/phi-2']


def giveup_handler(details):
    raise AIClientException(f"All retries failed after {details['tries']} attempts over {details['elapsed']:0.2f} "
                            f"seconds. Last exception: {details['exception']}")


class HuggingfaceClient(CommonClient):
    def __init__(self, api_key):
        self.client = None
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        self.semaphore = asyncio.Semaphore(1)

    @backoff.on_exception(backoff.expo, aiohttp.ClientResponseError, max_time=60, giveup=giveup_handler)
    async def infer(self, model_name: str, prompt: str, **model_options) -> Tuple[Optional[str], Optional[int]]:
        async with self.semaphore:
            if "max_tokens" in model_options:
                model_options["max_new_tokens"] = model_options["max_tokens"]
                del model_options["max_tokens"]

            if "presence_penalty" in model_options:
                if model_options["presence_penalty"] <= 0:
                    logging.warning("presence_penalty (repetition_penalty) should be greater than 0. "
                                    "This parameter will be ignored.")
                    del model_options["presence_penalty"]

                model_options["repetition_penalty"] = model_options["presence_penalty"]
                del model_options["presence_penalty"]

            logging.debug(f"Using model options: {model_options}")
            content = await self.generate_text(prompt=prompt, model_name=model_name, **model_options)
            logging.debug(f"Response: {content}")

            token_usage = await self.get_total_tokens(model_name, prompt, content)

            return content, token_usage

    async def generate_text(self, prompt: str, model_name: str, **model_options) -> str:
        endpoint = f'https://api-inference.huggingface.co/models/{model_name}'
        data = {
            'inputs': prompt,
            'parameters': model_options
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=self.headers, json=data) as response:
                result = await response.json()
                try:
                    response.raise_for_status()
                    return result[0]['generated_text']
                except Exception as e:
                    raise e

    @staticmethod
    async def get_total_tokens(model: str, prompt: str, response: str = "") -> int:
        return 0

    async def list_models(self) -> list:
        return SUPPORTED_MODELS
