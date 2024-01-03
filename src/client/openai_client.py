import logging
import openai
from openai import AsyncOpenAI, AsyncAzureOpenAI
from typing import Optional, Tuple, Callable

from .base import CommonClient, AIClientException


def on_giveup(details: dict):
    raise AIClientException(details.get("exception"))


class OpenAIClient(AsyncOpenAI, CommonClient):
    """
    Implements OpenAI and AzureOpenAI clients.
    TODO: Implement try except using wrapper.
    """

    def __init__(self, api_key: str):
        self.inference_function: callable = None
        super().__init__(api_key=api_key)

    async def infer(self, model_name: str, prompt: str, **model_options) -> Tuple[Optional[str], Optional[int]]:
        if not self.inference_function:
            self.inference_function = await self.get_inference_function(model_name)

        return await self.inference_function(model_name=model_name, prompt=prompt, **model_options)

    async def get_inference_function(self, model_name: str) -> Callable:
        """Returns appropriate inference function (either Completion or ChatCompletion)."""
        try:
            await self.get_chat_completion_result(model_name, prompt="This is a test prompt.", timeout=60,
                                                  max_tokens=20)
            return self.get_chat_completion_result
        except openai.OpenAIError:
            logging.warning(f"Cannot use chat_completion endpoint for model {model_name}, the component will try to use"
                            f"completion_result endpoint.")

        try:
            await self.get_completion_result(model_name, "This is a test prompt.", timeout=60,
                                             max_tokens=20)
            return self.get_completion_result
        except openai.OpenAIError:
            raise AIClientException(f"The component is unable to use model {model_name}. Please check your API key.")

    async def get_completion_result(self, model_name: str, prompt: str, **model_options) \
            -> Tuple[str, Optional[int]]:

        try:
            response = await self.completions.create(model=model_name, prompt=prompt, **model_options)
        except openai.OpenAIError as e:
            raise AIClientException(f"Encountered OpenAIError: {e}")

        content = response.choices[0].text
        token_usage = response.usage.total_tokens

        return content, token_usage

    async def get_chat_completion_result(self, model_name: str, prompt: str, **model_options) \
            -> Tuple[Optional[str], Optional[int]]:

        try:
            response = await self.chat.completions.create(model=model_name,
                                                          messages=[{"role": "user", "content": prompt}],
                                                          **model_options)
        except openai.OpenAIError as e:
            raise AIClientException(f"Encountered OpenAIError: {e}")

        content = response.choices[0].message.content
        token_usage = response.usage.total_tokens

        return content, token_usage

    async def list_models(self) -> list:
        r = await self.models.list()
        return [model.id for model in r.data]


class AzureOpenAIClient(AsyncAzureOpenAI, CommonClient):
    def __init__(self, api_key, api_base, deployment_id, api_version):
        super().__init__(api_key=api_key,
                         api_version=api_version,
                         azure_endpoint=api_base,
                         azure_deployment=deployment_id)

    async def infer(self, model_name: str, prompt: str, **model_options) -> Tuple[str, Optional[int]]:

        try:
            response = await self.chat.completions.create(model=model_name,
                                                          messages=[{"role": "user", "content": prompt}],
                                                          **model_options)
        except openai.BadRequestError as e:
            raise AIClientException(f"BadRequest Error: {e}")
        except openai.OpenAIError as e:
            raise AIClientException(f"Encountered OpenAIError: {e}")

        content = response.choices[0].message.content
        if not content:
            if response.choices[0].finish_reason == "content_filter":
                raise AIClientException(f"Cannot process prompt: {prompt}\nReason: content_filter\n"
                                        f"For more information visit https://learn.microsoft.com/en-us/azure/"
                                        f"ai-services/openai/concepts/content-filter?tabs=warning%2Cpython")
            content = ""

        token_usage = response.usage.total_tokens

        return content, token_usage

    async def list_models(self) -> list:
        r = await self.models.list()
        return [model.id for model in r.data]
