import logging
import openai
from openai import AsyncOpenAI, AsyncAzureOpenAI
from typing import Optional, Tuple, Callable, List

from .base import CommonClient, AIClientException


def on_giveup(details: dict):
    raise AIClientException(details.get("exception"))


class OpenAISharedClient(CommonClient):
    """
    Shared logic for OpenAI-based clients (OpenAI, Azure OpenAI).
    """

    async def _call_openai_api(
        self, model_name: str, messages: List[dict], **model_options
    ) -> Tuple[str, int]:
        """
        Calls OpenAI chat completion API and returns response content and token usage.
        Handles both OpenAI and Azure OpenAI.
        """

        try:
            response = await self.chat.completions.create(
                model=model_name, messages=messages, **model_options
            )
        except openai.OpenAIError as e:
            raise AIClientException(f"OpenAI API error: {e}")

        return response.choices[0].message.content, response.usage.total_tokens

    async def improve_prompt(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Enhances the given prompt using OpenAI's chat completion API.
        """

        default_system = (
            "You are an expert prompt engineer with "
            "deep knowledge of effective AI prompting. "
            "Your task is to refine and improve the following "
            "prompt while keeping it clear, structured, and effective. "
            "However, you must strictly preserve all placeholders "
            "enclosed in double brackets (e.g., `[[INPUT_COLUMN]]`). "
            "Do not modify, replace, or remove them, as they represent "
            "dynamic values that the user will define later."
        )

        messages = [{"role": "user", "content": f"{default_system}\n\n{prompt}"}]

        try:
            response_content, _ = await self._call_openai_api(
                model_name, messages, temperature=temperature, max_completion_tokens=max_tokens
            )
        except Exception as e:
            if "temperature" in str(e) and "Only the default (1) value is supported" in str(e):
                response_content, _ = await self._call_openai_api(
                    model_name, messages, temperature=1, max_completion_tokens=max_tokens
                )
            else:
                raise e

        return response_content


class OpenAIClient(AsyncOpenAI, OpenAISharedClient):
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


class AzureOpenAIClient(AsyncAzureOpenAI, OpenAISharedClient):
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
