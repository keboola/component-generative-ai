from abc import ABC

import backoff
import logging
import openai
from typing import Optional, Tuple

from .base import CommonClient, AIClientException

OPENAI_CHAT_MODELS = ["gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-3.5-turbo", "gpt-3.5-turbo-0301"]


def on_giveup(details: dict):
    raise AIClientException(details.get("exception"))


class BaseOpenAIClient(CommonClient, ABC):
    """
    Implements OpenAI and AzureOpenAI clients.
    """
    def __init__(self, api_token):
        openai.api_key = api_token

    def get_inference_function(self, model_name: str):
        if model_name in OPENAI_CHAT_MODELS:
            return self.get_chat_completion_result
        return self.get_completion_result

    @backoff.on_exception(
        backoff.expo,
        (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError),
        max_tries=3,
        on_giveup=on_giveup
    )
    def get_completion_result(self, model: str, prompt: str, **model_options) -> Tuple[Optional[str], Optional[int]]:
        try:
            response = openai.Completion.create(model=model, prompt=prompt, **model_options)
        except openai.error.InvalidRequestError as e:
            logging.error(f"Invalid Request Error: {e}")
            return None, 0
        except openai.error.AuthenticationError as e:
            raise AIClientException("Your OpenAI API key is invalid") from e
        except openai.error.APIConnectionError as e:
            raise AIClientException(f"API connection Error: {e}") from e

        content = response.choices[0].text
        token_usage = response.get("usage", {}).get("total_tokens")

        return content, token_usage

    @backoff.on_exception(
        backoff.expo,
        (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError),
        max_tries=3,
        on_giveup=on_giveup
    )
    def get_chat_completion_result(self, model: str, prompt: str, **model_options)\
            -> Tuple[Optional[str], Optional[int]]:
        try:
            response = openai.ChatCompletion.create(model=model,
                                                    messages=[{"role": "user", "content": prompt}], **model_options)
        except openai.error.InvalidRequestError as e:
            logging.error(f"Invalid Request Error: {e}")
            return None, 0
        except openai.error.AuthenticationError as e:
            raise AIClientException("Your OpenAI API key is invalid") from e
        except openai.error.APIConnectionError as e:
            raise AIClientException(f"API connection Error: {e}") from e

        content = response.choices[0].get("message", {}).get("content")
        token_usage = response.get("usage", {}).get("total_tokens")

        return content, token_usage

    def infer(self, model_name, prompt, **model_options) -> Tuple[Optional[str], Optional[int]]:
        inference_function = self.get_inference_function(model_name)
        return inference_function(model_name, prompt, **model_options)


class OpenAIClient(BaseOpenAIClient):
    def __init__(self, api_token):
        super().__init__(api_token)


class AzureOpenAIClient(BaseOpenAIClient):
    def __init__(self, api_token, api_base, api_version, deployment_id):
        super().__init__(api_token)
        openai.api_type = "azure"
        openai.api_base = api_base
        openai.api_version = api_version

        self.deployment_id = deployment_id
