from abc import ABC

import backoff
import logging
import openai
from typing import Optional, Tuple, Callable

from .base import CommonClient, AIClientException


def on_giveup(details: dict):
    raise AIClientException(details.get("exception"))


class OpenAIClient(CommonClient, ABC):
    """
    Implements OpenAI and AzureOpenAI clients.
    """

    def __init__(self, api_token):
        openai.api_key = api_token

    def get_inference_function(self, model_name: str) -> Callable:
        """Returns appropriate inference function (either Completion or ChatCompletion)."""
        try:
            self.get_chat_completion_result(model_name, "This is a test prompt.")
            return self.get_chat_completion_result
        except Exception:
            logging.warning(f"Cannot use chat_completion endpoint for model {model_name}, the component will try to use"
                            f"completion_result endpoint.")

        try:
            self.get_completion_result(model_name, "This is a test prompt.")
            return self.get_completion_result
        except Exception:
            raise AIClientException(f"The component is unable to use chat_completion and completion endpoints with "
                                    f"model {model_name}.")

    @staticmethod
    def get_completion_result(model_name: str, prompt: str, **model_options)\
            -> Tuple[Optional[str], Optional[int]]:
        response = openai.Completion.create(model=model_name, prompt=prompt, **model_options)

        content = response.choices[0].text
        token_usage = response.get("usage", {}).get("total_tokens")

        return content, token_usage

    def get_chat_completion_result(self, model_name: str, prompt: str, **model_options) \
            -> Tuple[Optional[str], Optional[int]]:
        response = openai.ChatCompletion.create(model=model_name,
                                                messages=[{"role": "user", "content": prompt}], **model_options)

        content = response.choices[0].get("message", {}).get("content")
        token_usage = response.get("usage", {}).get("total_tokens")

        return content, token_usage

    @backoff.on_exception(
        backoff.expo,
        (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError),
        max_tries=3,
        on_giveup=on_giveup
    )
    def infer(self, model_name, inference_function, prompt, **model_options) -> Tuple[Optional[str], Optional[int]]:

        try:
            content, token_usage = inference_function(model_name, prompt, **model_options)
        except openai.error.InvalidRequestError as e:
            logging.error(f"Invalid Request Error: {e}")
            return None, 0
        except openai.error.AuthenticationError as e:
            raise AIClientException("Your OpenAI API key is invalid") from e
        except openai.error.APIConnectionError as e:
            raise AIClientException(f"API connection Error: {e}") from e

        return content, token_usage

    def list_models(self) -> list:
        r = openai.Model.list()
        models = [item["id"] for item in r["data"]]
        return models


class AzureOpenAIClient(OpenAIClient):
    def __init__(self, api_token, api_base, deployment_id, api_version):
        super().__init__(api_token)
        openai.api_type = "azure"
        openai.api_base = api_base
        openai.api_version = api_version

        self.deployment_id = deployment_id

    def get_chat_completion_result(self, model_name: str, prompt: str, **model_options) \
            -> Tuple[Optional[str], Optional[int]]:
        response = openai.ChatCompletion.create(deployment_id=self.deployment_id, model=model_name,
                                                messages=[{"role": "user", "content": prompt}], **model_options)

        content = response.choices[0].get("message", {}).get("content")
        token_usage = response.get("usage", {}).get("total_tokens")

        return content, token_usage

    def get_inference_function(self, model_name: str) -> Callable:
        """Always returns ChatCompletion function"""
        return self.get_chat_completion_result
