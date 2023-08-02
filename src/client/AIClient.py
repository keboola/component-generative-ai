import backoff
import logging
import openai
from typing import Optional


from .Factory import CommonClient

OPENAI_MODELS = ["gpt-4", "gpt-4-0314", "gpt-4-32k", "gpt-4-32k-0314", "gpt-3.5-turbo", "gpt-3.5-turbo-0301"]


class AIClientException(Exception):
    pass


def on_giveup(details: dict):
    raise AIClientException(details.get("exception"))


class OpenAIClient(CommonClient):

    def __init__(self, api_token, model):
        super().__init__(api_token)
        openai.api_key = api_token
        self.model = model

        self.inference_function = self.get_inference_function(model)

    def infer(self, prompt, **model_options) -> str:
        return self.inference_function(self.model, prompt, **model_options)

    @backoff.on_exception(backoff.expo,
                          (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError),
                          max_tries=3, on_giveup=on_giveup)
    def get_completion_result(self, model: str, prompt: str, **model_options) -> Optional[str]:
        try:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                **model_options
            )
        except openai.error.InvalidRequestError as e:
            logging.error(f"Invalid Request Error: {e}")
            return None
        except openai.error.AuthenticationError as e:
            raise AIClientException("Your OpenAI API key is invalid") from e
        except openai.error.APIConnectionError as e:
            raise AIClientException(f"API connection Error: {e}") from e

        return response.choices[0].text

    @backoff.on_exception(backoff.expo,
                          (openai.error.RateLimitError, openai.error.APIError, openai.error.ServiceUnavailableError),
                          max_tries=3, on_giveup=on_giveup)
    def get_chat_completion_result(self, model: str, prompt: str, **model_options) -> Optional[str]:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **model_options
            )
        except openai.error.InvalidRequestError as e:
            logging.error(f"Invalid Request Error: {e}")
            return None
        except openai.error.AuthenticationError as e:
            raise AIClientException("Your OpenAI API key is invalid") from e
        except openai.error.APIConnectionError as e:
            raise AIClientException(f"API connection Error: {e}") from e

        return response.choices[0].get("message", {}).get("content")

    def get_inference_function(self, model_name: str):
        if model_name in OPENAI_MODELS:
            return self.get_chat_completion_result
        return self.get_completion_result
