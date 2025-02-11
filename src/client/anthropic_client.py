import asyncio

from anthropic import AsyncAnthropic, AnthropicError
from typing import Optional, Tuple, Callable

from .base import CommonClient, AIClientException


def on_giveup(details: dict):
    raise AIClientException(details.get("exception"))


class AnthropicClient(CommonClient):
    """
    Implements a client for the Anthropic library.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = AsyncAnthropic(api_key=api_key, max_retries=5)
        self.inference_function: Callable = None
        self.semaphore = asyncio.Semaphore(5)

    async def infer(self, model_name: str, prompt: str, **model_options) -> Tuple[Optional[str], Optional[int]]:
        if not self.inference_function:
            self.inference_function = await self.get_inference_function(model_name)

        return await self.inference_function(model_name=model_name, prompt=prompt, **model_options)

    async def get_inference_function(self, model_name: str) -> Callable:
        """Returns the appropriate inference function."""
        try:
            await self.get_chat_completion_result(model_name, prompt="This is a test prompt.", max_tokens=20)
            return self.get_chat_completion_result
        except AnthropicError:
            raise AIClientException(f"The component is unable to use model {model_name}. Please check your API key.")

    async def get_chat_completion_result(self, model_name: str, prompt: str, **model_options) \
            -> Tuple[Optional[str], Optional[int]]:

        max_tokens = model_options.get('max_tokens', 1024)
        messages = [{"role": "user", "content": prompt}]

        # Use semaphore to limit parallelism due to lower token per minute limit
        async with self.semaphore:
            try:
                response = await self.client.messages.create(
                    max_tokens=max_tokens,
                    model=model_name,
                    messages=messages
                )
            except AnthropicError as e:
                raise AIClientException(f"Encountered AnthropicError: {e}")

            content = response.content[0].text
            token_usage = response.usage.output_tokens

            return content, token_usage

    async def improve_prompt(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 300,
        system_instructions: Optional[str] = None
    ) -> str:
        """
        Enhances the given prompt using Anthropic API
        """
        default_system = (
            "You are an expert prompt engineer. "
            "Improve the clarity, conciseness, and the effectiveness of the following prompt."
        )

        messages = [
            {
                "role": "user",
                "content": f"{system_instructions or default_system}\n\nPrompt to improve:\n{prompt}"
            }
        ]

        try:
            response = await self.client.messages.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        except AnthropicError as e:
            raise AIClientException(f"AnthropicError: {e}")

        return response.content[0].text

    async def list_models(self) -> list:
        try:
            response = await self.client.models.list()
        except AnthropicError as e:
            raise AIClientException(f"Encountered AnthropicError while listing models: {e}")

        return [r.id for r in response.data]
