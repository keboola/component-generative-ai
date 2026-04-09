from typing import Protocol


class AIClientException(Exception):
    pass


class CommonClient(Protocol):
    """
    Declares default AIClient behaviour
    """

    async def infer(self, model_name: str, prompt: str, **model_options) -> tuple[str | None, int | None]:
        pass

    async def list_models(self) -> list:
        pass
