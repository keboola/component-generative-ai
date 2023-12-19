from typing import Protocol, Tuple, Optional


class AIClientException(Exception):
    pass


class CommonClient(Protocol):
    """
    Declares default AIClient behaviour
    """

    async def infer(self, model_name: str, prompt: str, **model_options) -> Tuple[Optional[str], Optional[int]]:
        pass

    async def list_models(self) -> list:
        pass
