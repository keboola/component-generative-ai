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

    async def improve_prompt(
        self,
        model_name: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 300,
        system_instruction: Optional[str] = None
    ) -> str:
        pass
