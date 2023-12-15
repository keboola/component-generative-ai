from abc import ABC, abstractmethod
from typing import Tuple, Optional


class AIClientException(Exception):
    pass


class CommonClient(ABC):
    """
    Declares default AIClient behaviour
    """

    @abstractmethod
    async def infer(self, model_name: str, prompt: str, **model_options) -> Tuple[Optional[str], Optional[int]]:
        pass

    def list_models(self) -> list:
        pass
