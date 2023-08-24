from abc import ABC, abstractmethod


class AIClientException(Exception):
    pass


class CommonClient(ABC):
    """
    Declares default AIClient behaviour
    """

    @abstractmethod
    def infer(self, model_name: str, prompt: str, **model_options):
        pass

    def list_models(self):
        pass
