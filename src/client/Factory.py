from abc import ABC, abstractmethod


class CommonClient(ABC):
    """
    Declares default Client behaviour
    """

    def __init__(self, api_token):
        self.api_token = api_token

    @abstractmethod
    def infer(self, prompt, **model_options):
        """
        Note that the Creator may also provide some default implementation of
        the factory method.
        """
        pass
