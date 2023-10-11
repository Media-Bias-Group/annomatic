from abc import ABC, abstractmethod


class Response(ABC):
    """
    Base class for the Answer given by LLMs.
    """

    @abstractmethod
    def __init__(self, answer: str, misc: None):
        self._answer = answer
        self.misc = misc


class Model(ABC):
    @abstractmethod
    def predict(self, message):
        """

        :param message:
        :return:
        """
