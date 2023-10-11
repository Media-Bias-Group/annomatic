from abc import ABC, abstractmethod
from typing import Any


class Response:
    """
    Base class for the Answer given by LLMs.

    Arguments:
        answer: the parsed answer
        _data: is the raw unedited output produced model
    """

    def __init__(self, answer: str, data: Any):
        self.answer = answer
        self._data = data


class Model(ABC):
    """
    Base Model for LLMs
    """
    @abstractmethod
    def predict(self, message):
        """
        Predict the given message. Message can be of type str or List[str]
        """
