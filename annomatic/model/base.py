from abc import ABC, abstractmethod


class Answer(ABC):
    """
    Base class for the Answer given by LLMs.
    """

    pass


class Model(ABC):
    @abstractmethod
    def predict(self, message: str):
        """

        :param message:
        :return:
        """
