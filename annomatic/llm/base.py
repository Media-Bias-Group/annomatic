from abc import ABC, abstractmethod

from typing import Any, Union


class Response:
    """
    Class for the Response given by LLMs.

    Arguments:
        answer: the parsed answer
        _data: is the raw unedited output produced by the model
    """

    def __init__(self, answer: str, data: Any):
        self.answer = answer
        self.data = data

    def __str__(self):
        return self.answer


class ResponseList:
    """
    Class for the list of Responses given by LLMs.

    Arguments:
        answers: list of parsed answers
        data: list of raw unedited outputs produced by the model
    """

    def __init__(self, answers=None, data=None):
        if answers is None:
            answers = []
        if data is None:
            data = []
        if len(answers) != len(data):
            raise ValueError(
                "The length of 'answers' and 'data' lists must be the same.",
            )
        self.responses = [
            Response(answer, data_point)
            for answer, data_point in zip(answers, data)
        ]

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, index):
        return self.responses[index]

    def __iter__(self):
        return iter(self.responses)

    def __str__(self):
        return "\n".join([str(response) for response in self.responses])

    def __repr__(self):
        return f"ResponseList({self.responses})"


class Model(ABC):
    """
    Base Model for LLMs
    """

    @abstractmethod
    def predict(self, messages) -> ResponseList:
        """
        Predict the given messages. Message can be of type str or List[str]
        """
