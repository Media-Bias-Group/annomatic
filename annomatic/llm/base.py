from abc import ABC, abstractmethod
from typing import Any, List


class Response:
    """
    Class for the Response given by LLMs.

    Arguments:
        answer: the parsed answer
        _data: is the raw unedited output produced by the model
        _query: the query that was asked
    """

    def __init__(self, answer: str, data: Any, query: str):
        self.answer = answer
        self._data = data
        self._query = query

    def __str__(self):
        return self.answer

    @property
    def data(self):
        return self._data

    @property
    def query(self):
        return self._query


class ResponseList:
    """
    Class for the list of Responses given by LLMs.

    Arguments:
        answers: list of parsed answers
        data: list of raw unedited outputs produced by the model
    """

    def __init__(self, answers=None, data=None, queries=None):
        if answers is None:
            answers = []
        if data is None:
            data = []
        if queries is None:
            queries = []
        if len(answers) != len(data):
            raise ValueError(
                "The length of 'answers' and 'data' lists must be the same.",
            )
        self.responses = [
            Response(answer=answer, data=data_point, query=query)
            for answer, data_point, query in zip(answers, data, queries)
        ]

    @staticmethod
    def from_responses(responses: List[Response]) -> "ResponseList":
        """
        Create a ResponseList from a list of Responses
        """
        return ResponseList(
            answers=[response.answer for response in responses],
            data=[response.data for response in responses],
            queries=[response.query for response in responses],
        )

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
    def __init__(self, model_name: str):
        """
        Initialize the model.
        """
        self.model_name = model_name

    @abstractmethod
    def predict(self, messages: List[str]) -> ResponseList:
        """
        Predict the given messages. Message can be of type str or List[str]
        """


class ModelConfig(ABC):
    """
    Base ModelConfig for LLMs
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the model config.
        """

    def to_dict(self):
        """
        Convert the model config to a dictionary.

        Values that are None or the default value are not included
        in the dictionary.
        """
        pass


class ModelPredictionError(Exception):
    """Custom exception for model prediction errors."""

    def __init__(self, message):
        super().__init__(message)
