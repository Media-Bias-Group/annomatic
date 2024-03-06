from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from annomatic.config.base import ModelConfig


class Response:
    """
    Class for the Response given by LLMs.

    Arguments:
        answer: the parsed answer
        _data: is the raw unedited output produced by the model
        _query: the query that was asked
    """

    def __init__(self, answer: str, data: Any, query: Union[str, List[str]]):
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
    def __init__(self, model_name: str, system_prompt: Optional[str]):
        """
        Initialize the model.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt

    @abstractmethod
    def predict(self, messages: List[str]) -> ResponseList:
        """
        Predict the given messages. Message can be of type str or List[str]
        # TODO introduce List[List[str]] for multiple conversations
        """


class ModelLoader(ABC):
    """
    Interface for model loaders.
    """

    def __init__(
        self,
        model_name: str,
        config: ModelConfig,
        system_prompt: Optional[str] = None,
        lib_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.config = config
        self.system_prompt = system_prompt
        self.lib_args = lib_args or {}
        self._model: Optional[Model] = None

    @abstractmethod
    def load_model(self) -> Model:
        """
        Loads the model and store it in self.model.

        Returns:
            The loaded model.
        """
        raise NotImplementedError()

    def update_config_generation_args(
        self,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        for key, value in (generation_args or {}).items():
            if (
                hasattr(self.config, key)
                and getattr(self.config, key) != value
            ):
                setattr(self.config, key, value)
            else:
                self.config.kwargs[key] = value


class ModelPredictionError(Exception):
    """Custom exception for model prediction errors."""

    def __init__(self, message):
        super().__init__(message)
