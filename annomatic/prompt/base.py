from abc import ABC, abstractmethod
from typing import Any


class BaseTemplater(ABC):
    """
    Abstract class representing the base class for a Template engine.
    """

    def __init__(self, template: str):
        self.template = template

    @abstractmethod
    def to_string(self, **kwargs: Any) -> str:
        """
        Performs the variable substitution and returns the filled string

        :param kwargs: dict containing template variables and their values
        :return: filled str
        """


class FstringTemplater(BaseTemplater):
    """
    This class represents a template which utilizes the python f-String syntax.
    """

    def to_string(self, **kwargs: Any) -> str:
        """
        Performs the variable substitution and returns the filled string.

        :param kwargs: dict containing template variables and their values
        :return: filled str
        """
        try:
            return self.template.format_map(kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable: {e.args[0]}") from None
