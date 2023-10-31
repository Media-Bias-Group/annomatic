import re
from abc import ABC, abstractmethod
from typing import Any, List


class BaseTemplater(ABC):
    """
    Abstract class representing the base class for a Template engine.
    """

    @abstractmethod
    def __init__(self):
        self._variables: List[str] = []

    @abstractmethod
    def get_variables(self) -> List[str]:
        """
        Collects all placeholder variables and returns them as a list.
        List is order by occurrence.

        Returns:
             list of placeholder variables
        """

    @abstractmethod
    def parse(self, **kwargs: Any) -> str:
        """
        Performs the variable substitution and returns the filled string

        :param kwargs: dict containing template variables and their values
        :return: filled str
        """


class FstringTemplater(BaseTemplater):
    """
    This class represents a template which utilizes the python f-String syntax.
    """

    VARIABLE_REGEX = r"\{(\w+)\}"
    """Regex for finding all variables"""

    def __init__(self, template: str):
        self._template = template
        self._variables = []

    def _compute_variables(self) -> List[str]:
        """
        Computes the list of variables used in this segment and stores it in
        _variables and returns it.
        """
        if not self._variables:
            self._variables = re.findall(self.VARIABLE_REGEX, self._template)
        return self._variables

    def get_variables(self) -> List[str]:
        """
        Returns a list of variables used in this segment.

        Returns:
            List of strings representing variables.
        """
        return self._compute_variables()

    def parse(self, **kwargs: Any) -> str:
        """
        Performs the variable substitution and returns the filled string.

        :param kwargs: dict containing template variables and their values
        :return: filled str
        """
        try:
            return self._template.format_map(kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable: {e.args[0]}") from None

    @property
    def template(self):
        return self._template
