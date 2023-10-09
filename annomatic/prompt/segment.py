from abc import ABC, abstractmethod
from typing import Any, List

from annomatic.prompt.base import FstringTemplater


class PromptSegment(ABC):
    """
    Base class for Prompt Segments. Prompt Segments represent parts
    a given prompt.
    """

    @abstractmethod
    def get_variables(self) -> List[str]:
        """
        Returns a list of variables used in this segment.

        Returns:
            List of strings representing variables.
        """
        pass

    @abstractmethod
    def to_string(self, **kwargs: Any) -> str:
        """
        Returns a string representation of this prompt Segment.

        Args:
            kwargs: A dictionary containing input variables for templates.
        Returns:
            String representation of the segment.
        """
        pass


class PromptPlainSegment(PromptSegment):
    """
    This is class represents a part of a Prompt as plaintext.

    Attributes:
        _content: string which contains the plaintext for this Segment
    """

    def __init__(self, content: str = ""):
        """Initializes the instance based on spam preference.

        Args:
          content: string which contains the plaintext for this Segment
        """
        self._content = content

    def get_variables(self) -> List[str]:
        return []

    def to_string(self, **kwargs: Any) -> str:
        return self._content


class PromptTemplateSegment(FstringTemplater, PromptSegment):
    """
    This is class represents a part of a Prompt that has his own
    sub-template.

    This class directly uses the F-String Templater and uses {..}
    to determine attributes

    """

    def __init__(self, content: str = ""):
        super().__init__(template=content)

    def get_variables(self) -> List[str]:
        return super().get_variables()

    def to_string(self, **kwargs: Any) -> str:
        return super().parse(**kwargs)
