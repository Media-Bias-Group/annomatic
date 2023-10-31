from abc import ABC, abstractmethod
from typing import Any, List, Optional

from annomatic.prompt.segment import (
    LabelTemplateSegment,
    PromptPlainSegment,
    PromptSegment,
    PromptTemplateSegment,
)
from annomatic.prompt.utils import check_template_format


class BasePrompt(ABC):
    """
    Base class for prompts.
    """

    DEFAULT_SEPARATOR = "\n\n"
    """Default separator between segments"""

    @abstractmethod
    def __init__(self):
        self._segments = []

    def __str__(self):
        """
        Concatenates the string representations of the segment and returns them
        without substitution of variables.
        """
        return BasePrompt.DEFAULT_SEPARATOR.join(
            segment.content() for segment in self._segments
        )

    def __call__(self, **kwargs: Any) -> str:
        """
        Concatenates the string representations of the segment and returns them
        with variables in kwargs.

        Calls the to_string method with the given kwargs.

        Args:
            kwargs: a dict containing the input variables for templates
        Returns:
            string representation of the prompt
        """
        return self.to_string(**kwargs)

    def to_string(
        self,
        separator: str = DEFAULT_SEPARATOR,
        **kwargs: Any,
    ) -> str:
        """
        Concatenates the string representations of the segment and returns them
        with variables in kwargs.

        Args:
            separator: The separator to include between segments.
                (default: "\n\n")
            kwargs: a dict containing the input variables for templates
        Returns:
            string representation of the prompt

        """
        return separator.join(
            segment.to_string(**kwargs) for segment in self._segments
        )


class Prompt(BasePrompt):
    """
    This class represents the Basic extendable prompt.

    Prompts can be constructed by concatenating Segments.

    Attributes:
        _segments: list of segments which make up the prompt


    """

    def __init__(self, content: Optional[str] = None):
        super().__init__()
        self._variables: List[str] = []

        if content is not None:
            self.add_part(content)

    def get_variables(self) -> List[str]:
        """
        Returns a list of all variables used in the prompt.

        Returns:
            A list of strings with all variables used in the prompt.
        """
        return [
            var
            for segment in self._segments
            for var in segment.get_variables()
        ]

    def add_part(self, content: str):
        """
        Adds the given new content as a new part of the Prompt.

        The content will by default be added on the rear of the prompt.

        Args:
            string containing the prompt
        """
        if check_template_format(content):
            self._segments.append(PromptTemplateSegment(template=content))
        else:
            self._segments.append(PromptPlainSegment(content=content))

    def add_labels_part(self, content, label_var="label"):
        """
        Adds the given new content as a new part of the Prompt.

        The content will by default be added on the rear of the prompt.

        Args:
            string containing the prompt
        """
        if check_template_format(content):
            # todo find var out of template
            self._segments.append(
                LabelTemplateSegment(template=content, label_var=label_var),
            )
        else:
            self._segments.append(PromptPlainSegment(content=content))
