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

    @abstractmethod
    def content(self):
        """
        Returns the content without input variables filled in.
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

    def content(self):
        return self._content


class PromptTemplateSegment(PromptSegment):
    """
    This is class represents a part of a Prompt that has his own
    sub-template.

    This class directly uses the F-String Templater and uses {..}
    to determine attributes

    """

    def __init__(self, template: str = ""):
        self._template = FstringTemplater(template)

    def get_variables(self) -> List[str]:
        return self._template.get_variables()

    def to_string(self, **kwargs: Any) -> str:
        return self._template.parse(**kwargs)

    def content(self):
        return self._template.template


class LabelTemplateSegment(PromptSegment):
    """
    This is class represents a part of a Prompt that has his own
    sub-template.

    This class directly uses the F-String Templater and uses {..}
    to determine attributes.
    """

    SEPARATOR = ", "
    """Separator between labels"""

    LAST_SEPARATOR = " or "
    """Separator between last two labels"""

    def __init__(self, template: str, label_var: str = "label"):
        self._template = FstringTemplater(template)
        self._label_var = label_var

    def get_variables(self) -> List[str]:
        return self._template.get_variables()

    def to_string(self, **kwargs: Any) -> str:
        if self._label_var in kwargs:
            labels = kwargs[self._label_var]
            if isinstance(labels, list) and labels:
                if len(labels) == 1:
                    kwargs[self._label_var] = labels[0]
                    return self._template.parse(**kwargs)
                else:
                    label_string = self.SEPARATOR.join(labels[:-1])
                    label_string += self.LAST_SEPARATOR + labels[-1]
                    kwargs[self._label_var] = label_string
                    return self._template.parse(**kwargs)
            elif isinstance(labels, str):
                return self._template.parse(**kwargs)
            else:
                raise ValueError(
                    f"Invalid value for labels variable: {self._label_var}",
                )
        else:
            return self._template.parse(**kwargs)

    def content(self):
        return self._template.template
