import logging
from abc import ABC, abstractmethod


class BaseAnnotator(ABC):
    """
    Base class for annotator classes
    """

    @abstractmethod
    def annotate(self, **kwargs):
        """
        Annotates the input data and stores the annotated data.

        Args:
            kwargs: a dict containing the input variables for prompt templates
        """
        raise NotImplementedError()

    @abstractmethod
    def _load_model(self):
        pass


# TODO add Mixin for Each LLM
