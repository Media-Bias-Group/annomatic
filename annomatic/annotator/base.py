import logging
from abc import ABC, abstractmethod

LOGGER = logging.getLogger(__name__)


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
