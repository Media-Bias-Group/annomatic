import logging
from abc import ABC, abstractmethod
from typing import Any

LOGGER = logging.getLogger(__name__)


class BaseIO(ABC):
    """
    Base class for input and output classes
    """

    @abstractmethod
    def __init__(self, path: str):
        """
        Arguments:
            path (str): Path to the file.
        """
        self._path = path


class BaseInput(BaseIO, ABC):
    """
    Base class for file input classes
    """

    @abstractmethod
    def __init__(self, path: str):
        """
        Arguments:
            path (str): Path to the file.
        """
        super().__init__(path)

    @abstractmethod
    def read(self, sep: str) -> Any:
        """
        Read the provided content to the file.
        """
        raise NotImplementedError()


class BaseOutput(BaseIO, ABC):
    """
    Base class for file output classes
    """

    @abstractmethod
    def __init__(self, path: str):
        """
        Arguments:
            path (str): Path to the file.
        """
        super().__init__(path)

    @abstractmethod
    def write(self, content: Any):
        """
        Write the provided content to the file.

        Arguments:
            content (str): The content to be written to the file.
        """
        raise NotImplementedError()
