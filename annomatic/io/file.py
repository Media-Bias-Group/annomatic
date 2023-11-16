import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

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


class BaseFileInput(BaseIO, ABC):
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


class BaseFileOutput(BaseIO, ABC):
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


class CsvInput(BaseFileInput):
    def __init__(self, path: str):
        """
        Arguments:
            path (str): Path to the file.
        """
        super().__init__(path)

    def read(self, sep: str) -> pd.DataFrame:
        """
        Read the provided content from the file.

        Arguments:
            sep (str): The separator to use for parsing the CSV file.
        """
        try:
            return pd.read_csv(self._path, sep=sep)
        except Exception as e:
            LOGGER.error(
                f"Error reading CSV file: {e}. Return Empty DataFrame",
            )
            # TODO own Exception
            raise IOError(e)


class CsvOutput(BaseFileOutput):
    def __init__(self, path: str):
        """
        Arguments:
            path (str): Path to the file.
        """
        super().__init__(path)

    def write(self, df: pd.DataFrame):
        """
        Write the provided content to the file.

        Arguments:
            df (pd.DataFrame): The content to be written to the file.
        """
        if df.empty:
            LOGGER.warning("DataFrame is empty. no file written.")
            return
        try:
            df.to_csv(self._path, index=False)
            LOGGER.info(f"CSV file written to {self._path}")
        except Exception as e:
            # TODO own Exception
            LOGGER.error(f"Error writing CSV file: {e}")
