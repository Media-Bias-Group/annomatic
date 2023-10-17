import logging
from abc import ABC, abstractmethod

import pandas as pd
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
    def read(self):
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

    def read(self) -> pd.DataFrame:
        """
        Read the provided content from the file.
        """
        try:
            return pd.read_csv(self._path)
        except Exception as e:
            LOGGER.error(
                f"Error reading CSV file: {e}. Return Empty DataFrame",
            )
            return pd.DataFrame()


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
            LOGGER.error(f"Error writing CSV file: {e}")
