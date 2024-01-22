import logging
from typing import Any

import pandas as pd

from annomatic.io.base import BaseInput, BaseOutput

LOGGER = logging.getLogger(__name__)


class CsvInput(BaseInput):
    def __init__(self, path: str):
        """
        Arguments:
            path (str): Path to the file.
        """
        super().__init__(path)

    def read(self, sep: str) -> Any:
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
            raise IOError(e)


class CsvOutput(BaseOutput):
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
            LOGGER.info(df)
