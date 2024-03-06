import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from annomatic.annotator.util import find_label

LOGGER = logging.getLogger(__name__)


class PostProcessor(ABC):
    """
    Base class for post processors.
    """

    def __init__(
        self,
        input_col: str,
        output_col: str,
        labels: Optional[List[str]],
    ):
        self.input_col = input_col
        self.output_col = output_col
        self.labels = labels

    @abstractmethod
    def process(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Processes the given dataframe.

        Args:
            df: dataframe to be processed as a DataFrame

        Returns:
            the processed dataframe
        """
        raise NotImplementedError()


class DefaultPostProcessor(PostProcessor):
    """
    Default class for post processors.

    If labels are known, the in_colum is processed to find the label and
    stored in the output column. Only store if the label is found uniquely.

    Otherwise, store '?' in output column.
    """

    def __init__(
        self,
        input_col: str = "response",
        output_col: str = "label",
        labels: Optional[List[str]] = None,
    ):
        super().__init__(input_col, output_col, labels)

    def process(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Processes the given dataframe.


        If labels are known, the in_colum is processed to find the label and
        stored in the output column. Only store if the label is found uniquely.
        Otherwise, store '?' in output column.

        Args:
                df: dataframe to be processed as a DataFrame

        Returns:
            the processed dataframe
        """

        if self.labels is None:
            LOGGER.info("No Post, processing done")
            return df
        df[self.output_col] = df[self.input_col].apply(
            lambda x: find_label(x, self.labels),
        )
        LOGGER.info("Post processing done")
        return df
