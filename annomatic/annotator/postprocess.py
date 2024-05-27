import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
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
        """
        Args:
            input_col: input column name as a string
            output_col: output column name as a string
            labels: optional list of labels as a list of strings
        """
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
        unknown_label: str = "?",
    ):
        super().__init__(input_col, output_col, labels)
        self.unknown_label = unknown_label

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
            LOGGER.info("No Post processing done")
            return df
        df[self.output_col] = df[self.input_col].apply(
            lambda x: find_label(x, self.labels, default=self.unknown_label),
        )
        LOGGER.info("Post processing done")
        return df


class MajorityVote(PostProcessor):
    """
    Post processor class which  performs a majority vote.
    """

    def __init__(
        self,
        labels: Optional[List[str]],
        input_col: str = "response",
        output_col: str = "label",
        final_label: str = "majority_label",
        unknown_label: str = "?",
    ):
        super().__init__(input_col, output_col, labels)
        self.final_label = final_label
        self.unknown_label = unknown_label

    def process(self, df: pd.DataFrame):
        if self.labels is None:
            LOGGER.info("No Post processing done")
            return df

        label_cols = [
            col for col in df.columns if col.endswith(self.output_col)
        ]

        mode = df[label_cols].mode(axis=1)

        df[self.final_label] = np.where(
            mode.eq(self.unknown_label).any(axis=1) | mode.isna().any(axis=1),
            self.unknown_label,
            mode[0],
        )

        return df
