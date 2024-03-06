from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd

from annomatic.annotator import util


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
        Processes the model output before it is stored.

        Args:
             df: the model output to be processed as a DataFrame
             input_col: the input column
             output_col: the output column
             labels: the labels to be used for soft parsing

        Returns:
            the processed model output to be stored as a DataFrame
        """
        raise NotImplementedError()


class DefaultPostProcessor(PostProcessor):
    """
    Base class for post processors.
    Post processors are used to process the model output before it is stored.
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
        Processes the model output before it is stored.

        Finds the label in the model output and stores it in the output column.

        If labels are not known, the model output is stored as is.

        Args:
                df: the model output to be processed as a DataFrame
                input_col: the input column
                output_col: the output column
                labels: the labels to be used for soft parsing

        Returns:
            the processed model output to be stored as a DataFrame
        """

        if self.labels is None:
            return df
        df[self.output_col] = df[self.input_col].apply(
            lambda x: util.find_label(x, self.labels),
        )

        return df
