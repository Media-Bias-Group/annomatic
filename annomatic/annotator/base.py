import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import pandas as pd

from annomatic.annotator.annotation import AnnotationProcess, DefaultAnnotation
from annomatic.annotator.postprocess import DefaultPostProcessor, PostProcessor
from annomatic.prompt import Prompt

LOGGER = logging.getLogger(__name__)


class BaseAnnotator(ABC):
    """
    Base class for annotator classes
    """

    def __init__(
        self,
        model,
        annotation_process: AnnotationProcess = DefaultAnnotation(),
        batch_size: int = 1,
        labels: Optional[List[str]] = None,
        post_processor: Optional[PostProcessor] = DefaultPostProcessor(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self._labels = labels
        self.kwargs = kwargs
        self.data: Optional[pd.DataFrame] = None
        self.data_variable: Optional[str] = None
        self._prompt: Optional[Prompt] = None
        self.post_processor = post_processor
        self.annotation_process = annotation_process
        self._model = model

    @abstractmethod
    def annotate(
        self,
        data: Optional[Any] = None,
        return_df: bool = False,
        **kwargs,
    ):
        """
        Annotates the input data and stores the annotated data.

        Args:
            data: the input data
            return_df: bool indicating if the annotated data should be returned
            kwargs: a dict containing the input variables for prompt templates
        """
        raise NotImplementedError()

    @abstractmethod
    def set_data(
        self,
        data: Any,
        data_variable: str,
    ):
        """
        Sets the data to be annotated.

        Args:
            data: the input data
            data_variable: the variable name of the input data
        """
        raise NotImplementedError()

    def _validate_data_variable(self) -> bool:
        """
        Validates the data variable.

        If a prompt is set, the data variable is valid if it occurs in the
        prompt. Otherwise, the data variable is valid if it is not None.


        Returns:
            bool: True if the data variable is valid, False otherwise.
        """
        if self._prompt is None or self.data_variable is None:
            # no validation possible
            return True

        return self.data_variable in self._prompt.get_variables()

    def set_prompt(self, prompt: Union[Prompt, str]):
        if self._prompt is not None:
            LOGGER.info("Prompt is already set. Will be overwritten.")

        if isinstance(prompt, Prompt):
            self._prompt = prompt

            if not self._validate_data_variable():
                raise ValueError("Input column does not occur in prompt!")

        elif isinstance(prompt, str):
            self._prompt = Prompt(content=prompt)
            if not self._validate_data_variable():
                raise ValueError("Input column does not occur in prompt!")
        else:
            raise ValueError(
                "Invalid input type! " "Only Prompt or str is supported.",
            )

    def _validate_labels(self, **kwargs):
        if self._labels is None:
            prompt_labels = self._prompt.get_label_variable()
            labels_from_kwargs = kwargs.get(prompt_labels, None)

            if labels_from_kwargs is not None:
                self._labels = labels_from_kwargs
        else:
            prompt_labels = self._prompt.get_label_variable()
            labels_from_kwargs = kwargs.get(prompt_labels)

            if labels_from_kwargs is not None and set(self._labels) != set(
                labels_from_kwargs,
            ):
                raise ValueError(
                    "Labels in prompt and Annotator do not match!",
                )
