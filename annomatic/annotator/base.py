import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import pandas as pd
from haystack.components.builders import PromptBuilder

from annomatic.annotator.annotation import AnnotationProcess, DefaultAnnotation
from annomatic.annotator.postprocess import DefaultPostProcessor, PostProcessor
from annomatic.io.base import BaseOutput

LOGGER = logging.getLogger(__name__)


class BaseAnnotator(ABC):
    """
    Base class for annotator classes
    """

    def __init__(
        self,
        model,
        annotation_process: AnnotationProcess = DefaultAnnotation(),
        post_processor: Optional[PostProcessor] = DefaultPostProcessor(),
        batch_size: int = 1,
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self._labels = labels
        self.kwargs = kwargs
        self.data: Optional[pd.DataFrame] = None
        self.data_variable: Optional[str] = None
        self._prompt: Optional[PromptBuilder] = None
        self.post_processor = post_processor
        self.annotation_process = annotation_process
        self._model = model

    def model_name(self):
        return self._model._get_telemetry_data().get("model")

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
    def set_input(
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
        if (
            isinstance(self._prompt, PromptBuilder)
            or self._prompt is None
            or self.data_variable is None
        ):
            # no validation possible
            return True

        return self.data_variable in self._prompt.get_variables()

    def set_prompt(self, prompt: Union[PromptBuilder, str]):
        if self._prompt is not None:
            LOGGER.info("Prompt is already set. Will be overwritten.")

        if isinstance(prompt, PromptBuilder):
            self._prompt = prompt
        elif isinstance(prompt, str):
            self._prompt = PromptBuilder(prompt)
        else:
            raise ValueError(
                "Invalid input type! "
                "Only PromptBuilder or str is supported.",
            )

    @classmethod
    def from_model(
        cls,
        model,
        annotation_process: AnnotationProcess = DefaultAnnotation(),
        post_processor: Optional[PostProcessor] = DefaultPostProcessor(),
        batch_size: int = 1,
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        return cls(
            model,
            annotation_process=annotation_process,
            post_processor=post_processor,
            batch_size=batch_size,
            labels=labels,
            **kwargs,
        )
