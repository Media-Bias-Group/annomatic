import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import pandas as pd

from annomatic.annotator.annotation import AnnotationProcess, DefaultAnnotation
from annomatic.llm.base import Model, ModelLoader, ResponseList
from annomatic.prompt import Prompt

LOGGER = logging.getLogger(__name__)


class PostProcessor(ABC):
    """
    Base class for post processors.
    """

    @abstractmethod
    def process(
        self,
        df: pd.DataFrame,
        input_col,
        output_col,
        labels,
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

    def process(
        self,
        df: pd.DataFrame,
        input_col: str,
        output_col: str,
        labels: List[str],
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

        if labels is None:
            return df
        df[output_col] = df[input_col].apply(
            lambda x: util.find_label(x, labels),
        )

        return df


class ModelLoadMixin(ABC):
    """
    Mixin for annotator to load a model.

    Attributes:
        model_name (str): The name of the model.
        config (ModelConfig): The configuration of the model.
        system_prompt (Optional[str]): The system prompt.
        lib_args (Optional[Dict[str, Any]]): The library arguments.
    """

    def __init__(
        self,
        model_name: str,
        config: ModelConfig,
        system_prompt: Optional[str] = None,
        lib_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.config = config
        self.system_prompt = system_prompt
        self.lib_args = lib_args or {}
        self._model: Optional[Model] = None
        super().__init__(**kwargs)

    @abstractmethod
    def _load_model(self) -> Model:
        """
        Loads the model and store it in self.model.

        Returns:
            The loaded model.
        """
        raise NotImplementedError()

    def update_config_generation_args(
        self,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        for key, value in (generation_args or {}).items():
            if (
                hasattr(self.config, key)
                and getattr(self.config, key) != value
            ):
                setattr(self.config, key, value)
            else:
                self.config.kwargs[key] = value


class FewShotMixin(ABC):
    """
    Mixin for annotator to load a few-shot examples.

    Attributes:
        context (Optional[pd.DataFrame]): The context for the ICL prompt.
        icl_prompt (Optional[Prompt]): The ICL prompt.
    """

    def __init__(self, **kwargs):
        self.context: Optional[pd.DataFrame] = None
        self.icl_prompt: Optional[Prompt] = None

    def set_context(
        self,
        context: Union[Retriever, pd.DataFrame],
        prompt: Optional[Prompt] = None,
    ) -> None:
        """
        Sets the context for the ICL prompt. The context can be either a
        Retriever or a pd.DataFrame.

        Args:
            context: the context for the ICL prompt
            icl_prompt: a specific prompt used for the examples. If no
                additional prompt is set, the regular prompt is used and the
                examples are added at the end.
        """
        self.context = context
        self.icl_prompt = prompt

    def create_context_part(
        self,
        query: Optional[str],
        **kwargs,
    ) -> str:
        """
        Creates an ICL prompt. If the label is known, it is added to the
        prompt at the end.

        Args:
            query: the sentence to get the icl context for
            kwargs: a dict containing the input variables for templates

        Returns:
            str: the ICL prompt part.
        """

        # if no special icl prompt set use regular prompt
        if self.icl_prompt is None:
            if hasattr(self, "_prompt"):
                self.icl_prompt = self._prompt
            else:
                raise ValueError("Prompt is not set!")

        label_var = self.icl_prompt.get_label_variable()
        if label_var is None:
            raise ValueError("Label variable not found in the ICL prompt.")

        if self.context is None or label_var is None:
            raise ValueError("Examples are not set!")

        pred_label = None
        message = ""

        if isinstance(self.context, Retriever):
            context = self.context.select(query=query)
        else:
            context = self.context

        for idx, row in context.iterrows():
            row_dict: Dict[str, Any] = row.to_dict()

            if label_var in row_dict:
                pred_label = row_dict[label_var]

            row_dict[label_var] = kwargs[label_var]
            prompt = self.icl_prompt(**row_dict)

            if pred_label is not None:
                prompt += f"{pred_label}\n\n"
            else:
                prompt += "\n\n"

            message += prompt

        return message


class BaseAnnotator(FewShotMixin, ABC):
class BaseAnnotator(ABC):
    """
    Base class for annotator classes
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        annotation_process: AnnotationProcess = DefaultAnnotation(),
        batch_size: Optional[int] = None,
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
        self._model: Optional[Model] = None
        self._prompt: Optional[Prompt] = None
        self.post_processor = post_processor

        self.model_loader = model_loader
        self.annotation_process = annotation_process

        # TODO make lazy loading possible
        self._model = self.model_loader.load_model()

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

    @abstractmethod
    def store_annotated_data(self, output_data: pd.DataFrame):
        """
        Stores the annotated data in a csv file.

        Args:
            output_data: a list of dicts containing the annotated data

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

    # TODO remove how to handle this?
    def _model_predict(self, messages: List[str]) -> ResponseList:
        """
        Wrapper of the model predict method.

        Args:
            messages: List[str] representing the input messages.

        Returns:
            ResponseList: an object containing the Responses.
        """
        if self._model is None:
            raise ValueError("Model is not initialized!")

        return self._model.predict(messages=messages)

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
