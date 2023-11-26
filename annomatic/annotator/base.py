import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from annomatic.config.base import (
    HuggingFaceConfig,
    ModelConfig,
    OpenAiConfig,
    VllmConfig,
)
from annomatic.llm.base import Model
from annomatic.prompt import Prompt

LOGGER = logging.getLogger(__name__)


class BaseAnnotator(ABC):
    """
    Base class for annotator classes
    """

    def __init__(
        self,
        model_name: str,
        model_lib: str,
        config: ModelConfig,
        batch_size: Optional[int] = None,
        labels: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        lib_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.model_lib = model_lib
        self.config = config
        self.batch_size = batch_size
        self._labels = labels
        self.system_prompt = system_prompt
        self.lib_args = lib_args or {}
        self.kwargs = kwargs

        self.data: Optional[pd.DataFrame] = None
        self._prompt: Optional[Prompt] = None

    @abstractmethod
    def annotate(self, **kwargs):
        """
        Annotates the input data and stores the annotated data.

        Args:
            kwargs: a dict containing the input variables for prompt templates
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_samples(self):
        """
        Returns the number of data instances to be annotated.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_data(
        self,
        data: Any,
    ):
        """
        Sets the data to be annotated.

        Args:
            data: a pandas DataFrame or a path to a csv file
        """
        raise NotImplementedError()

    @abstractmethod
    def fill_prompt(self, batch: pd.DataFrame, **kwargs) -> List[str]:
        """
        Fills the prompt template with the given kwargs.

        Args:
            batch: pd.DataFrame representing the input data.
            kwargs: a dict containing the input variables for prompt templates

        Returns:
            The filled prompt as a str.
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

    @abstractmethod
    def _validate_input_variable(self) -> bool:
        """
        Validates if the input variable occurs in the prompt.

        Returns:
            True if the input variable occurs in the prompt, False otherwise.
        """
        raise NotImplementedError()

    def set_prompt(self, prompt: Union[Prompt, str]):
        if self._prompt is not None:
            LOGGER.info("Prompt is already set. Will be overwritten.")

        if isinstance(prompt, Prompt):
            self._prompt = prompt

            if not self._validate_input_variable():
                raise ValueError("Input column does not occur in prompt!")

        elif isinstance(prompt, str):
            self._prompt = Prompt(content=prompt)
            if not self._validate_input_variable():
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

    def _num_batches(self, total_rows: int):
        """
        Calculates the number of batches.

        If self.batch_size is not set, the whole dataset is used as a batch.

        Args:
            total_rows: int representing the total number of rows.
        """
        if self.batch_size:
            return total_rows // self.batch_size
        else:
            # if no batch size is set, use the whole dataset as a batch
            self.batch_size = total_rows
            return 1


class ModelLoadMixin(ABC):
    """
    Mixin for annotator to load a model from different libraries.
    """

    def __init__(self):
        self._model: Optional[Model] = None

    def _load_model(
        self,
        model_name: str,
        model_lib: str,
        config: ModelConfig,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Model:
        """
        Loads the model and store it in self.model.

        Returns:
            The loaded model.
        """

        if model_lib == "openai":
            from annomatic.llm.openai import OpenAiModel

            if not isinstance(config, OpenAiConfig):
                raise ValueError(
                    "OpenAI models require a OpenAiConfig object.",
                )

            api_key = kwargs.get("api_key", None)
            self._model = OpenAiModel(
                model_name=model_name,
                api_key=api_key,
                generation_args=config.to_dict(),
            )
            return self._model

        elif model_lib == "huggingface":
            if not isinstance(config, HuggingFaceConfig):
                raise ValueError(
                    "Huggingface models require a HuggingfaceConfig object.",
                )

            model_args = config.model_args
            tokenizer_args = config.tokenizer_args
            generation_args = config.to_dict()
            auto_model = kwargs.get("auto_model", "AutoModelForCausalLM")
            use_chat_template = kwargs.get("use_chat_template", False)

            if auto_model == "AutoModelForCausalLM":
                from annomatic.llm.huggingface import HFAutoModelForCausalLM

                self._model = HFAutoModelForCausalLM(
                    model_name=model_name,
                    model_args=model_args,
                    tokenizer_args=tokenizer_args,
                    generation_args=generation_args,
                    system_prompt=system_prompt,
                    use_chat_template=use_chat_template,
                )
                return self._model
            elif auto_model == "AutoModelForSeq2SeqLM":
                from annomatic.llm.huggingface import HFAutoModelForSeq2SeqLM

                self._model = HFAutoModelForSeq2SeqLM(
                    model_name=model_name,
                    model_args=model_args,
                    tokenizer_args=tokenizer_args,
                    generation_args=generation_args,
                    system_prompt=system_prompt,
                    use_chat_template=use_chat_template,
                )
                return self._model
            else:
                raise ValueError(
                    "auto_model must be either "
                    "'AutoModelForCausalLM' or 'AutoModelForSeq2SeqLM')",
                )
        elif model_lib == "vllm":
            from annomatic.llm.vllm import VllmModel

            if not isinstance(config, VllmConfig):
                raise ValueError(
                    "VLLM models require a VllmConfig object.",
                )
            _model = VllmModel(
                model_name=model_name,
                model_args=config.model_args,
                generation_args=config.to_dict(),
                system_prompt=system_prompt,
            )

            return _model
        else:
            raise ValueError(
                f"Model library {model_lib} not supported."
                f" Please choose from 'openai', 'huggingface', or 'vllm'.",
            )
