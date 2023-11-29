import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from annomatic.annotator.base import BaseAnnotator
from annomatic.config.base import (
    HuggingFaceConfig,
    ModelConfig,
    OpenAiConfig,
    VllmConfig,
)
from annomatic.io import CsvInput, CsvOutput

LOGGER = logging.getLogger(__name__)


class CsvAnnotator(BaseAnnotator):
    """
    Base annotator class for models that stores the output to a csv file.

    Arguments:
        model_name (str): Name of the model.
        model_lib (str): Name of the model library.
        model_args (dict): Arguments for the model.
        batch_size (int): Size of the batch.
        labels (List[str]): List of labels that should be used for soft
            parsing.
        out_path (str): Path to the output file.
        lib_args (dict): Special arguments for model libs (used for creation).
        kwargs: a dict containing additional arguments
    """

    def __init__(
        self,
        model_name: str,
        model_lib: str,
        config: ModelConfig,
        batch_size: Optional[int] = None,
        labels: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        out_path: str = "",
        lib_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            model_lib=model_lib,
            config=config,
            batch_size=batch_size,
            labels=labels,
            system_prompt=system_prompt,
            lib_args=lib_args or {},
            **kwargs,
        )

        self.out_path = out_path
        self._output_handler: Optional[CsvOutput] = CsvOutput(out_path)

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

    def set_data(
        self,
        data: Union[pd.DataFrame, str],
        data_variable: str = "input",
        sep: str = ",",
    ):
        """
        Sets the input data for the annotator.



        Args:
            data: Union[pd.DataFrame, str] representing the input data.
            data_variable: str representing the name of the input column.
            sep: str representing the separator for the CSV file.
        """
        if self.data is not None:
            LOGGER.info("Input data is already set. Will be overwritten.")

        self.data_variable = data_variable

        if not self._validate_data_variable():
            raise ValueError("Input column does not occur in prompt!")

        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, str):
            self.data = CsvInput(data).read(sep=sep)
        else:
            raise ValueError(
                "Invalid input type! "
                "Only Dataframe or CSV file path is supported.",
            )

    def annotate(
        self,
        data: Union[pd.DataFrame, str] = None,
        return_df: bool = False,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Annotates the input data and writes the annotated data to the
        output CSV file. Also performs some setup if needed.

        This method also accepts the arguments for set_data and set_prompt.


        Args:
            data: Union[pd.DataFrame, str] representing the input data.
            return_df: bool whether to return the annotated data as a DataFrame
            kwargs: a dict containing the input variables for templates
        """
        if data is not None:
            self.set_data(
                data=data,
                data_variable=kwargs.get("data_variable", "input"),
            )

        if self._prompt is None:
            self.set_prompt(prompt=kwargs.get("prompt", None))

        self._validate_labels(**kwargs)

        annotated_data = self._annotate(**kwargs)

        if return_df:
            return annotated_data
        else:
            return None

    def get_num_samples(self):
        """
        Returns the number of data instances to be annotated.
        """
        return self.data.shape[0]

    def store_annotated_data(self, output_data: pd.DataFrame):
        """
        Write the output data to the output CSV file.

        Args:
            output_data: List[dict] representing the output data.
        """
        if self._output_handler is None:
            raise ValueError("Output handler is not set!")

        self._output_handler.write(output_data)

    def fill_prompt(self, batch: pd.DataFrame, **kwargs) -> List[str]:
        """
        Creates the prompt passed to the model.

        Args:
            batch: pd.DataFrame representing the input data.
            kwargs: a dict containing the input variables for templates(
        """
        if self._prompt is None:
            raise ValueError("Prompt is not set!")

        messages = []
        for index, row in batch.iterrows():
            kwargs[str(self.data_variable)] = row[str(self.data_variable)]
            messages.append(self._prompt(**kwargs))

        return messages


class OpenAiCsvAnnotator(CsvAnnotator):
    """
    Annotator class for OpenAI models that use CSV files as input and output.

    This class can use LLMs loaded by the OpenAiModel class.
    """

    DEFAULT_BATCH_SIZE = 1

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gpt-3.5-turbo",
        config: Optional[OpenAiConfig] = None,
        generation_args: Optional[dict] = None,
        system_prompt: Optional[str] = None,
        out_path: str = "",
        labels: Optional[List[str]] = None,
    ):
        super().__init__(
            model_name=model_name,
            model_lib="openai",
            lib_args={"api_key": api_key},
            config=config or OpenAiConfig(),
            out_path=out_path,
            system_prompt=system_prompt,
            batch_size=OpenAiCsvAnnotator.DEFAULT_BATCH_SIZE,
            labels=labels,
        )

        self.update_config_generation_args(generation_args)
        self.api_key = api_key

        self.lib_args = {"api_key": api_key}


class HFAutoModels(Enum):
    AutoModelForCausalLM = "AutoModelForCausalLM"
    AutoModelForSeq2SeqLM = "AutoModelForSeq2SeqLM"


class HuggingFaceCsvAnnotator(CsvAnnotator):
    """
    Annotator class for HuggingFace models that use CSV files as output.

    This class can use LLMs loaded by the AutoModelForCausalLM and
    AutoModelForSeq2SeqLM classes.
    """

    DEFAULT_BATCH_SIZE = 5

    def __init__(
        self,
        model_name: str,
        out_path: str,
        config: Optional[HuggingFaceConfig] = None,
        model_args: Optional[Dict[str, Any]] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        auto_model: str = "AutoModelForCausalLM",
        use_chat_template: bool = False,
        labels: Optional[List[str]] = None,
    ):
        """
        Arguments:
            model_name: str representing the Name of the HuggingFace model
            auto_model: str representing the AutoModel class
            out_path: str representing the path to the output file
            model_args: dict representing the model 'arguments'
            tokenizer_args: dict representing the token arguments
        """
        super().__init__(
            model_name=model_name,
            model_lib="huggingface",
            lib_args={
                "auto_model": auto_model,
                "use_chat_template": use_chat_template,
            },
            config=config or HuggingFaceConfig(),
            out_path=out_path,
            system_prompt=system_prompt,
            batch_size=batch_size,
            labels=labels,
        )

        # Override values in config
        if hasattr(self.config, "model_args"):
            self.config.model_args = (
                getattr(
                    self.config,
                    "model_args",
                    {},
                )
                or {}
            )
            self.config.model_args.update(model_args or {})

        if hasattr(self.config, "tokenizer_args"):
            self.config.tokenizer_args = (
                getattr(
                    self.config,
                    "tokenizer_args",
                    {},
                )
                or {}
            )
            self.config.tokenizer_args.update(tokenizer_args or {})

        self.update_config_generation_args(generation_args)


class VllmCsvAnnotator(CsvAnnotator):
    """
    Annotator class for Vllm models that use CSV files as input and output.

    This class can use LLMs loaded by the VllmModel class.
    """

    def __init__(
        self,
        model_name: str,
        out_path: str,
        config: Optional[VllmConfig] = None,
        model_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        batch_size: Optional[int] = None,
        labels: Optional[List[str]] = None,
    ):
        """
        Arguments:
            model_name: str representing the Name of the OpenAI model.
        """
        super().__init__(
            model_name=model_name,
            model_lib="vllm",
            lib_args={},
            config=config or VllmConfig(),
            out_path=out_path,
            system_prompt=system_prompt,
            batch_size=batch_size,
            labels=labels,
        )

        if hasattr(self.config, "model_args"):
            self.config.model_args = (
                getattr(
                    self.config,
                    "model_args",
                    {},
                )
                or {}
            )
            self.config.model_args.update(model_args or {})

        self.update_config_generation_args(generation_args)
