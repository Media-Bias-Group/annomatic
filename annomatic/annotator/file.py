from typing import List, Optional, Union

import pandas as pd

from annomatic.annotator.base import LOGGER, BaseAnnotator
from annomatic.config.base import HuggingFaceConfig, OpenAiConfig, VllmConfig
from annomatic.io.base import BaseOutput
from annomatic.io.file import create_input_handler, create_output_handler
from annomatic.llm.huggingface.model import HuggingFaceModelLoader
from annomatic.llm.openai.model import OpenAiModelLoader
from annomatic.llm.vllm.model import VllmModelLoader


class FileAnnotator(BaseAnnotator):
    """
    Base annotator class for models that work with file inputs and outputs.

    Arguments:
        batch_size (int): Size of the batch.
        labels (List[str]): List of labels that should be used
                            for soft parsing.
        output_handler (BaseOutput): Output handler for the annotated data.
        out_path (str): Path to the output file.
        out_format (str): Format of the output file. Supported formats are
            'csv' and 'parquet'. Defaults to 'csv'.
        kwargs: a dict containing additional arguments
    """

    def __init__(
        self,
        batch_size: Optional[int] = None,
        labels: Optional[List[str]] = None,
        output_handler: Optional[BaseOutput] = None,
        out_path: Optional[str] = None,
        out_format: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size,
            labels=labels,
            **kwargs,
        )

        if output_handler:
            self._output_handler = output_handler
        elif out_path and out_format:
            self._output_handler = create_output_handler(
                path=out_path,
                type=out_format,
            )
        else:
            raise ValueError(
                "Must provide either an output_handler "
                "or both out_path and out_format.",
            )

    def set_data(
        self,
        data: Union[pd.DataFrame, str],
        data_variable: str = "input",
        in_format: str = "csv",
        sep: str = ",",
    ):
        """
        Sets the input data for the annotator. The data can be provided as a
        DataFrame or as a path to a file. If the data is provided as a path,
        the format of the file must be specified. Supported formats are 'csv'
        and 'parquet'.

        Args:
            data: Union[pd.DataFrame, str] representing the input data.
            data_variable: str representing the name of the input column.
            in_format: str representing the format of the input file. Only
                       used if data is a string. Supported formats are 'csv',
                          'parquet'.
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
            self.data = create_input_handler(
                path=data,
                type=in_format,
            ).read(sep=sep)
        else:
            raise ValueError(
                "Invalid input type! "
                "Only Dataframe or file path (str) is supported.",
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


class OpenAiFileAnnotator(OpenAiModelLoader, FileAnnotator):
    """
    Annotator class for OpenAI models that use file inputs and outputs.

    Arguments:
        model_name (str): Name of the model.
        config (Optional[OpenAiConfig]): Configuration for the model.
        batch_size (Optional[int]): Size of the batch.
        labels (Optional[List[str]]): List of labels that should be used
                                    for soft parsing.
        system_prompt (Optional[str]): System prompt for the model.
        out_path (str): Path to the output file.
        out_format (str): Format of the output file.
        api_key (str): API key for the OpenAI model.
        kwargs: a dict containing additional arguments

    """

    def __init__(
        self,
        model_name: str,
        config: Optional[OpenAiConfig] = None,
        batch_size: Optional[int] = None,
        labels: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        out_path: str = "",
        out_format: str = "csv",
        api_key: str = "",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            config=config,
            batch_size=batch_size,
            labels=labels,
            system_prompt=system_prompt,
            out_path=out_path,
            out_format=out_format,
            api_key=api_key,
            **kwargs,
        )


class HuggingFaceFileAnnotator(HuggingFaceModelLoader, FileAnnotator):
    """
    Annotator class for HuggingFace models that work with file inputs
    and outputs.

    This class can use LLMs loaded by the AutoModelForCausalLM and
    AutoModelForSeq2SeqLM classes.

    Arguments:
        model_name (str): Name of the model.
        config (Optional[HuggingFaceConfig]): Configuration for the model.
        batch_size (Optional[int]): Size of the batch.
        labels (Optional[List[str]]): List of labels that should be used
                                        for soft parsing.
        system_prompt (Optional[str]): System prompt for the model.
        out_path (str): Path to the output file.
        out_format (str): Format of the output file.
        auto_model (str): Name of the AutoModel class to be used.
        use_chat_template (bool): Whether to use the chat template.
        kwargs: a dict containing additional arguments
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[HuggingFaceConfig] = None,
        batch_size: Optional[int] = None,
        labels: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        out_path: str = "",
        out_format: str = "csv",
        auto_model: str = "AutoModelForCausalLM",
        use_chat_template: bool = False,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            config=config,
            batch_size=batch_size,
            labels=labels,
            system_prompt=system_prompt,
            out_path=out_path,
            out_format=out_format,
            auto_model=auto_model,
            use_chat_template=use_chat_template,
            **kwargs,
        )


class VllmFileAnnotator(VllmModelLoader, FileAnnotator):
    """
    Annotator class for Vllm models that use file inputs and outputs.

    Arguments:
        model_name (str): Name of the model.
        config (Optional[VllmConfig]): Configuration for the model.
        batch_size (Optional[int]): Size of the batch.
        labels (Optional[List[str]]): List of labels that should be used
                                        for soft parsing.
        system_prompt (Optional[str]): System prompt for the model.
        out_path (str): Path to the output file.
        out_format (str): Format of the output file.
        kwargs: a dict containing additional arguments

    """

    def __init__(
        self,
        model_name: str,
        config: Optional[VllmConfig] = None,
        batch_size: Optional[int] = None,
        labels: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        out_path: str = "",
        out_format: str = "csv",
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            config=config,
            batch_size=batch_size,
            labels=labels,
            system_prompt=system_prompt,
            out_path=out_path,
            out_format=out_format,
            **kwargs,
        )
