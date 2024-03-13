from typing import List, Optional, Union

import pandas as pd

from annomatic.annotator.annotation import AnnotationProcess, DefaultAnnotation
from annomatic.annotator.base import LOGGER, BaseAnnotator
from annomatic.config.base import HuggingFaceConfig, OpenAiConfig, VllmConfig
from annomatic.io.base import BaseOutput
from annomatic.io.file import create_input_handler, create_output_handler
from annomatic.llm.huggingface.loader import HuggingFaceModelLoader
from annomatic.llm.openai.loader import OpenAiModelLoader
from annomatic.llm.vllm.loader import VllmModelLoader
from annomatic.prompt import Prompt


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
        model,
        annotation_process: AnnotationProcess = DefaultAnnotation(),
        output_handler: Optional[BaseOutput] = None,
        out_path: Optional[str] = None,
        out_format: Optional[str] = None,
        labels: Optional[List[str]] = None,
        batch_size: int = 1,  # default to 1 for non-batch models
        **kwargs,
    ):
        super().__init__(
            model=model,
            annotation_process=annotation_process,
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

    def set_context(
        self,
        context: Union[pd.DataFrame, str],
        prompt: Union[str, Prompt, None] = None,
    ):
        """
        Sets the context for context-based annotations.
        """
        if self.annotation_process is None:
            raise ValueError("Annotation process is not set!")

        if isinstance(prompt, str):
            prompt = Prompt(content=prompt)

        self.annotation_process.set_context(context, prompt or self._prompt)

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

        # check that all required components are set
        if self.annotation_process is None:
            raise ValueError("Annotation process is not set!")

        if self.post_processor is None:
            raise ValueError("Post processor is not set!")

        if self._output_handler is None:
            raise ValueError("Output handler is not set!")

        if data is not None:
            self.set_data(
                data=data,
                data_variable=kwargs.get("data_variable", "input"),
            )

        if self._prompt is None:
            self.set_prompt(prompt=kwargs.get("prompt", None))

        self._validate_labels(**kwargs)

        if self._labels is not None:
            self.post_processor.labels = self._labels
            self.annotation_process.labels = self._labels

        if (
            self._model is None
            or self._prompt is None
            or self.data_variable is None
        ):
            raise ValueError(
                "Model, prompt or data variable is not set! ",
            )

        # Lazy warm-up of the model (if needed)
        if hasattr(self._model, "pipeline") and self._model.pipeline is None:
            self._model.warm_up()

        annotated_data = self.annotation_process.annotate(
            model=self._model,
            prompt=self._prompt,
            data=self.data,
            data_variable=self.data_variable,
            batch_size=self.batch_size,
            **kwargs,
        )

        self.post_processor.process(df=annotated_data)
        self._output_handler.write(annotated_data)

        if return_df:
            return annotated_data
        else:
            return None


class OpenAiFileAnnotator(FileAnnotator):
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
        model_loader: Optional[OpenAiModelLoader] = None,
        model_name: Optional[str] = None,
        config: Optional[OpenAiConfig] = None,
        batch_size: Optional[int] = None,
        labels: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        out_path: str = "",
        out_format: str = "csv",
        api_key: str = "",
        **kwargs,
    ):
        if model_loader is None:
            if model_name is None:
                raise ValueError(
                    "Model loader or model name must be provided!",
                )
            model_loader = OpenAiModelLoader(
                model_name=model_name,
                config=config,
                batch_size=batch_size,
                labels=labels,
                system_prompt=system_prompt,
                api_key=api_key,
                **kwargs,
            )

        super().__init__(
            model_loader=model_loader,
            labels=labels,
            out_path=out_path,
            out_format=out_format,
            **kwargs,
        )


class VllmFileAnnotator(FileAnnotator):
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
        model_loader: Optional[VllmModelLoader] = None,
        model_name: Optional[str] = None,
        config: Optional[VllmConfig] = None,
        batch_size: Optional[int] = None,
        labels: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        out_path: str = "",
        out_format: str = "csv",
        **kwargs,
    ):
        if model_loader is None:
            if model_name is None:
                raise ValueError(
                    "Model loader or model name must be provided!",
                )
            model_loader = VllmModelLoader(
                model_name=model_name,
                config=config,
                batch_size=batch_size,
                labels=labels,
                system_prompt=system_prompt,
                **kwargs,
            )

        super().__init__(
            model_loader=model_loader,
            labels=labels,
            out_path=out_path,
            out_format=out_format,
            **kwargs,
        )


class HuggingFaceFileAnnotator(FileAnnotator):
    """
    Annotator class for HuggingFace models that work with file inputs
    and outputs.

    This class can use LLMs loaded by the AutoModelForCausalLM and
    AutoModelForSeq2SeqLM classes.

    Arguments:
        model_loader (Optional[HuggingFaceModelLoader]): Model loader for the
                                                        HuggingFace model.
        model_name (Optional[str]): Name of the model.
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
        model_loader: Optional[HuggingFaceModelLoader] = None,
        model_name: Optional[str] = None,
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
        if model_loader is None:
            if model_name is None:
                raise ValueError(
                    "Model loader or model name must be provided!",
                )
            model_loader = HuggingFaceModelLoader(
                model_name=model_name,
                config=config,
                batch_size=batch_size,
                labels=labels,
                system_prompt=system_prompt,
                auto_model=auto_model,
                use_chat_template=use_chat_template,
                **kwargs,
            )

        super().__init__(
            model_loader=model_loader,
            out_path=out_path,
            out_format=out_format,
            labels=labels,
            **kwargs,
        )
