from typing import Any, Dict, List, Optional, Union

import pandas as pd
from haystack.components.builders import PromptBuilder

from annomatic.annotator.annotation import (
    AnnotationProcess,
    DefaultAnnotationProcess,
)
from annomatic.annotator.base import LOGGER, BaseAnnotator
from annomatic.annotator.postprocess import DefaultPostProcessor, PostProcessor
from annomatic.io.base import BaseOutput
from annomatic.io.file import create_input_handler, create_output_handler


class FileAnnotator(BaseAnnotator):
    """
    Base annotator class for models that work with file inputs and outputs.

    Arguments:
        batch_size (int): Size of the batch.
        labels (List[str]): List of labels that should be used
                            for soft parsing.
        output_handler (BaseFileOutput): Output handler for the annotated data.
        out_path (str): Path to the output file.
        out_format (str): Format of the output file. Supported formats are
            'csv' and 'parquet'. Defaults to 'csv'.
        kwargs: a dict containing additional arguments
    """

    def __init__(
        self,
        model,
        output: Union[BaseOutput, str],
        prompt: Optional[PromptBuilder] = None,
        annotation_process: AnnotationProcess = DefaultAnnotationProcess(),
        post_processor: Optional[PostProcessor] = DefaultPostProcessor(),
        labels: Optional[List[str]] = None,
        batch_size: int = 1,  # default to 1 for non-batch models
        **kwargs,
    ):
        super().__init__(
            model=model,
            prompt=prompt,
            annotation_process=annotation_process,
            batch_size=batch_size,
            post_processor=post_processor,
            labels=labels,
            **kwargs,
        )

        if isinstance(output, str):
            output = create_output_handler(output)
        elif not isinstance(output, BaseOutput):
            raise ValueError(
                "Please provide eather a path with a "
                "supported filetype or a BaseOutput",
            )
        self.output = output

    def set_context(
        self,
        context: Dict[str, Any],
    ):
        """
        Sets the context for context-based annotations.
        """
        if self.annotation_process is None:
            raise ValueError("Annotation process is not set!")

        self.annotation_process.set_context(context)

    def set_input(
        self,
        data: Union[pd.DataFrame, str],
        data_variable: Optional[str] = "input",
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

        This method also accepts the arguments for set_input and set_prompt.


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

        if self.output is None:
            raise ValueError("Output handler is not set!")

        if data is not None:
            self.set_input(
                data=data,
                data_variable=self.data_variable
                or kwargs.get("data_variable", "input"),
            )

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
        self.output.write(annotated_data)

        if return_df:
            return annotated_data
        else:
            return None
