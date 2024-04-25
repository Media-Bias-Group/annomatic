import logging
from abc import ABC
from typing import List, Optional, Union

import pandas as pd
from haystack import component
from haystack.components.builders import PromptBuilder

from annomatic.annotator.annotation import AnnotationProcess, DefaultAnnotation
from annomatic.annotator.base import BaseAnnotator
from annomatic.annotator.postprocess import DefaultPostProcessor, PostProcessor
from annomatic.io.base import BaseOutput
from annomatic.io.file import create_input_handler, create_output_handler

LOGGER = logging.getLogger(__name__)


class AnnotatorEnsemble(ABC):
    def __init__(
        self,
        annotators: List[BaseAnnotator],
        prompt: Union[PromptBuilder, str] = None,
        output_handler: Optional[BaseOutput] = None,
        out_path: Optional[str] = None,
        out_format: Optional[str] = None,
    ):
        if annotators is None:
            annotators = []
        self.annotators = annotators
        self.prompt = prompt
        self.data: Optional[pd.DataFrame] = None
        self.data_variable: Optional[str] = None
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

        # set data for all annotators
        for annotator in self.annotators:
            annotator.set_data(self.data, data_variable=self.data_variable)

    def set_prompt(self, prompt: Union[PromptBuilder, str]):
        """
        Sets the prompt for the annotator ensemble.

        Args:
            prompt: Union[PromptBuilder, str] representing the prompt for the
                    annotator ensemble.
        """
        if not (isinstance(prompt, PromptBuilder) or isinstance(prompt, str)):
            raise ValueError(
                "Invalid input type! "
                "Only PromptBuilder or str is supported.",
            )

        if isinstance(prompt, str):
            prompt = PromptBuilder(prompt)

        for a in self.annotators:
            a.set_prompt(prompt)

    def annotate(
        self,
        data: Optional[pd.DataFrame] = None,
        data_variable: Optional[str] = None,
        return_df: bool = False,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Annotates the input data with the annotators in the ensemble.

        Args:
            data: Optional[Any] representing the input data. If not provided,
                  the data set with set_data will be used.
            return_df: bool representing whether to return the annotated data
                       as a DataFrame or not.
            **kwargs: Optional arguments to be passed to the annotators.
        """
        # TODO start result with only the data variable

        if data is not None:
            self.data = data
        if data_variable is not None:
            self.data_variable = data_variable

        # verify inputs
        if self.data is not None and self.data_variable is not None:
            res = pd.DataFrame(self.data[self.data_variable])
        else:
            raise ValueError("Input data is not set!")

        for idx, annotator in enumerate(self.annotators):
            # find the name of the annotator
            model_name = annotator.model_name()
            if model_name is None:
                name = f"annotator_{idx}"
            else:
                # get last part of the model name
                name = model_name.split("/")[-1]

            LOGGER.info(f"Starting annotation with {name}")

            # annotate data with annotator and map the columns
            annotator_data = annotator.annotate(self.data, return_df=True)
            annotator_data = annotator_data.rename(
                columns={
                    col: name + "_" + col if col != self.data_variable else col
                    for col in annotator_data.columns
                },
            )
            LOGGER.info(f"Annotating with {name} done.")
            # merge the data with the previous annotator
            res = res.merge(annotator_data, on=self.data_variable, how="left")
            # TODO release memory

        self._output_handler.write(res)

        return res if return_df else None

    def add_annotator(self, annotator):
        self.annotators.append(annotator)

    @classmethod
    def from_annotators(
        cls,
        annotators: List[BaseAnnotator],
        prompt: Union[PromptBuilder, str] = None,
        output_handler: Optional[BaseOutput] = None,
        out_path: Optional[str] = None,
        out_format: Optional[str] = None,
        **kwargs,
    ):
        return cls(
            annotators=annotators,
            prompt=prompt,
            output_handler=output_handler,
            out_path=out_path,
            out_format=out_format,
        )

    @classmethod
    def from_models(
        cls,
        models: List[component],
        prompt: Union[PromptBuilder, str],
        annotation_process: AnnotationProcess = DefaultAnnotation(),
        post_processor: Optional[PostProcessor] = DefaultPostProcessor(),
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Creates an AnnotatorEnsemble from a list of models.

        Args:
            models: representing the models used in the ensemble.
            prompt: prompt for the ensemble.
            annotation_process: the annotation process.
            post_processor: the post-processor.
            labels: representing the target labels.

            **kwargs: Optional arguments to be passed to the annotators.
        """
        from annomatic.annotator import FileAnnotator
        from annomatic.io.base import DummyOutput

        annotators = [
            FileAnnotator.from_model(
                model,
                annotation_process=annotation_process,
                post_processor=post_processor,
                labels=labels,
                output_handler=DummyOutput(),
            )
            for model in models
        ]

        return cls(annotators=annotators, prompt=prompt, **kwargs)
