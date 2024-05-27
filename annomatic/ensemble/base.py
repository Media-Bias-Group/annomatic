import logging
from abc import ABC
from typing import List, Optional, Union

import pandas as pd
from haystack import component
from haystack.components.builders import PromptBuilder

from annomatic.annotator.annotation import (
    AnnotationProcess,
    DefaultAnnotationProcess,
)
from annomatic.annotator.base import BaseAnnotator
from annomatic.annotator.postprocess import (
    DefaultPostProcessor,
    MajorityVote,
    PostProcessor,
)
from annomatic.io.base import BaseOutput
from annomatic.io.file import create_input_handler, create_output_handler

LOGGER = logging.getLogger(__name__)


class AnnotatorEnsemble(ABC):
    def __init__(
        self,
        annotators: List[BaseAnnotator],
        output: Union[BaseOutput, str],
        labels: Optional[List[str]] = None,
        post_processor: Optional[PostProcessor] = None,
    ):
        if annotators is None:
            annotators = []
        self.annotators = annotators
        self.data: Optional[pd.DataFrame] = None
        self.data_variable: Optional[str] = None
        self.labels = labels
        self.post_processor = post_processor

        if isinstance(output, str):
            output = create_output_handler(output)
        elif not isinstance(output, BaseOutput):
            raise ValueError(
                "Please provide eather a path with a "
                "supported filetype or a BaseOutput",
            )
        self.output = output

    def get_common_labels(self, annotators):
        labels = set()
        for obj in annotators:
            if hasattr(obj, "labels"):
                labels.add(obj.labels)

        if len(labels) == 1:
            return labels.pop()
        else:
            return None

    def set_input(
        self,
        data: Union[pd.DataFrame, str],
        data_variable: str = "input",
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
            ).read(sep=sep)
        else:
            raise ValueError(
                "Invalid input type! "
                "Only Dataframe or file path (str) is supported.",
            )

        # set data for all annotators
        for annotator in self.annotators:
            annotator.set_input(self.data, data_variable=self.data_variable)

    def set_prompt(self, prompt: Union[PromptBuilder, str]):
        """
        Sets the prompt for the annotator ensemble.

        Args:
            prompt: Union[PromptBuilder, str] representing the prompt for the
                    annotator ensemble.
        """
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
                  the data set with set_input will be used.
            return_df: bool representing whether to return the annotated data
                       as a DataFrame or not.
            **kwargs: Optional arguments to be passed to the annotators.
        """
        if data is not None:
            self.data = data
        if data_variable is not None:
            self.data_variable = data_variable

        # if labels and no post_processor make majority vote
        if not self.post_processor:
            labels_annotation = self.get_common_labels(self.annotators)
            self.labels = (
                labels_annotation
                if self.labels is None
                else self.labels
                if labels_annotation is None
                else labels_annotation
                if self.labels == labels_annotation
                else None
            )
            self.post_processor = MajorityVote(labels=self.labels)

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

            # release memory after use (for local hf models)
            if (
                hasattr(annotator, "pipeline")
                and annotator.pipeline is not None
            ):
                import torch

                del annotator.pipeline
                torch.cuda.empty_cache()

        if self.post_processor:
            res = self.post_processor.process(res)

        self.output.write(res)

        return res if return_df else None

    def add_annotator(self, annotator):
        self.annotators.append(annotator)

    @classmethod
    def from_annotators(
        cls,
        annotators: List[BaseAnnotator],
        output: Union[BaseOutput, str],
        **kwargs,
    ):
        return cls(
            annotators=annotators,
            output=output,
            **kwargs,
        )

    @classmethod
    def from_models(
        cls,
        models: List[component],
        output: Union[BaseOutput, str],
        prompt: Union[PromptBuilder, str],
        annotation_process: AnnotationProcess = DefaultAnnotationProcess(),
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
                prompt=prompt,
                annotation_process=annotation_process,
                post_processor=post_processor,
                labels=labels,
                output=DummyOutput(),
            )
            for model in models
        ]

        return cls(
            annotators=annotators,
            output=output,
            labels=labels,
            **kwargs,
        )
