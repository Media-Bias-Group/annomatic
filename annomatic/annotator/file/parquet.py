from typing import Any, Dict, List, Optional

from annomatic.annotator.base import (
    HuggingFaceAnnotator,
    OpenAiAnnotator,
    VllmAnnotator,
)
from annomatic.annotator.file.base import FileAnnotator
from annomatic.config.base import ModelConfig


class ParquetAnnotator(FileAnnotator):
    """
    Annotator class for models that store the output to a Parquet file.

    Arguments:
        model_name (str): Name of the model.
        model_lib (str): Name of the model library.
        config (ModelConfig): Configuration for the model.
        batch_size (int): Size of the batch.
        labels (List[str]): List of labels that should be
                            used for soft parsing.
        system_prompt (Optional[str]): System prompt for the annotator.
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
            out_path=out_path,
            out_format="parquet",
            lib_args=lib_args or {},
            **kwargs,
        )


class OpenAiParquetAnnotator(OpenAiAnnotator, ParquetAnnotator):
    """
    Annotator class for OpenAI models that use Parquet files as input
    and output.
    """

    pass


class HuggingFaceParquetAnnotator(HuggingFaceAnnotator, ParquetAnnotator):
    """
    Annotator class for HuggingFace models that use Parquet files as output.

    This class can use LLMs loaded by the AutoModelForCausalLM and
    AutoModelForSeq2SeqLM classes.
    """

    pass


class VllmParquetAnnotator(VllmAnnotator, ParquetAnnotator):
    """
    Annotator class for Vllm models that use Parquet files as input and output.
    """

    pass
