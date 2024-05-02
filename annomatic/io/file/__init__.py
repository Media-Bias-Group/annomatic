import os
from typing import Dict, Type, Union

from ..base import BaseInput, BaseIO, BaseOutput
from .csv import CsvInput, CsvOutput
from .parquet import ParquetInput, ParquetOutput


def _create_handler(
    path: str,
    handlers: Dict[str, Type[BaseIO]],
) -> BaseIO:
    file_extension = os.path.splitext(path)[-1].lower()

    if file_extension in handlers:
        return handlers[file_extension](path)
    else:
        raise ValueError(f"Unsupported type: {file_extension}")


def create_input_handler(path: str) -> BaseInput:
    handler = _create_handler(
        path,
        {".csv": CsvInput, ".parquet": ParquetInput},
    )
    if isinstance(handler, BaseInput):
        return handler
    else:
        raise ValueError("Unexpected handler type for input")


def create_output_handler(path: str) -> BaseOutput:
    handler = _create_handler(
        path,
        {".csv": CsvOutput, ".parquet": ParquetOutput},
    )
    if isinstance(handler, BaseOutput):
        return handler
    else:
        raise ValueError("Unexpected handler type for output")
