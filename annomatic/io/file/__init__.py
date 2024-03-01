from ..base import BaseInput, BaseOutput
from .csv import CsvInput, CsvOutput
from .parquet import ParquetInput, ParquetOutput


def create_input_handler(path: str, type: str) -> BaseInput:
    if type.lower() == "csv":
        return CsvInput(path)
    elif type.lower() == "parquet":
        return ParquetInput(path)
    else:
        raise ValueError(f"Unsupported input type: {type}")


def create_output_handler(path: str, type: str) -> BaseOutput:
    if type.lower() == "csv":
        return CsvOutput(path)
    elif type.lower() == "parquet":
        return ParquetOutput(path)
    else:
        raise ValueError(f"Unsupported output type: {type}")
