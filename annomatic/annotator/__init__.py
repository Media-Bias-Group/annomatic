from .file.csv import (
    CsvAnnotator,
    HuggingFaceCsvAnnotator,
    OpenAiCsvAnnotator,
    VllmCsvAnnotator,
)
from .file.parquet import (
    HuggingFaceParquetAnnotator,
    OpenAiParquetAnnotator,
    ParquetAnnotator,
    VllmParquetAnnotator,
)
