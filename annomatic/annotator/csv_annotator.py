import logging
from abc import ABC, abstractmethod

import pandas as pd
from typing import Optional

from annomatic.annotator.base import BaseAnnotator
from annomatic.io.file import CsvInput, CsvOutput
from annomatic.model.base import Response
from annomatic.model.openai.open_ai import OpenAiModel
from annomatic.prompt.prompt import Prompt

LOGGER = logging.getLogger(__name__)


class CSVMixin(ABC):
    """
    Mixin class for annotators that use CSV files as input and output.
    """

    @abstractmethod
    def __init__(self, in_path: str, out_path: str):
        """
        Arguments:
            in_path (str): Path to the input file.
            out_path (str): Path to the output file.
        """
        self._input_handler = CsvInput(in_path)
        self._output_handler = CsvOutput(out_path)


class OpenAICsvAnnotator(BaseAnnotator, CSVMixin):
    """
    Annotator class for OpenAI models that use CSV files as input and output.
    """

    def __init__(
        self,
        in_path: str,
        out_path: str,
        in_col: str = "input",
        model: str = "gpt-3.5-turbo",
        api_key: str = "",
        temperature: float = 0.0,
    ):
        """
        Arguments:
            in_path: str representing the Path to the input file.
            out_path: str representing the Path to the output file.
            in_col: str representing the Name of the input column.
            model: str representing the Name of the OpenAI model.
            api_key: str representing the OpenAI API key.
            temperature: float value for Temperature for the model.
        """
        super().__init__(in_path, out_path)
        self._in_col = in_col
        self._prompt = Optional[Prompt]

        self._model = OpenAiModel(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )

    def set_prompt(self, prompt: Prompt):
        """
        Sets the prompt for the annotator.

        Args:
            prompt: Prompt object
        """
        self._prompt = prompt

    def annotate(self, **kwargs):
        """
        Annotates the input CSV file and writes the annotated data to the
        output CSV file.

        Args:
            kwargs: a dict containing the input variables for templates
        """
        output_data = []

        if self._prompt is None:
            raise ValueError("Prompt is not set")

        try:
            for idx, row in self._input_handler.read().iterrows():
                text_prompt = row[self._in_col]

                # extend kwargs with the current row
                kwargs[self._in_col] = text_prompt
                try:
                    content = self._prompt.to_string(**kwargs)
                    prediction: Response = self._model.predict(content=content)

                    output_data.append(
                        {
                            self._in_col: text_prompt,
                            "label": prediction.answer,
                        },
                    )
                except Exception as prediction_error:
                    # Handle the prediction error
                    LOGGER.error(f"Prediction error: {str(prediction_error)}")

        except Exception as read_error:
            # Handle the input reading error
            LOGGER.error(f"Input reading error: {str(read_error)}")

        # Write the annotated data to the output CSV file
        try:
            output_df = pd.DataFrame(output_data)
            self._output_handler.write(output_df)
        except Exception as write_error:
            LOGGER.error(f"Output writing error: {str(write_error)}")
