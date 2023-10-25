import logging

import pandas as pd
from typing import Optional, Union

from annomatic.annotator.base import BaseAnnotator
from annomatic.io import CsvInput, CsvOutput
from annomatic.llm import Response
from annomatic.llm.base import Model, ResponseList
from annomatic.llm.huggingface import HFAutoModelForCausalLM
from annomatic.llm.openai import OpenAiModel
from annomatic.prompt import Prompt

LOGGER = logging.getLogger(__name__)


class CsvAnnotator(BaseAnnotator):
    """
    Annotator class for models that stores the output to a csv file.

    Arguments:
        model_name (str): Name of the model.
        model_lib (str): Name of the model library.
        model_args (dict): Arguments for the model.
        out_path (str): Path to the output file.

    """

    def __init__(
        self,
        model_name: str,
        model_lib: str,
        model_args: Optional[dict] = None,
        out_path: str = "",
        **kwargs,
    ):
        """
        Arguments:
            model_name (str): Name of the model.
            model_lib (str): Name of the model library.
            model_args (dict): Arguments for the model.
            out_path (str): Path to the output file.
            kwargs: a dict containing additional arguments
        """
        if model_args is None:
            self.model_args = {}
        else:
            self.model_args = model_args

        self.model_name = model_name
        self.out_path = out_path
        self.to_kwargs = False
        self.kwargs = kwargs

        # store input as a dataframe
        self._input: Optional[pd.DataFrame] = None
        self._output_handler: Optional[CsvOutput] = CsvOutput(out_path)

        self._prompt: Optional[Prompt] = None
        self.in_col: Optional[str] = None
        self.model: Optional[Model] = None  # lazy loaded with first annotation
        self.model_lib = model_lib

    def set_data(
        self,
        data: Union[pd.DataFrame, str],
        in_col: str = "input",
        to_kwargs: bool = False,
    ):
        """
        Sets the input data for the annotator.

        Args:
            data: Union[pd.DataFrame, str] representing the input data.
            in_col: str representing the name of the input column.
            to_kwargs: bool representing whether to add the other rows
                       to the kwargs.
        """
        if self._input is not None:
            LOGGER.info("Input data is already set. Will be overwritten.")

        self.in_col = in_col
        self.to_kwargs = to_kwargs

        if isinstance(data, pd.DataFrame):
            self._input = data
        elif isinstance(data, str):
            self._input = CsvInput(data).read()
        else:
            raise ValueError(
                "Invalid input type! "
                "Only Dataframe or CSV file path is supported.",
            )

    def set_prompt(self, prompt: Union[Prompt, str]):
        if self._prompt is not None:
            LOGGER.info("Prompt is already set. Will be overwritten.")

        if isinstance(prompt, Prompt):
            self._prompt = prompt
        elif isinstance(prompt, str):
            self._prompt = Prompt(content=prompt)
        else:
            raise ValueError(
                "Invalid input type! " "Only Prompt or str is supported.",
            )

    def _load_model(self):
        """
        Loads the model to the annotator. If a model is already loaded,
        just return it.

        This is a method that should be overridden by child classes.

        Raises:
            ValueError: If the model library is not supported.
        """
        # if model is already loaded, just return it
        if self.model is not None:
            return self.model

        # if is subclass use dedicated load method
        if (
            issubclass(self.__class__, CsvAnnotator)
            and self.__class__ is not CsvAnnotator
        ):
            super(self.__class__, self)._load_model()

        # TODO impl general load method

        else:
            # TODO own exception
            raise ValueError(
                f"Model library {self.model_lib} is not supported!",
            )

        return self.model

    def annotate(
        self,
        data: Union[pd.DataFrame, str] = None,
        return_df: bool = False,
        **kwargs,
    ):
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
                in_col=kwargs.get("input_col", "input"),
                to_kwargs=kwargs.get("to_kwargs", False),
            )

        if self._prompt is None:
            # TODO: add a default prompt(s)
            self.set_prompt(prompt=kwargs.get("prompt", None))

        if self.model is None:
            self._load_model()

        self._annotate(**kwargs)

    def _annotate(self, **kwargs):
        """
        Annotates the input data and writes the annotated data to the
        output CSV file.

        Assumes that data and prompt is set.

        Args:
            kwargs: a dict containing the input variables for templates
        """
        output_data = []

        try:
            # TODO chunk the iteration
            for idx, row in self._input.iterrows():
                entry = self._annotate_batch(row, **kwargs)
                if entry:
                    output_data.append(entry)

        except Exception as read_error:
            # Handle the input reading error
            LOGGER.error(f"Input reading error: {str(read_error)}")

        # Write the annotated data to the output CSV file
        try:
            output_df = pd.DataFrame(output_data)
            self._output_handler.write(output_df)
        except Exception as write_error:
            LOGGER.error(f"Output writing error: {str(write_error)}")

    def _annotate_batch(self, row: pd.Series, **kwargs):
        """
        Annotates the input CSV file and writes the annotated data to the
        output CSV file.

        TODO the batch is only 1 row for now

        Args:
            kwargs: a dict containing the input variables for templates
        """

        if self.model is None or self._prompt is None:
            raise ValueError(
                "Model or prompt is not set! "
                "Please call set_data and set_prompt before annotate.",
            )

        text_prompt = row[str(self.in_col)]
        # extend kwargs with the current row
        kwargs[str(self.in_col)] = text_prompt

        # TODO add other rows to kwargs if needed
        try:
            messages = [self._prompt(**kwargs)]
            responses: ResponseList = self.model.predict(messages=messages)

            prediction: Response = responses[0]
            return {
                self.in_col: text_prompt,
                "label": prediction.answer,
                "raw_data": prediction.data,
            }
        except Exception as prediction_error:
            # TODO introduce a custom exception
            LOGGER.error(f"Prediction error: {str(prediction_error)}")
            return None


class OpenAiCsvAnnotator(CsvAnnotator):
    """
    Annotator class for OpenAI models that use CSV files as input and output.
    """

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gpt-3.5-turbo",
        temperature=0.0,
        model_args: Optional[dict] = None,
        out_path: str = "",
    ):
        """
        Arguments:
            model_name: str representing the Name of the OpenAI model.
            api_key: str representing the OpenAI API key.
            temperature: float value for Temperature for the model.
        """
        super().__init__(
            model_name=model_name,
            model_lib="openai",
            model_args=model_args,
            out_path=out_path,
        )
        self.api_key = api_key
        self.temperature = temperature

    def _load_model(self):
        self.model = OpenAiModel(
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        return self.model


class HuggingFaceCsvAnnotator(CsvAnnotator):
    """
    Annotator class for OpenAI models that use CSV files as input and output.
    """

    def __init__(
        self,
        model_name: str,
        out_path: str,
        model_args: Optional[dict] = None,
        token_args: Optional[dict] = None,
    ):
        """
        Arguments:
            model_name: str representing the Name of the OpenAI model.
            api_key: str representing the OpenAI API key.
            temperature: float value for Temperature for the model.
        """
        super().__init__(
            model_name=model_name,
            model_lib="huggingface",
            model_args=model_args,
            out_path=out_path,
        )
        if token_args is None:
            self.token_args = {}
        else:
            self.token_args = token_args

    def _load_model(self):
        self.model = HFAutoModelForCausalLM(
            model_name=self.model_name,
            model_args=self.model_args,
            token_args=self.token_args,
        )
        return self.model
