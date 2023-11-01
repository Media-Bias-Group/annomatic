import logging
from enum import Enum
from typing import List, Optional, Union

import pandas as pd

from annomatic.annotator.base import BaseAnnotator
from annomatic.io import CsvInput, CsvOutput
from annomatic.llm.base import Model, ResponseList
from annomatic.llm.huggingface import HFAutoModelForCausalLM
from annomatic.llm.huggingface.model import HFAutoModelForSeq2SeqLM
from annomatic.llm.openai import OpenAiModel
from annomatic.llm.vllm import VllmModel
from annomatic.prompt import Prompt

LOGGER = logging.getLogger(__name__)


class CsvAnnotator(BaseAnnotator):
    """
    Base annotator class for models that stores the output to a csv file.

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
        batch_size: Optional[int] = None,
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
        self.batch_size = batch_size

        # store input as a dataframe
        self._input: Optional[pd.DataFrame] = None
        self._output_handler: Optional[CsvOutput] = CsvOutput(out_path)

        self._prompt: Optional[Prompt] = None
        self.in_col: Optional[str] = None
        self.model: Optional[Model] = None  # lazy loaded with first annotation
        self.model_lib = model_lib

    def _validate_input_column(self) -> bool:
        if self._prompt is None or self.in_col is None:
            # no validation possible
            return True

        return self.in_col in self._prompt.get_variables()

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

        if not self._validate_input_column():
            raise ValueError("Input column does not occur in prompt!")

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

            if not self._validate_input_column():
                raise ValueError("Input column does not occur in prompt!")

        elif isinstance(prompt, str):
            self._prompt = Prompt(content=prompt)
            if not self._validate_input_column():
                raise ValueError("Input column does not occur in prompt!")
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
                in_col=kwargs.get("in_col", "input"),
                to_kwargs=kwargs.get("to_kwargs", False),
            )

        if self._prompt is None:
            # TODO: add a default prompt(s)
            self.set_prompt(prompt=kwargs.get("prompt", None))

        if self.model is None:
            self._load_model()

        # TODO: add return_df if True return the annotated data as a DataFrame
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
        total_rows = self._input.shape[0]
        LOGGER.info(f"Starting Annotation of {total_rows}")

        try:
            num_batch = total_rows // self.batch_size
            for idx in range(num_batch):
                batch = self._input.iloc[
                    idx * self.batch_size : (idx + 1) * self.batch_size
                ]
                entries = self._annotate_batch(batch, **kwargs)
                if entries:
                    output_data.extend(entries)

                LOGGER.info(
                    f"Annotated... {(idx + 1) * self.batch_size} "
                    f"out of {self._input.shape[0]}",
                )

            # handle rest of the data
            if num_batch * self.batch_size < total_rows:
                batch = self._input.iloc[num_batch * self.batch_size :]
                entries = self._annotate_batch(batch, **kwargs)
                if entries:
                    output_data.extend(entries)

        except Exception as read_error:
            # Handle the input reading error
            LOGGER.error(f"Input reading error: {str(read_error)}")

        LOGGER.info("Annotation done!")
        LOGGER.info(f"Successfully annotated {len(output_data)} rows.")

        # Write the annotated data to the output CSV file
        try:
            output_df = pd.DataFrame(output_data)
            self._output_handler.write(output_df)
        except Exception as write_error:
            LOGGER.error(f"Output writing error: {str(write_error)}")

    def _annotate_batch(self, batch: pd.DataFrame, **kwargs) -> List[dict]:
        """
        Annotates the input CSV file and writes the annotated data to the
        output CSV file.

        Args:
            batch: pd.DataFrame representing the input data.
            kwargs: a dict containing the input variables for templates

        Returns:
            List[dict]: a list of dicts containing the annotated data
        """

        if self.model is None or self._prompt is None:
            raise ValueError(
                "Model or prompt is not set! "
                "Please call set_data and set_prompt before annotate.",
            )

        try:
            messages = []
            for index, row in batch.iterrows():
                kwargs[str(self.in_col)] = row[str(self.in_col)]
                messages.append(self._prompt(**kwargs))

            responses = self._model_predict(messages)

            annotated_data = []
            for idx, response in enumerate(responses):
                annotated_data.append(
                    {
                        self.in_col: batch.iloc[idx][str(self.in_col)],
                        "label": response.answer,
                        "raw_data": response.data,
                        "query": response.query,
                    },
                )
            return annotated_data

        except Exception as exception:
            # TODO introduce a custom exception
            LOGGER.error(f"Prediction error: {str(exception)}")

            return []

    def _model_predict(self, messages: List[str]) -> ResponseList:
        """
        Wrapper of the model predict method.

        Args:
            messages: List[str] representing the input messages.

        Returns:
            ResponseList: an object containing the Responses.
        """
        if self.model is None:
            raise ValueError("Model is not initialized!")

        return self.model.predict(messages=messages)

    def _num_chunks(self, total_rows: int):
        """
        Calculates the number of batches.

        If self.batch_size is not set, the whole dataset is used as a batch.

        Args:
            total_rows: int representing the total number of rows.
        """
        if self.batch_size:
            return total_rows // self.batch_size
        else:
            # if no batch size is set, use the whole dataset as a batch
            self.batch_size = total_rows
            return 1


class OpenAiCsvAnnotator(CsvAnnotator):
    """
    Annotator class for OpenAI models that use CSV files as input and output.
    """

    DEFAULT_BATCH_SIZE = 1

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gpt-3.5-turbo",
        temperature=0.0,
        model_args: Optional[dict] = None,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
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
            batch_size=batch_size,
        )
        self.api_key = api_key
        self.temperature = temperature
        self.batch_size = OpenAiCsvAnnotator.DEFAULT_BATCH_SIZE

    def _load_model(self):
        self.model = OpenAiModel(
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=self.temperature,
        )
        return self.model


class HFAutoModels(Enum):
    AutoModelForCausalLM = "AutoModelForCausalLM"
    AutoModelForSeq2SeqLM = "AutoModelForSeq2SeqLM"


class HuggingFaceCsvAnnotator(CsvAnnotator):
    """
    Annotator class for HuggingFace models that use CSV files as output.

    This class can use LLMs loaded by the AutoModelForCausalLM and
    AutoModelForSeq2SeqLM classes.
    """

    DEFAULT_BATCH_SIZE = 5

    def __init__(
        self,
        model_name: str,
        out_path: str,
        model_args: Optional[dict] = None,
        token_args: Optional[dict] = None,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        auto_model: str = "AutoModelForCausalLM",
    ):
        """
        Arguments:
            model_name: str representing the Name of the HuggingFace model
            auto_model: str representing the AutoModel class
            out_path: str representing the path to the output file
            model_args: dict representing the model arguments
            token_args: dict representing the token arguments
        """
        super().__init__(
            model_name=model_name,
            model_lib="huggingface",
            model_args=model_args,
            out_path=out_path,
            batch_size=batch_size,
        )
        if token_args is None:
            self.token_args = {}
        else:
            self.token_args = token_args

        self.auto_model = auto_model

    def _load_model(self):
        if self.auto_model == "AutoModelForCausalLM":
            self.model = HFAutoModelForCausalLM(
                model_name=self.model_name,
                model_args=self.model_args,
                token_args=self.token_args,
            )
            return self.model

        elif self.auto_model == "AutoModelForSeq2SeqLM":
            self.model = HFAutoModelForSeq2SeqLM(
                model_name=self.model_name,
                model_args=self.model_args,
                token_args=self.token_args,
            )
            return self.model


class VllmCsvAnnotator(CsvAnnotator):
    """
    Annotator class for Vllm models that use CSV files as input and output.

    This class can use LLMs loaded by the VllmModel class.
    """

    def __init__(
        self,
        model_name: str,
        out_path: str,
        model_args: Optional[dict] = None,
        token_args: Optional[dict] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Arguments:
            model_name: str representing the Name of the OpenAI model.
        """
        super().__init__(
            model_name=model_name,
            model_lib="vllm",
            model_args=model_args,
            out_path=out_path,
            batch_size=batch_size,
        )
        if token_args is None:
            self.token_args = {}
        else:
            self.token_args = token_args

    def _load_model(self):
        self.model = VllmModel(
            model_name=self.model_name,
            model_args=self.model_args,
            param_args=self.token_args,
        )
        return self.model
