import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from haystack.components.builders import PromptBuilder
from tqdm import tqdm

from annomatic.retriever.base import Retriever

LOGGER = logging.getLogger(__name__)


def _num_batches(
    total_rows: int,
    batch_size: int,
) -> Tuple[int, int]:
    """
    Calculates the number of batches and the batch size.

    If batch size is not set, default to batch size of 1.

    Args:
        total_rows: int representing the total number of rows
        batch_size: int representing a predefined batch size

    Returns:
        Tuple of number of batches and batch_size as int
    """
    if not batch_size:
        return total_rows, 1

    if total_rows < batch_size:
        return 1, total_rows

    return total_rows // batch_size, batch_size


def to_format(
    batch: pd.DataFrame,
    messages: Union[List[str], str],
    responses: Dict,
    data_variable: str,
) -> List[Dict]:
    annotated_data = []
    for i, response in enumerate(responses["replies"]):
        parsed_response = {
            data_variable: batch.iloc[i][data_variable],
            "response": response,
            "query": messages[i] if isinstance(messages, list) else messages,
        }

        # Add meta data if available
        if "meta" in responses:
            parsed_response["meta"] = responses["meta"]

        annotated_data.append(parsed_response)
    return annotated_data


class AnnotationProcess(ABC):
    """
    Abstract class for the annotation process.
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
    ):
        self.labels = labels
        self.context: Optional[dict] = None

    @abstractmethod
    def annotate(
        self,
        model,
        prompt: PromptBuilder,
        data: pd.DataFrame,
        data_variable: str,
        batch_size: int,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Annotates the input data and returns it as a DataFrame.

        Args:
            model: the model used for annotation
            prompt: the prompt used for annotation
            data: the input data
            data_variable: the variable in the input data
            batch_size: the batch size for the annotation process
            kwargs: a dict containing the input variables for templates
        """
        raise NotImplementedError()

    @abstractmethod
    def _annotate_batch(
        self,
        model,
        prompt: PromptBuilder,
        batch: pd.DataFrame,
        data_variable: str,
        **kwargs,
    ) -> List[dict]:
        raise NotImplementedError()

    def set_context(
        self,
        context: dict,
    ) -> None:
        """
        Sets the context for the ICL prompt. The context is
        a dict where the key represents the variable name in the prompt.

        Args:
            context: the context used for the annotation
        """
        self.context = context

    def build_context(self, query) -> Dict:
        if self.context is None:
            return {}

        return {
            k: v.select(query=query) if isinstance(v, Retriever) else v
            for k, v in self.context.items()
        }

    def fill_prompt(
        self,
        prompt: PromptBuilder,
        batch: pd.DataFrame,
    ) -> Union[List[str], str]:
        """
        Creates the prompt passed to the model.

        Args:
            prompt: Prompt representing the prompt used for annotation.
            batch: pd.DataFrame representing the input data.
        """
        if prompt is None:
            raise ValueError("Prompt is not set!")

        messages = [
            prompt.run(**row.to_dict(), **self.build_context(row))["prompt"]
            for _, row in batch.iterrows()
        ]

        return messages[0] if len(messages) == 1 else messages


class DefaultAnnotationProcess(AnnotationProcess):
    """
    Default annotation process.
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        intermediate_save: Optional[float] = None,
    ):
        super().__init__(labels=labels)
        self.context: Optional[dict] = None
        self.intermediate_save = intermediate_save

    def annotate(
        self,
        model,
        prompt: PromptBuilder,
        data: pd.DataFrame,
        data_variable: str,
        batch_size: int,
        **kwargs,
    ) -> pd.DataFrame:
        if data is None:
            raise ValueError("Data is not set!")

        output_data = []

        total_rows = data.shape[0]
        num_batches, batch_size = _num_batches(total_rows, batch_size)

        LOGGER.info(f"Starting Annotation of {total_rows}")
        for idx in tqdm(range(num_batches)):
            batch = data.iloc[idx * batch_size : (idx + 1) * batch_size]
            entries = self._annotate_batch(
                model=model,
                prompt=prompt,
                batch=batch,
                data_variable=data_variable,
                **kwargs,
            )
            if entries:
                output_data.extend(entries)

            if (
                self.intermediate_save
                and idx % int(self.intermediate_save * num_batches) == 0
            ):
                try:
                    temp = pd.DataFrame(output_data)
                    location = f"./temp_{len(entries)}.parquet"
                    temp.to_parquet(location, index=False)
                    LOGGER.info(f"Intermediate saved at {location}")
                except Exception as df_temp:
                    LOGGER.error(
                        f"intermediate save didn't work: " f"{str(df_temp)}",
                    )

        # handle rest of the data
        if num_batches * batch_size < total_rows:
            batch = data.iloc[num_batches * batch_size :]
            entries = self._annotate_batch(batch, **kwargs)
            if entries:
                output_data.extend(entries)

        LOGGER.info(f"Successfully annotated {len(output_data)} rows.")

        try:
            return pd.DataFrame(output_data)
        except Exception as df_error:
            LOGGER.error(f"Output dataframe error: {str(df_error)}")
            return pd.DataFrame()

    def _annotate_batch(
        self,
        model,
        prompt: PromptBuilder,
        batch: pd.DataFrame,
        data_variable: str,
        **kwargs,
    ) -> List[dict]:
        """
        Annotates a batch of data.

        This method is called by the annotate method to perform the annotation
        of a batch of data. It is not meant to be called directly.

        Args:
            batch: pd.DataFrame representing the input data.
            prompt: Prompt representing the prompt used for annotation.
            kwargs: a dict containing the input variables for templates

        Returns:
            List[dict]: a list of dicts containing the annotated data
        """

        if model is None or prompt is None:
            raise ValueError(
                "Model or prompt is not set! ",
            )

        messages = self.fill_prompt(
            prompt=prompt,
            batch=batch,
        )
        try:
            responses = model.run(messages)
            return to_format(batch, messages, responses, data_variable)

        except Exception as exception:
            LOGGER.error(f"Prediction error: {str(exception)}")
            return []
