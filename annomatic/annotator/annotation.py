import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

from annomatic.prompt import Prompt
from annomatic.retriever.base import Retriever

LOGGER = logging.getLogger(__name__)


class AnnotationProcess(ABC):
    """
    Abstract class for the annotation process.
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
    ):
        self.labels = labels

    @abstractmethod
    def annotate(
        self,
        model,
        prompt: Prompt,
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
    def set_context(
        self,
        context: Union[Retriever, pd.DataFrame],
        prompt: Optional[Prompt] = None,
    ) -> None:
        """
        Sets the context for context-based annotations.

        Args:
            context: the context used for the annotation
            prompt: a specific prompt used for the examples. If no
                additional prompt is set, the regular prompt is used and the
                examples are added at the end.
        """
        raise NotImplementedError()


class DefaultAnnotation(AnnotationProcess):
    """
    Default annotation process.

       Args:
           batch_size: the batch size for the annotation process


    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
    ):
        super().__init__(labels=labels)
        self.context: Union[Retriever, pd.DataFrame, None] = None

    def set_context(
        self,
        context: Union[Retriever, pd.DataFrame],
        prompt: Optional[Prompt] = None,
    ) -> None:
        """
        Sets the context for the ICL prompt. The context can be either a
        Retriever or a pd.DataFrame.

        Args:
            context: the context for the ICL prompt
            prompt: a specific prompt handling the context. If no additional
                prompt is set, the regular prompt is used and the examples are
                added at the end.
        """
        self.context = context
        self.context_prompt = prompt

    def create_context_part(
        self,
        query: Optional[str],
        **kwargs,
    ) -> str:
        """
        Creates an ICL prompt. If the label is known, it is added to the
        prompt at the end.

        Args:
            query: the sentence to get the icl context for
            kwargs: a dict containing the input variables for templates

        Returns:
            str: the ICL prompt part.
        """

        # if no special icl prompt set use regular prompt
        if self.context_prompt is None:
            if hasattr(self, "_prompt"):
                self.context_prompt = self._prompt
            else:
                raise ValueError("Prompt is not set!")

        label_var = self.context_prompt.get_label_variable()
        if label_var is None:
            raise ValueError("Label variable not found in the ICL prompt.")

        if self.context is None or label_var is None:
            raise ValueError("Examples are not set!")

        pred_label = None
        message = ""

        if isinstance(self.context, Retriever):
            context = self.context.select(query=query)
        else:
            context = self.context

        for idx, row in context.iterrows():
            row_dict: Dict[str, Any] = row.to_dict()

            if label_var in row_dict:
                pred_label = row_dict[label_var]

            row_dict[label_var] = kwargs[label_var]
            prompt = self.context_prompt(**row_dict)

            if pred_label is not None:
                prompt += f"{pred_label}\n\n"
            else:
                prompt += "\n\n"

            message += prompt

        return message

    def _num_batches(
        self,
        total_rows: int,
        batch_size: int,
    ) -> Tuple[int, int]:
        """
        Calculates the number of batches and the batch size.

        If self.batch_size is not set, the whole dataset is used as a batch.

        Args:
            total_rows: int representing the total number of rows.
        """
        if not batch_size:
            return total_rows, 1

        if total_rows < batch_size:
            return 1, total_rows

        return total_rows // batch_size, batch_size

    def annotate(
        self,
        model,
        prompt: Prompt,
        data: pd.DataFrame,
        data_variable: str,
        batch_size: int,
        **kwargs,
    ) -> pd.DataFrame:
        if data is None:
            raise ValueError("Data is not set!")

        output_data = []

        total_rows = data.shape[0]
        num_batches, batch_size = self._num_batches(total_rows, batch_size)

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
            return None

    def _annotate_batch(
        self,
        model,
        prompt: Prompt,
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

        try:
            messages = self.fill_prompt(
                prompt=prompt,
                batch=batch,
                data_variable=data_variable,
                **kwargs,
            )
            print("messages:", messages)

            responses = model.run(messages)

            print("responses:", responses)

            return self.to_format(batch, messages, responses, data_variable)

        except Exception as exception:
            LOGGER.error(f"Prediction error: {str(exception)}")
            return []

    def to_format(
        self,
        batch: pd.DataFrame,
        messages: Union[List[str], str],
        responses: Dict,
        data_variable: str,
    ) -> list[Dict]:
        annotated_data = []

        for i in range((len(responses["replies"]))):
            response = responses["replies"][i]
            raw_data = responses.get("meta", responses["replies"][i])
            message = messages[i] if isinstance(messages, list) else messages

            parsed_response = {
                data_variable: batch.iloc[i][data_variable],
                "response": response,
                "raw_data": raw_data,
                "query": message,
            }
            print("parsed_response:", parsed_response)

            annotated_data.append(parsed_response)
        return annotated_data

    def fill_prompt(
        self,
        prompt: Prompt,
        batch: pd.DataFrame,
        data_variable: str,
        **kwargs,
    ) -> Union[List[str], str]:
        """
        Creates the prompt passed to the model.

        Args:
            prompt: Prompt representing the prompt used for annotation.
            batch: pd.DataFrame representing the input data.
            data_variable: str representing the variable in the input data.
            kwargs: a dict containing the input variables for templates(
        """
        if prompt is None:
            raise ValueError("Prompt is not set!")

        label_var = prompt.get_label_variable()
        if label_var is not None and self.labels is not None:
            kwargs[label_var] = self.labels

        messages: List[str] = []
        for index, row in batch.iterrows():
            if self.context is not None:
                icl_part = self.create_context_part(
                    query=row[str(data_variable)],
                    **kwargs,
                )
            else:
                icl_part = ""

            kwargs[str(data_variable)] = row[str(data_variable)]
            messages.append(icl_part + prompt(**kwargs))

        return messages[0] if len(messages) == 1 else messages
