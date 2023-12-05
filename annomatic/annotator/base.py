import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from annomatic.annotator import util
from annomatic.config.base import (
    HuggingFaceConfig,
    ModelConfig,
    OpenAiConfig,
    VllmConfig,
)
from annomatic.llm.base import Model, ResponseList
from annomatic.prompt import Prompt
from annomatic.retriever.base import Retriever

LOGGER = logging.getLogger(__name__)


class ModelLoadMixin(ABC):
    """
    Mixin for annotator to load a model from different libraries.
    """

    def _load_model(
        self,
        model_name: str,
        model_lib: str,
        config: ModelConfig,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> Model:
        """
        Loads the model and store it in self.model.

        Returns:
            The loaded model.
        """

        if model_lib == "openai":
            from annomatic.llm.openai import OpenAiModel

            if not isinstance(config, OpenAiConfig):
                raise ValueError(
                    "OpenAI models require a OpenAiConfig object.",
                )

            api_key = kwargs.get("api_key", None)
            return OpenAiModel(
                model_name=model_name,
                api_key=api_key,
                system_prompt=system_prompt,
                generation_args=config.to_dict(),
            )

        elif model_lib == "huggingface":
            if not isinstance(config, HuggingFaceConfig):
                raise ValueError(
                    "Huggingface models require a HuggingfaceConfig object.",
                )

            model_args = config.model_args
            tokenizer_args = config.tokenizer_args
            generation_args = config.to_dict()
            auto_model = kwargs.get("auto_model", "AutoModelForCausalLM")
            use_chat_template = kwargs.get("use_chat_template", False)

            if auto_model == "AutoModelForCausalLM":
                from annomatic.llm.huggingface import HFAutoModelForCausalLM

                return HFAutoModelForCausalLM(
                    model_name=model_name,
                    model_args=model_args,
                    tokenizer_args=tokenizer_args,
                    generation_args=generation_args,
                    system_prompt=system_prompt,
                    use_chat_template=use_chat_template,
                )
            elif auto_model == "AutoModelForSeq2SeqLM":
                from annomatic.llm.huggingface import HFAutoModelForSeq2SeqLM

                return HFAutoModelForSeq2SeqLM(
                    model_name=model_name,
                    model_args=model_args,
                    tokenizer_args=tokenizer_args,
                    generation_args=generation_args,
                    system_prompt=system_prompt,
                    use_chat_template=use_chat_template,
                )
            else:
                raise ValueError(
                    "auto_model must be either "
                    "'AutoModelForCausalLM' or 'AutoModelForSeq2SeqLM')",
                )
        elif model_lib == "vllm":
            from annomatic.llm.vllm import VllmModel

            if not isinstance(config, VllmConfig):
                raise ValueError(
                    "VLLM models require a VllmConfig object.",
                )
            return VllmModel(
                model_name=model_name,
                model_args=config.model_args,
                generation_args=config.to_dict(),
                system_prompt=system_prompt,
            )
        else:
            raise ValueError(
                f"Model library {model_lib} not supported."
                f" Please choose from 'openai', 'huggingface', or 'vllm'.",
            )


class FewShotMixin(ABC):
    """
    Mixin for annotator to load a few-shot examples.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context: Optional[pd.DataFrame] = None
        self.icl_prompt: Optional[Prompt] = None

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
            icl_prompt: a specific prompt used for the examples. If no
                additional prompt is set, the regular prompt is used and the
                examples are added at the end.
        """
        self.context = context
        self.icl_prompt = prompt

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
        if self.icl_prompt is None:
            if hasattr(self, "_prompt"):
                self.icl_prompt = self._prompt
            else:
                raise ValueError("Prompt is not set!")

        label_var = self.icl_prompt.get_label_variable()
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
            prompt = self.icl_prompt(**row_dict)

            if pred_label is not None:
                prompt += f"{pred_label}\n\n"
            message += prompt

        return message


class BaseAnnotator(FewShotMixin, ModelLoadMixin, ABC):
    """
    Base class for annotator classes
    """

    def __init__(
        self,
        model_name: str,
        model_lib: str,
        config: ModelConfig,
        batch_size: Optional[int] = None,
        labels: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        lib_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_lib = model_lib
        self.config = config
        self.batch_size = batch_size
        self._labels = labels
        self.system_prompt = system_prompt
        self.lib_args = lib_args or {}
        self.kwargs = kwargs

        self.data: Optional[pd.DataFrame] = None
        self.data_variable: Optional[str] = None

        self._prompt: Optional[Prompt] = None

        self._model = self._load_model(
            model_name=self.model_name,
            model_lib=self.model_lib,
            config=self.config,
            system_prompt=self.system_prompt,
            **self.lib_args,
        )

    @abstractmethod
    def annotate(
        self,
        data: Optional[Any] = None,
        return_df: bool = False,
        **kwargs,
    ):
        """
        Annotates the input data and stores the annotated data.

        Args:
            data: the input data
            return_df: bool indicating if the annotated data should be returned
            kwargs: a dict containing the input variables for prompt templates
        """
        raise NotImplementedError()

    @abstractmethod
    def get_num_samples(self):
        """
        Returns the number of data instances to be annotated.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_data(
        self,
        data: Any,
        data_variable: str,
    ):
        """
        Sets the data to be annotated.

        Args:
            data: the input data
            data_variable: the variable name of the input data
        """
        raise NotImplementedError()

    def fill_prompt(self, batch: pd.DataFrame, **kwargs) -> List[str]:
        """
        Creates the prompt passed to the model.

        Args:
            batch: pd.DataFrame representing the input data.
            kwargs: a dict containing the input variables for templates(
        """
        if self._prompt is None:
            raise ValueError("Prompt is not set!")

        label_var = self._prompt.get_label_variable()

        if (
            label_var is not None
            and kwargs.get(
                label_var,
            )
            is None
            and self._labels is not None
        ):
            kwargs[label_var] = self._labels

        messages: List[str] = []
        for index, row in batch.iterrows():
            if self.context is not None:
                icl_part = self.create_context_part(
                    query=row[str(self.data_variable)],
                    **kwargs,
                )
            else:
                icl_part = ""

            kwargs[str(self.data_variable)] = row[str(self.data_variable)]
            messages.append(icl_part + self._prompt(**kwargs))

        return messages

    @abstractmethod
    def store_annotated_data(self, output_data: pd.DataFrame):
        """
        Stores the annotated data in a csv file.

        Args:
            output_data: a list of dicts containing the annotated data

        """
        raise NotImplementedError()

    def _validate_data_variable(self) -> bool:
        """
        Validates the data variable.

        If a prompt is set, the data variable is valid if it occurs in the
        prompt. Otherwise, the data variable is valid if it is not None.


        Returns:
            bool: True if the data variable is valid, False otherwise.
        """
        if self._prompt is None or self.data_variable is None:
            # no validation possible
            return True

        return self.data_variable in self._prompt.get_variables()

    def _model_predict(self, messages: List[str]) -> ResponseList:
        """
        Wrapper of the model predict method.

        Args:
            messages: List[str] representing the input messages.

        Returns:
            ResponseList: an object containing the Responses.
        """
        if self._model is None:
            raise ValueError("Model is not initialized!")

        return self._model.predict(messages=messages)

    def set_prompt(self, prompt: Union[Prompt, str]):
        if self._prompt is not None:
            LOGGER.info("Prompt is already set. Will be overwritten.")

        if isinstance(prompt, Prompt):
            self._prompt = prompt

            if not self._validate_data_variable():
                raise ValueError("Input column does not occur in prompt!")

        elif isinstance(prompt, str):
            self._prompt = Prompt(content=prompt)
            if not self._validate_data_variable():
                raise ValueError("Input column does not occur in prompt!")
        else:
            raise ValueError(
                "Invalid input type! " "Only Prompt or str is supported.",
            )

    def _validate_labels(self, **kwargs):
        if self._labels is None:
            prompt_labels = self._prompt.get_label_variable()
            labels_from_kwargs = kwargs.get(prompt_labels, None)

            if labels_from_kwargs is not None:
                self._labels = labels_from_kwargs
        else:
            prompt_labels = self._prompt.get_label_variable()
            labels_from_kwargs = kwargs.get(prompt_labels)

            if labels_from_kwargs is not None and set(self._labels) != set(
                labels_from_kwargs,
            ):
                raise ValueError(
                    "Labels in prompt and Annotator do not match!",
                )

    def update_config_generation_args(
        self,
        generation_args: Optional[Dict[str, Any]] = None,
    ):
        for key, value in (generation_args or {}).items():
            if (
                hasattr(self.config, key)
                and getattr(self.config, key) != value
            ):
                setattr(self.config, key, value)
            else:
                self.config.kwargs[key] = value

    def _num_batches(self, total_rows: int):
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

    def _soft_parse(
        self,
        df: pd.DataFrame,
        in_col: str,
        parsed_col: str,
    ) -> pd.DataFrame:
        if self._labels is None:
            raise ValueError("Labels are not set!")

        df[parsed_col] = df[in_col].apply(
            lambda x: util.find_label(x, self._labels),
        )

        return df

    def _annotate(
        self,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Annotates the input data and returns it as a DataFrame.
        Assumes that data and prompt is set.

        Args:
            kwargs: a dict containing the input variables for templates
        """

        if self.data is None:
            raise ValueError("Data is not set!")

        output_data = []
        try:
            total_rows = self.get_num_samples()
            num_batches = self._num_batches(total_rows)

            LOGGER.info(f"Starting Annotation of {total_rows}")
            for idx in tqdm(range(num_batches)):
                batch = self.data.iloc[
                    idx * self.batch_size : (idx + 1) * self.batch_size
                ]
                entries = self._annotate_batch(batch, **kwargs)
                if entries:
                    output_data.extend(entries)

            # handle rest of the data
            if num_batches * self.batch_size < total_rows:
                batch = self.data.iloc[num_batches * self.batch_size :]
                entries = self._annotate_batch(batch, **kwargs)
                if entries:
                    output_data.extend(entries)

        except Exception as read_error:
            # Handle the input reading error
            LOGGER.error(f"Input reading error: {str(read_error)}")

        LOGGER.info("Annotation done!")
        LOGGER.info(f"Successfully annotated {len(output_data)} rows.")

        try:
            output_df = pd.DataFrame(output_data)
        except Exception as df_error:
            LOGGER.error(f"Output dataframe error: {str(df_error)}")
            return None

        try:
            # if labels are known perform soft parsing
            if self._labels:
                self._soft_parse(
                    df=output_df,
                    in_col="response",
                    parsed_col="label",
                )
            self.store_annotated_data(output_df)
            return output_df

        except Exception as write_error:
            LOGGER.error(f"Output writing error: {str(write_error)}")
            return output_df

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

        if self._model is None or self._prompt is None:
            raise ValueError(
                "Model or prompt is not set! "
                "Please call set_data and set_prompt before annotate.",
            )

        try:
            messages = self.fill_prompt(batch=batch, **kwargs)
            responses = self._model_predict(messages)

            annotated_data = []
            for idx, response in enumerate(responses):
                annotated_data.append(
                    {
                        self.data_variable: batch.iloc[idx][
                            str(self.data_variable)
                        ],
                        "response": response.answer,
                        "raw_data": response.data,
                        "query": response.query,
                    },
                )
            return annotated_data

        except Exception as exception:
            LOGGER.error(f"Prediction error: {str(exception)}")
            return []
