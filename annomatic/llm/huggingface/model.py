from abc import ABC, abstractmethod
from typing import List, Optional, Union

from annomatic.llm import ResponseList
from annomatic.llm.base import Model

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )
except ImportError as e:
    raise ValueError(
        'Install "poetry install --with huggingface" before using this model!'
        "Alongside make sure that torch is installed. If not run"
        '"pip install torch"',
        e,
    ) from None


class HuggingFaceModel(Model, ABC):
    """
    Base class for all HuggingFace models.
    """

    def __init__(self, model_name: str, token_args=None):
        super().__init__(model_name=model_name)
        if token_args is None:
            token_args = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **token_args,
        )
        self.model: Optional[
            Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]
        ] = None

    @abstractmethod
    def _format_output(self, decoded_output, messages):
        pass

    def predict(self, messages: List[str]) -> ResponseList:
        """
        Predicts the response for the given list of messages.

        Args:
            messages: List of string messages to predict the response for.

        Returns:
            The predicted responses as a ResponseList.
        """
        if self.model is None:
            raise ValueError(
                "Model is not initialized!",
            )

        if isinstance(messages, str):
            messages = [messages]

        if isinstance(messages, List) and len(messages) > 1:
            padding = True
            if not self.tokenizer.pad_token:
                print(
                    "Tokenizer doesn't have a pad_token! Use pad_token_id = 0",
                )
                self.tokenizer.pad_token_id = 0
        else:
            padding = False

        return self._predict(messages=messages, padding=padding)

    def _predict(self, messages, padding):
        model_inputs = self.tokenizer(
            messages,
            padding=padding,
            return_tensors="pt",
            return_length=True,
        )

        # Pop length from model_inputs, otherwise set length to default (20)
        input_length = int(model_inputs.pop("length", 20).max())
        if self.model is not None and self.model.device.type == "cuda":
            model_inputs = model_inputs.to("cuda")

        decoded_output = self._call_llm_and_decode(
            model_inputs,
            2 * input_length,
        )

        # remove the input from any response (if needed)
        responses = self._format_output(decoded_output, messages)
        return ResponseList(
            answers=responses,
            data=decoded_output,
            queries=messages,
        )

    def _call_llm_and_decode(
        self,
        model_inputs,
        output_length: int,
    ) -> List[str]:
        """
        makes the library call for the LLM prediction.
        """
        if self.model is None:
            raise ValueError("Model is not initialized!")

        self.model.eval()  # make outputs deterministic
        model_outputs = self.model.generate(
            **model_inputs,
            max_length=output_length,
        )
        return self.tokenizer.batch_decode(
            model_outputs,
            skip_special_tokens=True,
        )


class HFAutoModelForCausalLM(HuggingFaceModel):
    """
    A model that uses the AutoModelForCausalLM class from the HuggingFace
    transformers library.

    This model uses models that are available for the
    AutoModelForCausalLM class.

    See
    https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#tfautomodelforcausallm
    for details.
    """

    def __init__(
        self,
        model_name: str,
        model_args=None,
        token_args=None,
    ):
        super().__init__(
            model_name=model_name,
            token_args=token_args,
        )
        if model_args is None:
            model_args = {}

        self.model: AutoModelForCausalLM = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_args,
            )
        )

    def _format_output(self, decoded_output, messages):
        return [
            response[len(prefix) :].strip()
            for response, prefix in zip(decoded_output, messages)
        ]


class HFAutoModelForSeq2SeqLM(HuggingFaceModel):
    """
    A model that uses the AutoModelForSeq2SeqLM class from the HuggingFace
    transformers library.

    This model uses models that are available for the
    AutoModelForSeq2SeqLM class.

    See
    https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#tfautomodelforseq2seqlm
    for details.
    """

    def __init__(
        self,
        model_name: str,
        model_args=None,
        token_args=None,
    ):
        super().__init__(
            model_name=model_name,
            token_args=token_args,
        )
        if model_args is None:
            model_args = {}

        self.model: AutoModelForSeq2SeqLM = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                **model_args,
            )
        )

    def _format_output(self, decoded_output, messages):
        return decoded_output
