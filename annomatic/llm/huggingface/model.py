from abc import ABC, abstractmethod
from typing import List, Optional, Union

from annomatic.llm import ResponseList
from annomatic.llm.base import Model
from annomatic.llm.huggingface.config import HuggingFaceConfig

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

    def __init__(
        self,
        model_name: str,
        config: HuggingFaceConfig = HuggingFaceConfig(),
    ):
        super().__init__(model_name=model_name)
        self.config = config or HuggingFaceConfig()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **self.config.to_dict(),
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

    def _predict(self, messages: List[str], **kwargs) -> ResponseList:
        model_inputs = self.tokenizer(
            messages,
            return_tensors="pt",
            return_length=True,
            **kwargs,
        )

        # Pop length from model_inputs, otherwise set length to default (20)
        input_length = int(model_inputs.pop("length", 20).max())
        if self.model is not None and self.model.device.type == "cuda":
            model_inputs = model_inputs.to("cuda")

        if self.config.to_dict().get("max_length", 20) == 20:
            self.config.kwargs["max_length"] = input_length * 2

        decoded_output = self._call_llm_and_decode(
            model_inputs,
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
    ) -> List[str]:
        """
        makes the library call for the LLM prediction.
        """
        if self.model is None:
            raise ValueError("Model is not initialized!")

        model_outputs = self.model.generate(
            **model_inputs,
            **self.config.to_dict(),
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
        config: HuggingFaceConfig = HuggingFaceConfig(),
    ):
        super().__init__(
            model_name=model_name,
            config=config,
        )

        self.model: AutoModelForCausalLM = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                **self.config.to_dict(),
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
        config: HuggingFaceConfig = HuggingFaceConfig(),
    ):
        super().__init__(
            model_name=model_name,
            config=config,
        )

        self.model: AutoModelForSeq2SeqLM = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                **self.config.to_dict(),
            )
        )

    def _format_output(self, decoded_output, messages):
        return decoded_output
