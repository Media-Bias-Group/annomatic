import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from annomatic.llm import ResponseList
from annomatic.llm.base import Model
from annomatic.llm.util import build_message

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

LOGGER = logging.getLogger(__name__)


class HuggingFaceModel(Model, ABC):
    """
    Base class for all HuggingFace models.
    """

    def __init__(
        self,
        model_name: str,
        model_args: Optional[Dict[str, Any]] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        use_chat_template: bool = False,
    ):
        super().__init__(model_name=model_name)

        self.model_args = model_args or {}
        self.tokenizer_args = tokenizer_args or {}
        self.generation_args = generation_args or {}
        self.system_prompt = system_prompt
        self.use_chat_template = use_chat_template

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            **self.tokenizer_args,
        )

        if self.use_chat_template and self.tokenizer.chat_template is None:
            LOGGER.warning(
                "Tokenizer doesn't have a chat template! Use concatenation "
                "instead.",
            )

        self.model: Optional[
            Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]
        ] = None

    @abstractmethod
    def _format_output(self, decoded_output, messages) -> List[str]:
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
                pad_token_id = self.tokenizer_args.get("pad_token_id", 0)
                LOGGER.warning(
                    f"Tokenizer doesn't have a pad_token!"
                    f"Using pad_token_id = {pad_token_id}",
                )
                self.tokenizer.pad_token_id = pad_token_id
        else:
            padding = False

        # add system prompt if needed
        messages = self._add_system_prompt(messages)

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

        if self.generation_args.get("max_length", 20) == 20:
            self.generation_args["max_length"] = input_length * 2

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

        self.model.eval()  # make outputs deterministic
        model_outputs = self.model.generate(
            **model_inputs,
            **self.generation_args,
        )
        return self.tokenizer.batch_decode(
            model_outputs,
            skip_special_tokens=True,
        )

    def _add_system_prompt(self, messages: List[str]) -> List[str]:
        """
        Validates the system prompt and adds it to the messages if needed.

        Args:
            messages: List of string messages to predict the response for.

        Returns:
            The messages with the system prompt added if needed.
        """
        if self.system_prompt is None:
            return messages
        else:
            if (
                self.use_chat_template
                and self.tokenizer.chat_template is not None
            ):
                system_prompt = build_message(self.system_prompt, "system")
                messages_with_system = [
                    self.tokenizer.apply_chat_template(
                        [system_prompt, build_message(message, "user")],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for message in messages
                ]
            else:
                messages_with_system = [
                    self.system_prompt + "\n\n" + message
                    for message in messages
                ]

        return messages_with_system


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
        model_args: Optional[Dict[str, Any]] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        use_chat_template: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            model_args=model_args,
            tokenizer_args=tokenizer_args,
            generation_args=generation_args,
            system_prompt=system_prompt,
            use_chat_template=use_chat_template,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **self.model_args,
        )

    def _format_output(self, decoded_output, messages) -> List[str]:
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
        model_args: Optional[Dict[str, Any]] = None,
        tokenizer_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        use_chat_template: bool = False,
    ):
        super().__init__(
            model_name=model_name,
            model_args=model_args,
            tokenizer_args=tokenizer_args,
            generation_args=generation_args,
            system_prompt=system_prompt,
            use_chat_template=use_chat_template,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            **self.model_args,
        )

    def _format_output(self, decoded_output, messages) -> List[str]:
        return decoded_output
