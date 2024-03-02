import logging
from typing import Any, Dict, List, Optional

from annomatic.config.base import VllmConfig
from annomatic.llm import ResponseList
from annomatic.llm.base import Model, ModelLoader

LOGGER = logging.getLogger(__name__)

try:
    from vllm import LLM, RequestOutput, SamplingParams
except ImportError as e:
    raise ValueError(
        'Install "poetry install --with vllm" before using this model!',
        e,
    ) from None


class VllmModel(Model):
    """
    A model that uses the vLLM library.
    """

    def __init__(
        self,
        model_name: str,
        model_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
        )

        model_args = model_args or {}
        generation_args = generation_args or {}

        self.model = LLM(model_name, **model_args)
        self.samplingParams = SamplingParams(**generation_args)

    def predict(self, messages: List[str]) -> ResponseList:
        """
        Predicts the output of the model for the given messages. This method
        also contains validation logic.

        Arguments:
            messages: list of messages to predict

        Returns:
            ResponseList: list of responses
        """
        if self.model is None or self.samplingParams is None:
            raise ValueError("Model or SamplingParams not initialized.")

        # add system prompt if needed
        messages = self._add_system_prompt(messages=messages)

        return self._predict(messages=messages)

    def _predict(self, messages: List[str]) -> ResponseList:
        """
        Predicts the output of the model for the given messages.

        Arguments:
            messages: list of messages to predict

        Returns:
            ResponseList: list of responses
        """
        model_outputs = self._call_llm(messages=messages)
        answers = [output.outputs[0].text for output in model_outputs]
        query = [output.prompt for output in model_outputs]

        return ResponseList(answers=answers, data=model_outputs, queries=query)

    def _call_llm(self, messages: List[str]) -> list[RequestOutput]:
        """
        Wrapper for the vLLM model call.
        """
        try:
            return self.model.generate(messages, self.samplingParams)
        except Exception as exception:
            raise ValueError("Error while calling vLLM model.") from exception

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
            return [
                self.system_prompt + "\n\n" + message for message in messages
            ]


class VllmModelLoader(ModelLoader):
    """
    Abstract base class for Vllm annotators.

    Attributes:
        model_name (str): The name of the model.
        config (VllmConfig): The configuration of the model.
        system_prompt (Optional[str]): The system prompt.
        lib_args (Optional[Dict[str, Any]]): The library arguments.

    """

    DEFAULT_BATCH_SIZE = 5

    def __init__(
        self,
        model_name: str,
        config: Optional[VllmConfig] = None,
        model_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            config=config or VllmConfig(),
            system_prompt=system_prompt,
            lib_args={},
            batch_size=batch_size,
            labels=labels,
            **kwargs,
        )

        if hasattr(self.config, "model_args"):
            self.config.model_args = (
                getattr(
                    self.config,
                    "model_args",
                    {},
                )
                or {}
            )
            self.config.model_args.update(model_args or {})

        self.update_config_generation_args(generation_args)

    def load_model(self) -> Model:
        from annomatic.llm.vllm import VllmModel

        if not isinstance(self.config, VllmConfig):
            raise ValueError(
                "VLLM models require a VllmConfig object.",
            )
        return VllmModel(
            model_name=self.model_name,
            model_args=self.config.model_args,
            generation_args=self.config.to_dict(),
            system_prompt=self.system_prompt,
        )
