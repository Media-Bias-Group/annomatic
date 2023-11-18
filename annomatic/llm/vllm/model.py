import logging
from typing import Any, Dict, List, Optional

from annomatic.llm import ResponseList
from annomatic.llm.base import Model

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
    ):
        super().__init__(model_name=model_name)

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
