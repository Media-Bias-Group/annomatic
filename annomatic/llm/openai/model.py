import logging
from abc import ABC
from typing import Any, Dict, List, Optional

from annomatic.config.base import OpenAiConfig
from annomatic.llm.base import (
    Model,
    ModelLoader,
    ModelPredictionError,
    Response,
    ResponseList,
)
from annomatic.llm.openai.util import _build_response
from annomatic.llm.util import build_message

LOGGER = logging.getLogger(__name__)

try:
    import openai
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_fixed,
        wait_random_exponential,
    )
except ImportError as e:
    raise ValueError(
        'Install "poetry install --with openai" before using this model!',
        e,
    ) from None


def valid_model(model_name: str) -> str:
    if model_name not in OpenAiModel.SUPPORTED_MODEL:
        LOGGER.warning(
            "Given Model not found use default (gpt-3.5-turbo)",
        )
        return "gpt-3.5-turbo"

    return model_name


class OpenAiModel(Model):
    SUPPORTED_MODEL = [
        "gpt-4",
        "gpt-4-1106-preview",
        "gpt-3.5-turbo",
    ]
    COMPLETION_ONLY = ["gpt-3.5-turbo-instruct"]

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
        generation_args: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize the OpenAI model.

        Arguments:
            api_key: The API key for accessing the OpenAI API.
            model_name: string representing the selected model.
                (Default="gpt-3.5-turbo")
            generation_args: dict containing the generation arguments.
        Raises:
            ValueError: If no API key is provided.
        """
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
        )
        if model_name in self.COMPLETION_ONLY:
            LOGGER.info("Warning. Legacy API used!")
        valid_model(model_name=model_name)

        self.generation_args = generation_args or {}
        if api_key == "":
            raise ValueError("No OPEN AI key given!")

        openai.api_key = api_key

    def predict(self, messages: List[str]) -> ResponseList:
        """
        Predict output for the provided content.

        This method is dispatched based on the type of content
        and calls the appropriate prediction method.

        Arguments:
            messages: The content for which predictions should be made.

        Returns:
            Response: Predicted output based on the provided content.

        Raises:
            NotImplementedError:
            If the content type is not supported (not str or List[str]).
        """
        try:
            if isinstance(messages, str):
                messages = [messages]

            if isinstance(messages, list):
                return self._predict(messages=messages)

            else:
                raise NotImplementedError(
                    "unknown type! Needs to be str or List[str]!",
                )
        except Exception as exception:
            raise ModelPredictionError(
                f"OpenAI Model prediction Error: {exception}",
            )

    def _predict(self, messages: List[str]) -> ResponseList:
        """
        Predict response for a batch of prompts.

        Arguments:
            messages: A list of prompts.

        Returns:
            Any: Predicted output based on the provided prompts.

        """
        if len(messages) > 1:
            raise ValueError("Batch messages are not supported!")

        if self.model_name in self.COMPLETION_ONLY:
            api_response = self._call_completion_api(prompt=messages[0])
        else:
            messages = self.build_chat_messages(messages)
            api_response = self._call_chat_completions_api(
                messages=messages,
            )

        return ResponseList.from_responses(
            [_build_response(message=messages, api_response=api_response)],
        )

    def _call_completion_api(self, prompt: str):
        """
        Makes the function call to the Completion API like specified in
        https://platform.openai.com/docs/guides/gpt/completions-api

        This is a legacy API checkpoint. There is only used when the selected
        model don't support the chat_completion API.

        Args:
            prompt: string representation of the Prompt

        Returns:
            The Completion object produced by the OpenAI Model
        """
        try:
            return self._completion_api(prompt=prompt)
        except Exception as exception:
            raise exception

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    def _completion_api(self, prompt: str):
        return openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            **self.generation_args,
            request_timeout=200,
        )

    def _call_chat_completions_api(self, messages: List[str]):
        """
        Makes the function call to the Chat Completion API like specified in
        https://platform.openai.com/docs/guides/gpt/chat-completions-api


        Args:
            messages: List of string representation of the given conversation

        Returns:
            The Completion object produced by the OpenAI Model
        """
        try:
            return self._chat_completion_api(messages=messages)
        except Exception as exception:
            raise exception

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    def _chat_completion_api(self, messages: List[str]):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            **self.generation_args,
            request_timeout=200,
        )

    def build_chat_messages(self, prompts: List[str]):
        """
        Build a list of chat messages.

        This function creates the messages list in the format specified in the
        Chat Completions API provided by OpenAI.
        https://platform.openai.com/docs/guides/gpt/chat-completions-api

        The format is a list containing messages. The prompt given to this

        Arguments:
            prompts: content of the prompt, made by the user.

        Returns:
            list: A list of the messages in the conversation
        """
        messages = []
        if self.system_prompt is not None:
            messages.append(build_message(self.system_prompt, "system"))

        for prompt in prompts:
            messages.append(build_message(prompt, "user"))

        return messages


class OpenAiModelLoader(ModelLoader, ABC):
    """
    Model loader for OpenAI models.

    This class is responsible for loading OpenAI models.

    Attributes:
        model_name (str): The name of the model.
        config (OpenAiConfig): The configuration of the model.
        system_prompt (Optional[str]): The system prompt.
        lib_args (Optional[Dict[str, Any]]): The library arguments.
    """

    DEFAULT_BATCH_SIZE = 1

    def __init__(
        self,
        model_name: str,
        config: Optional[OpenAiConfig] = None,
        model_args: Optional[Dict[str, Any]] = None,
        generation_args: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        batch_size: Optional[int] = DEFAULT_BATCH_SIZE,
        api_key: str = "",
        labels: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            config=config or OpenAiConfig(),
            system_prompt=system_prompt,
            lib_args={"api_key": api_key},
            batch_size=batch_size,
            labels=labels,
            **kwargs,
        )

        if hasattr(self.config, "model_args"):
            self.config.model_args = getattr(self.config, "model_args", {})
            self.config.model_args.update(model_args or {})

        self.update_config_generation_args(generation_args)

    def _load_model(self) -> Model:
        from annomatic.llm.openai import OpenAiModel

        if not isinstance(self.config, OpenAiConfig):
            raise ValueError(
                "OpenAI models require a OpenAiConfig object.",
            )

        api_key = self.lib_args.get("api_key", None)
        return OpenAiModel(
            model_name=self.model_name,
            api_key=api_key,
            system_prompt=self.system_prompt,
            generation_args=self.config.to_dict(),
        )
