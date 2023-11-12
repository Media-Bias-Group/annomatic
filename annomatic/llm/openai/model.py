import logging
from typing import Any, List, Optional

from annomatic.llm.base import (
    Model,
    ModelPredictionError,
    Response,
    ResponseList,
)
from annomatic.llm.openai.config import OpenAiConfig
from annomatic.llm.openai.utils import _build_response, build_message

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


def _handle_open_ai_exception(exception: Exception):
    if isinstance(exception, openai.error.APIError):
        LOGGER.warning(f"APIError: {exception}")
        pass
    elif isinstance(exception, openai.error.RateLimitError):
        LOGGER.warning(f"Rate Limit Error: {exception}")
        pass
    else:
        raise exception


class OpenAiModel(Model):
    SUPPORTED_MODEL = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct"]
    COMPLETION_ONLY = ["gpt-3.5-turbo-instruct"]

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gpt-3.5-turbo",
        config: OpenAiConfig = OpenAiConfig(),
    ):
        """
        Initialize the OpenAI model.

        Arguments:
            api_key: The API key for accessing the OpenAI API.
            model_name: string representing the selected model.
                (Default="gpt-3.5-turbo")

        Raises:
            ValueError: If no API key is provided.
        """
        super().__init__(model_name=model_name)
        self.config = config or OpenAiConfig()

        if model_name in self.COMPLETION_ONLY:
            LOGGER.info("Warning. Legacy API used!")

        self._model = valid_model(model_name=model_name)
        self.system_prompt: Optional[dict[str, str]] = None

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

        if self._model in self.COMPLETION_ONLY:
            api_response = self._call_completion_api(prompt=messages)
        else:
            messages = self.build_chat_messages(messages)
            api_response = self._call_chat_completions_api(
                messages=messages,
            )

        return ResponseList.from_responses(
            [_build_response(message=messages[0], api_response=api_response)],
        )

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(openai.error.RateLimitError),
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
            return openai.Completion.create(
                model=self._model,
                prompt=prompt,
                **self.config.to_dict(),
            )
        except Exception as exception:
            _handle_open_ai_exception(exception)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(openai.error.RateLimitError),
    )
    def _call_chat_completions_api(self, messages: List[str]):
        """
        Makes the function call to the Chat Completion API like specified in
        https://platform.openai.com/docs/guides/gpt/chat-completions-api


        Args:
            messages: List of string representation of the given Prompts

        Returns:
            The Completion object produced by the OpenAI Model
        """
        if len(messages) > 1:
            raise ValueError("Only one message is supported")

        try:
            return openai.ChatCompletion.create(
                model=self._model,
                messages=messages,
                **self.config.to_dict(),
            )
        except Exception as exception:
            _handle_open_ai_exception(exception)

    def add_system_prompt(self, content: str):
        """
        Add an initial system prompt to the conversation.

        This method adds a system prompt to the conversation
        if the selected model supports it. Only models which supporting the new
        Chat Completion are allowed to have system prompts.

        If it is not possible to add a system prompt a warning message is
        printed and no system prompt is added.

        Arguments:
            content: string containing the system prompt content.
        """
        if self._model not in self.COMPLETION_ONLY:
            self.system_prompt = build_message(
                content=content,
                role="system",
            )
        else:
            LOGGER.warning(
                "The used model is a Legacy model. System prompt is NOT used!",
            )

    def build_messages(self, prompts: List[str]) -> List[dict[Any, Any]]:
        """
        Build a list o encoded messages for the OpenAI Library.

        This function creates the messages list in the format specified in the
        Chat Completions API provided by OpenAI.
        https://platform.openai.com/docs/guides/gpt/chat-completions-api

        The format is a list containing messages. The prompt given to this

        Arguments:
            prompts: content of the prompt, made by the user.

        Returns:
            list: A list of the individual conversations messages
        """
        messages = []
        if self.system_prompt is not None:
            messages.append(self.system_prompt)

        for prompt in prompts:
            messages.append(build_message(prompt))

        return messages

    # Deprecation Warning

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
            messages.append(self.system_prompt)

        for prompt in prompts:
            messages.append(build_message(prompt))

        return messages

    def _predict_chat(self, messages: List[str]):
        """
        Predict response for a list of prompts.

        Arguments:
            messages (List[str]): A list of prompts.

        Returns:
            Any: Predicted output based on the provided prompts.

        Raises:
            ValueError: If using a legacy model that does not support multiple
            messages.
        """

        if len(messages) <= 1:
            raise ValueError("Only more than  is supported")

        if self._model in self.COMPLETION_ONLY:
            raise ValueError(
                "multiple messages for Legacy API not implemented!",
            )

        messages = self.build_chat_messages(messages)
        api_response = self._call_chat_completions_api(messages=messages)
        return _build_response(message=messages[0], api_response=api_response)
