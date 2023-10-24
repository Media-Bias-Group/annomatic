import logging

from typing import Any, List, Optional

from annomatic.llm.base import Model, Response
from annomatic.llm.openai.utils import _build_response, build_message

LOGGER = logging.getLogger(__name__)

try:
    import openai
except ImportError as e:
    raise ValueError(
        'Install "poetry install --with openai" before using this model!',
        e,
    ) from None

SUPPORTED_MODEL = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct"]
COMPLETION_ONLY = ["gpt-3.5-turbo-instruct"]


def valid_model(model: str) -> str:
    if model not in SUPPORTED_MODEL:
        LOGGER.warning(
            "Given Model not found use default (gpt-3.5-turbo)",
        )
        return "gpt-3.5-turbo"

    return model


class OpenAiModel(Model):
    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-3.5-turbo",
        temperature=0.0,
    ):
        """
        Initialize the OpenAI model.

        Arguments:
            api_key: The API key for accessing the OpenAI API.
            model: string representing the selected model.
                (Default="gpt-3.5-turbo")
            temperature: The temperature parameter for text generation.
                (Default=0.0)

        Raises:
            ValueError: If no API key is provided.
        """
        self._model = valid_model(model=model)
        self._temperature = temperature

        self.system_prompt: Optional[str] = None

        if api_key == "":
            raise ValueError("No OPEN AI key given!")

        openai.api_key = api_key

    def predict(self, content: Any) -> Response:
        """
        Predict output for the provided content.

        This method is dispatched based on the type of content
        and calls the appropriate prediction method.

        Arguments:
            content: The content for which predictions should be made.

        Returns:
            Response: Predicted output based on the provided content.

        Raises:
            NotImplementedError:
            If the content type is not supported (not str or List[str]).
        """
        if isinstance(content, str):
            return self._predict_single(content)
        elif isinstance(content, list) and all(
            isinstance(item, str) for item in content
        ):
            return self._predict_list(content)
        else:
            raise NotImplementedError(
                "unknown type! Needs to be str or List[str]!",
            )

    def _predict_list(self, content: List[str]) -> Response:
        """
        Predict response for a list of prompts.

        Arguments:
            content (List[str]): A list of prompts.

        Returns:
            Any: Predicted output based on the provided prompts.

        Raises:
            ValueError: If using a legacy model that does not support multiple
            messages.
        """
        if self._model in COMPLETION_ONLY:
            raise ValueError(
                "multiple messages for Legacy API not implemented!",
            )

        messages = self.build_messages(content)
        api_response = self._call_chat_completions_api(messages=messages)
        return _build_response(api_response)

    def _predict_single(self, content: str) -> Response:
        """
        Predict response for a single prompt.

        Arguments:
            content: string with the content of the prompt.

        Returns:
            Any: Predicted output based on the provided prompt.

        """
        if self._model in COMPLETION_ONLY:
            LOGGER.info("Legacy API used!")
            api_response = self._call_completion_api(prompt=content)
            return _build_response(api_response)

        messages = self.build_messages([content])
        api_response = self._call_chat_completions_api(messages=messages)
        return _build_response(api_response)

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

        return openai.Completion.create(
            model=self._model,
            prompt=prompt,
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
        return openai.ChatCompletion.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )

    def add_system_prompt(self, content: str):
        """
        Add a initial system prompt to the conversation.

        This method adds a system prompt to the conversation
        if the selected model supports it. Only models which supporting the new
        Chat Completion are allowed to have system prompts.

        If it is not possible to add a system prompt a warning message is
        printed and no system prompt is added.

        Arguments:
            content: string containing the system prompt content.
        """
        if self._model not in COMPLETION_ONLY:
            self.system_prompt = build_message(
                content=content,
                role="system",
            )
        else:
            LOGGER.warning(
                "The used model is a Legacy model. System prompt is NOT used!",
            )

    def build_messages(self, prompts: List[str]):
        """
        Build a list of messages.

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
