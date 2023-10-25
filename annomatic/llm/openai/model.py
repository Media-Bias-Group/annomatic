import logging

from typing import Any, List, Optional

from annomatic.llm.base import Model, Response, ResponseList
from annomatic.llm.openai.utils import _build_response, build_message

LOGGER = logging.getLogger(__name__)

try:
    import openai
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
    SUPPORTED_MODEL = ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-instruct"]
    COMPLETION_ONLY = ["gpt-3.5-turbo-instruct"]

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gpt-3.5-turbo",
        temperature=0.0,
    ):
        """
        Initialize the OpenAI model.

        Arguments:
            api_key: The API key for accessing the OpenAI API.
            model_name: string representing the selected model.
                (Default="gpt-3.5-turbo")
            temperature: The temperature parameter for text generation.
                (Default=0.0)

        Raises:
            ValueError: If no API key is provided.
        """
        self._model = valid_model(model_name=model_name)
        self._temperature = temperature

        self.system_prompt: Optional[str] = None

        # TODO otherwise use environment variable
        if api_key == "":
            raise ValueError("No OPEN AI key given!")

        openai.api_key = api_key

    def predict(self, messages: Any) -> ResponseList:
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
        if isinstance(messages, str):
            return self._predict_single(messages)
        elif isinstance(messages, list) and all(
            isinstance(item, str) for item in messages
        ):
            return self._predict_list(messages)
        else:
            raise NotImplementedError(
                "unknown type! Needs to be str or List[str]!",
            )

    def _predict_list(self, messages: List[str]):
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
        if self._model in self.COMPLETION_ONLY:
            raise ValueError(
                "multiple messages for Legacy API not implemented!",
            )

        messages = self.build_messages(messages)
        api_response = self._call_chat_completions_api(messages=messages)
        return _build_response(api_response)

    def _predict_single(self, content: str):
        """
        Predict response for a single prompt.

        Arguments:
            content: string with the content of the prompt.

        Returns:
            Any: Predicted output based on the provided prompt.

        """
        if self._model in self.COMPLETION_ONLY:
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

        try:
            # Make your OpenAI API request here
            return openai.Completion.create(
                model=self._model,
                prompt=prompt,
            )
        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            LOGGER.error(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            LOGGER.error(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.error.RateLimitError as e:
            LOGGER.error(f"OpenAI API returned an API Error: {e}")
            pass

    def _call_chat_completions_api(self, messages: List[str]):
        """
        Makes the function call to the Chat Completion API like specified in
        https://platform.openai.com/docs/guides/gpt/chat-completions-api


        Args:
            messages: List of string representation of the given Prompts

        Returns:
            The Completion object produced by the OpenAI Model
        """

        try:
            return openai.ChatCompletion.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )
        except openai.error.APIError as e:
            # Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            # Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            pass
        except openai.error.RateLimitError as e:
            # Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass

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
        if self._model not in self.COMPLETION_ONLY:
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
