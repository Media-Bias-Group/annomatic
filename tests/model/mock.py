from typing import List

from annomatic.llm.openai.model import OpenAiModel

TEST_OPEN_AI_RESPONSE_CHAT = {
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "The 2020 World Series was played in "
                "Texas at Globe Life Field in Arlington.",
                "role": "assistant",
            },
        },
    ],
    "created": 1677664795,
    "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
    "model": "gpt-3.5-turbo-0613",
    "object": "chat.completion",
    "usage": {
        "completion_tokens": 17,
        "prompt_tokens": 57,
        "total_tokens": 74,
    },
}

TEST_OPEN_AI_RESPONSE_LEGACY = {
    "choices": [
        {
            "finish_reason": "length",
            "index": 0,
            "logprobs": "null",
            "text": '\n\n"Let Your Sweet Tooth Run Wild at Our '
            "Creamy Ice Cream Shack",
        },
    ],
    "created": 1683130927,
    "id": "cmpl-7C9Wxi9Du4j1lQjdjhxBlO22M61LD",
    "model": "gpt-3.5-turbo-instruct",
    "object": "text_completion",
    "usage": {
        "completion_tokens": 16,
        "prompt_tokens": 10,
        "total_tokens": 26,
    },
}


class FakeOpenAiModel(OpenAiModel):
    """
    Mock model of the OpenAI model
    """

    def __init__(self, model="gpt-3.5-turbo"):
        super().__init__(api_key="test_key", model=model)

    def _call_completion_api(self, prompt: str):
        """
        Mocking the API call

        Output is the example at
        https://platform.openai.com/docs/guides/gpt/completions-api

        Returns am mocked output for testing
        """

        result = TEST_OPEN_AI_RESPONSE_LEGACY

        return result

    def _call_chat_completions_api(self, messages: List[str]):
        """
        Mocking the API call

        Output is the example at
        https://platform.openai.com/docs/guides/gpt/chat-completions-api

        Returns am mocked output for testing
        """

        response = TEST_OPEN_AI_RESPONSE_CHAT

        return response
