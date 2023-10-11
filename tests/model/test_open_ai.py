from typing import List

import pytest

from annomatic.model.openai.openai import OpenAiModel, build_message


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

        Returns a tuple of the (input, output) for testing
        """

        result = {
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

        return result

    def _call_chat_completions_api(self, messages: List[str]):
        """
        Mocking the API call

        Output is the example at
        https://platform.openai.com/docs/guides/gpt/chat-completions-api

        Returns a tuple of the (input, output) for testing
        """

        response = {
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

        return response


def test_message_system():
    exp_res = {"role": "system", "content": "This is a prompt!"}
    res = build_message("This is a prompt!", role="system")

    assert exp_res == res


def test_add_system_prompt():
    model = FakeOpenAiModel()

    exp_res = {"role": "system", "content": "This is a prompt!"}
    model.add_system_prompt("This is a prompt!")

    assert exp_res == model.system_prompt


def test_build_messages_with_system():
    model = FakeOpenAiModel()

    exp_res = [
        {"role": "system", "content": "This is a system prompt!"},
        {"role": "user", "content": "This is a prompt!"},
    ]
    model.add_system_prompt("This is a system prompt!")
    res = model.build_messages(["This is a prompt!"])

    assert exp_res == res


def test_predict_invalid_type():
    inp = [1, "This is also a prompt!"]
    model = FakeOpenAiModel()

    with pytest.raises(NotImplementedError):
        model.predict(inp)
