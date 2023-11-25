import pytest

from annomatic.llm.base import ResponseList
from annomatic.llm.util import build_message
from tests.model.mock import (
    TEST_OPEN_AI_RESPONSE_CHAT,
    TEST_OPEN_AI_RESPONSE_LEGACY,
    FakeOpenAiModel,
)


def test_message_system():
    exp_res = {"role": "system", "content": "This is a prompt!"}
    res = build_message("This is a prompt!", role="system")

    assert exp_res == res


def test_build_messages_with_system():
    model = FakeOpenAiModel()

    exp_res = [
        {"role": "system", "content": "This is a system prompt!"},
        {"role": "user", "content": "This is a prompt!"},
    ]
    model.system_prompt = "This is a system prompt!"
    res = model.build_chat_messages(["This is a prompt!"])

    assert exp_res == res


def test_build_response_chat_single():
    model = FakeOpenAiModel()

    res = model.predict("This is a nice prompt")

    assert (
        isinstance(res, ResponseList)
        and res.responses[0].answer == "The 2020 World Series was played in "
        "Texas at Globe Life Field in Arlington."
        and res.responses[0].data == TEST_OPEN_AI_RESPONSE_CHAT
    )


def test_build_response_legacy():
    model = FakeOpenAiModel(model_name="gpt-3.5-turbo-instruct")
    res = model.predict("This is a nice prompt")
    assert (
        isinstance(res, ResponseList)
        and res.responses[0].answer
        == '\n\n"Let Your Sweet Tooth Run Wild at Our '
        "Creamy Ice Cream Shack"
        and res.responses[0].data == TEST_OPEN_AI_RESPONSE_LEGACY
    )
