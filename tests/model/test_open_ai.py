from unittest.mock import patch

from annomatic.llm.base import Response, ResponseList
from annomatic.llm.openai import OpenAiModel
from annomatic.llm.util import build_message
from tests.model.mock import (
    TEST_OPEN_AI_RESPONSE_CHAT,
    TEST_OPEN_AI_RESPONSE_LEGACY,
)


def test_message_system():
    exp_res = {"role": "system", "content": "This is a prompt!"}
    res = build_message("This is a prompt!", role="system")

    assert exp_res == res


def test_build_messages_with_system():
    with patch.object(
        OpenAiModel,
        "predict",
        return_value=ResponseList.from_responses(
            [
                Response(
                    answer="The 2020 World Series was played in Texas at "
                    "Globe Life Field in Arlington.",
                    data=TEST_OPEN_AI_RESPONSE_CHAT,
                    query="This is a nice prompt",
                ),
            ],
        ),
    ):
        model = OpenAiModel(api_key="test_key")

        exp_res = [
            {"role": "system", "content": "This is a system prompt!"},
            {"role": "user", "content": "This is a prompt!"},
        ]
        model.system_prompt = "This is a system prompt!"
        res = model.build_chat_messages(["This is a prompt!"])

    assert exp_res == res


def test_build_response_chat_single():
    with patch.object(
        OpenAiModel,
        "predict",
        return_value=ResponseList.from_responses(
            [
                Response(
                    answer="The 2020 World Series was played in Texas at "
                    "Globe Life Field in Arlington.",
                    data=TEST_OPEN_AI_RESPONSE_CHAT,
                    query="This is a nice prompt",
                ),
            ],
        ),
    ):
        model = OpenAiModel(api_key="test_key")
        res = model.predict("This is a nice prompt")

    assert (
        isinstance(res, ResponseList)
        and res.responses[0].answer == "The 2020 World Series was played in "
        "Texas at Globe Life Field in Arlington."
        and res.responses[0].data == TEST_OPEN_AI_RESPONSE_CHAT
    )


def test_build_response_legacy():
    with patch.object(
        OpenAiModel,
        "predict",
        return_value=ResponseList.from_responses(
            [
                Response(
                    answer='\n\n"Let Your Sweet Tooth Run Wild at Our '
                    "Creamy Ice Cream Shack",
                    data=TEST_OPEN_AI_RESPONSE_LEGACY,
                    query="This is a nice prompt",
                ),
            ],
        ),
    ):
        model = OpenAiModel(api_key="test_key")
        res = model.predict("This is a nice prompt")

    assert (
        isinstance(res, ResponseList)
        and res.responses[0].answer
        == '\n\n"Let Your Sweet Tooth Run Wild at Our '
        "Creamy Ice Cream Shack"
        and res.responses[0].data == TEST_OPEN_AI_RESPONSE_LEGACY
    )
