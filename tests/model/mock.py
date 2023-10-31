from typing import List

from vllm import CompletionOutput, RequestOutput

from annomatic.llm.huggingface import HFAutoModelForCausalLM
from annomatic.llm.huggingface.model import HFAutoModelForSeq2SeqLM
from annomatic.llm.openai import OpenAiModel
from annomatic.llm.vllm import VllmModel

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

    def __init__(self, model_name="gpt-3.5-turbo"):
        super().__init__(api_key="test_key", model_name=model_name)

    def _call_completion_api(self, prompt: str):
        """
        Mocking the API call

        Output is the example at
        https://platform.openai.com/docs/guides/gpt/completions-api

        Returns am mocked output for testing
        """

        return TEST_OPEN_AI_RESPONSE_LEGACY

    def _call_chat_completions_api(self, messages: List[str]):
        """
        Mocking the API call

        Output is the example at
        https://platform.openai.com/docs/guides/gpt/chat-completions-api

        Returns am mocked output for testing
        """

        return TEST_OPEN_AI_RESPONSE_CHAT


class FakeHFAutoModelForCausalLM(HFAutoModelForCausalLM):
    def __init__(self):
        self.model = None

    def _format_output(self, decoded_output, messages):
        return decoded_output

    def _call_llm_and_decode(
        self,
        model_inputs,
        output_length: int,
    ) -> List[str]:
        return [
            "mocked output",
            "mocked output2",
            "mocked output3",
            "mocked output4" "mocked output5",
            "mocked output6",
        ]


class FakeHFAutoModelForSeq2SeqLM(HFAutoModelForSeq2SeqLM):
    def __init__(self):
        self.model = None

    def _format_output(self, decoded_output, messages):
        return decoded_output

    def _call_llm_and_decode(
        self,
        model_inputs,
        output_length: int,
    ) -> List[str]:
        return [
            "mocked output",
            "mocked output2",
            "mocked output3",
            "mocked output4" "mocked output5",
            "mocked output6",
        ]


class FakeVllmModel(VllmModel):
    def __init__(self, model_name: str, model_args=None, param_args=None):
        self.model = "dummy"
        self.samplingParams = "Dummy"

    def _call_llm(self, messages: List[str]) -> list[RequestOutput]:
        common_request_id = "common_request"
        common_prompt = "This is a common prompt."
        common_prompt_token_ids = [1, 2, 3, 4]
        common_prompt_logprobs = None
        common_finished = True

        requestOutputs = []
        request_output_1 = RequestOutput(
            request_id=common_request_id,
            prompt=common_prompt,
            prompt_token_ids=common_prompt_token_ids,
            prompt_logprobs=common_prompt_logprobs,
            outputs=[
                CompletionOutput(
                    index=1,
                    text="output 1",
                    token_ids=[101, 102, 103, 104],
                    cumulative_logprob=0.9,
                    logprobs=None,
                    finish_reason="Completed",
                ),
                # Add more CompletionOutput instances as needed
            ],
            finished=common_finished,
        )

        requestOutputs.extend([request_output_1] * len(messages))

        return requestOutputs
