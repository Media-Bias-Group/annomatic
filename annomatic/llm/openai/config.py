from typing import List, Optional

from annomatic.llm.base import ModelConfig


class OpenAiConfig(ModelConfig):
    """
    OpenAiConfig is a class that holds the Request body for the
    OpenAI chat models.

    The default values are aligned with the default values in the
    https://platform.openai.com/docs/api-reference/chat/create

    """

    def __init__(
        self,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        presence_penalty: float = 0.0,
        response_format: Optional[str] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        user: Optional[str] = None,
        **kwargs,
    ) -> None:
        super.__init__(**kwargs)
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.max_tokens = max_tokens
        self.n = n
        self.presence_penalty = presence_penalty
        self.response_format = response_format
        self.stop = stop
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        self.user = user

    @staticmethod
    def get_default_values() -> dict:
        return {
            "frequency_penalty": 0.0,
            "logit_bias": None,
            "max_tokens": None,
            "n": 1,
            "presence_penalty": 0.0,
            "response_format": None,
            "stop": None,
            "seed": None,
            "temperature": 1.0,
            "top_p": 1.0,
            "user": None,
        }


class OpenAiBenchmarkConfig(OpenAiConfig):
    """
    OpenAiBenchmarkConfig is a class that holds the configuration for
    the HuggingFace models that use for the benchmarking of the models.

    The temperature is set to 0.2.

    """

    def __init__(self, **kwargs):
        super().__init__(temperature=0.2, **kwargs)


if __name__ == "__main__":
    config = OpenAiBenchmarkConfig()
    print(config.to_dict())
