from typing import List, Optional

from annomatic.llm.base import ModelConfig


class OpenAiConfig(ModelConfig):
    """
    OpenAiConfigMixin is a class that holds the configuration for the
    OpenAI models.
    """

    def __init__(
        self,
        frequency_penalty: float = 0.0,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: int = 1,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        top_p: float = 1.0,
        user: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.max_tokens = max_tokens
        self.n = n
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.temperature = temperature
        self.seed = seed
        self.top_p = top_p
        self.user = user
        self.kwargs = kwargs
