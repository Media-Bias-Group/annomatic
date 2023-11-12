from typing import List, Optional, Union

from annomatic.llm.base import ModelConfig


class VllmConfig(ModelConfig):
    """
    VllmConfig is a class is a wrapper for the configuration of the
    SamplingParams class of the VLLM library.
    """

    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        length_penalty: float = 1.0,
        early_stopping: Union[bool, str] = False,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: Optional[int] = None,
        prompt_logprobs: Optional[int] = None,
        skip_special_tokens: bool = True,
    ) -> None:
        self.n = n
        self.best_of = best_of
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.stop = stop
        self.stop_token_ids = stop_token_ids
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
        self.logprobs = logprobs
        self.prompt_logprobs = prompt_logprobs
        self.skip_special_tokens = skip_special_tokens


class VllmBenchmarkConfig(VllmConfig):
    """
    VllmBenchmarkConfig is a class that holds the configuration for
    the HuggingFace models that use for the benchmarking of the models.

    The temperature is set to 0.2 and do_sample to True.

    """

    def __init__(self, **kwargs):
        super().__init__(temperature=0.2, **kwargs)
