from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union


class ModelConfig(ABC):
    """
    Base ModelConfig for LLMs. The ModelConfig is used to hold the
    hyperparameters for generations of the models.
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initialize the model config.
        """
        self.kwargs = kwargs

    @staticmethod
    def get_default_values() -> Dict[str, Any]:
        """
        Return a dictionary of the default values for the model config.
        """
        raise NotImplementedError()

    def to_dict(self, exclude_kwargs: bool = False) -> Dict[str, Any]:
        """
        Convert the model config to a dictionary.

        Values that are different from the values set in the __init__ method
        are included in the dictionary, and the kwargs are flattened.

        Returns:
            dict: A dictionary representing the model configuration.
        """
        default_values = self.get_default_values()
        config_dict = {}

        for key, value in default_values.items():
            if getattr(self, f"{key}", None) != value:
                config_dict[key] = getattr(self, f"{key}")

        if not exclude_kwargs:
            config_dict.update(self.kwargs.items())

        return config_dict


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
        super().__init__(**kwargs)
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


class HuggingFaceConfig(ModelConfig):
    """
    HuggingFaceConfigMixin is a class that holds the generation hyperparams
    for the HuggingFace models.

    Along with the generation hyperparams, the class also holds the
    load_args and the token_args that are used to initialize the model and
    tokenizer.

    The default values are aligned with the default of the generation method
     of the transformers' library. See 'Generative models' in the
    https://huggingface.co/transformers/v3.4.0/main_classes/model.html

    """

    def __init__(
        self,
        max_length: int = 20,
        min_length: int = 10,
        do_sample: bool = False,
        early_stopping: bool = False,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        bad_words_ids: Optional[List[int]] = None,
        num_return_sequences: int = 1,
        load_args: Optional[Dict[str, Any]] = None,
        token_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.max_length = max_length
        self.min_length = min_length
        self.do_sample = do_sample
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.bad_words_ids = bad_words_ids
        self.num_return_sequences = num_return_sequences
        self.load_args = load_args
        self.token_args = token_args

    @staticmethod
    def get_default_values() -> Dict[str, Any]:
        return {
            "max_length": 20,
            "min_length": 10,
            "do_sample": False,
            "early_stopping": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "bad_words_ids": None,
            "num_return_sequences": 1,
        }


class VllmConfig(ModelConfig):
    """
    VllmConfig is a class is a wrapper for the configuration of the
    SamplingParams class of the VLLM library.

    Alongside these parameters, the class also holds the load_args
    that are used to initialize the LLM.
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
        load_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
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
        self.load_args = load_args

    @staticmethod
    def get_default_values() -> dict:
        return {
            "n": 1,
            "best_of": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "use_beam_search": False,
            "length_penalty": 1.0,
            "early_stopping": False,
            "stop": None,
            "stop_token_ids": None,
            "ignore_eos": False,
            "max_tokens": 16,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }
