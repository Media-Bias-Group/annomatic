from typing import Any, Dict, List, Optional

from annomatic.config.base import ModelConfig


class HuggingFaceConfig(ModelConfig):
    """
    HuggingFaceConfigMixin is a class that holds the configuration for the
    HuggingFace models.

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


class HuggingFaceBenchmarkConfig(HuggingFaceConfig):
    """
    HuggingFaceBenchmarkConfig is a class that holds the configuration for
    the HuggingFace models that use for the benchmarking of the models.

    The temperature is set to 0.2 and do_sample to True.
    """

    def __init__(self, **kwargs):
        if (
            kwargs.get("temperature") is not None
            or kwargs.get("do_sample") is not None
        ):
            raise ValueError(
                "Temperature should not be modified for benchmarking!",
            )
        super().__init__(temperature=0.2, do_sample=True, **kwargs)
