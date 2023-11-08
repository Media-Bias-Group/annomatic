from typing import List, Optional

from annomatic.llm.base import ModelConfig


class HuggingFaceConfig(ModelConfig):
    """
    HuggingFaceConfigMixin is a class that holds the configuration for the
    HuggingFace models.

    """

    def to_dict(self):
        pass

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
        self.kwargs = kwargs
