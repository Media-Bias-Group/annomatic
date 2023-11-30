from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from sentence_transformers import SentenceTransformer


class Retriever(ABC):
    """
    Base class for all retrievers.

    Args:
        k: Number of examples to be selected.
        pool: Pool of examples to select from.
        text_variable: Name of the text column in the pool.
        label_variable: Name of the label column in the pool.
        model_name: Name of the SentenceTransformer model to use.
        seed: Seed to use for reproducibility.
    """

    def __init__(
        self,
        k: int,
        pool: pd.DataFrame,
        text_variable: str = "text",
        label_variable: str = "label",
        model_name: str = "BAAI/llm-embedder",
        seed: int = 42,
        **kwargs,
    ):
        self.k = k
        self.pool = pool
        self.text_variable = text_variable
        self.label_variable = label_variable
        self.model = SentenceTransformer(model_name)
        self.seed = seed

        self.embeddings = self._compute_embeddings(
            self.pool[text_variable].tolist(),
        )

    def _compute_embeddings(self, pool: list):
        """
        Computes the embeddings for the given pool of messages.

        Computes the embeddings for the given pool of messages using the
        SentenceTransformer model.


        Args:
            pool: List of messages to compute the embeddings for.

        """
        self.embeddings = self.model.encode(
            pool,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        return self.embeddings

    @abstractmethod
    def select(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Selects the best examples from the given pool of messages.

        Args:
            query: Query to select the best response for (optional).
                    only used in models where the selection depends
                    on the query.
        """
        raise NotImplementedError
