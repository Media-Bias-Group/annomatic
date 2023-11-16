from abc import ABC, abstractmethod
from typing import Optional

from sentence_transformers import SentenceTransformer


class Retriever(ABC):
    """
    Base class for all retrievers.
    """

    def __init__(
        self,
        k: int,
        model_name: str = "BAAI/llm-embedder",
        seed: int = 42,
    ):
        self.model_name = model_name
        self.k = k
        self.model = SentenceTransformer(model_name)
        self.seed = seed
        self.embeddings: Optional[list] = None

    def _compute_embeddings(self, pool: list):
        """
        Computes the embeddings for the given pool of messages.

        Computes the embeddings for the given pool of messages using the
        SentenceTransformer model.
        Cached the embeddings if the pool size has not changed.

        Args:
            pool: List of messages to compute the embeddings for.

        """
        # Compute embeddings for the pool if not already computed
        if self.embeddings is None or len(self.embeddings) != len(pool):
            self.embeddings = self.model.encode(
                pool,
                show_progress_bar=True,
                convert_to_tensor=True,
            )
        return self.embeddings

    @abstractmethod
    def select(self, pool, query: Optional[str] = None) -> list:
        """
        Selects the best response from the given pool of messages.

        Args:
            pool: List of messages to select the best response from.
            query: Query to select the best response for (optional).
        """
        raise NotImplementedError
