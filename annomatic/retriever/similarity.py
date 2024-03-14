from typing import Optional

import pandas as pd
from sentence_transformers import util

from .base import Retriever


class SimilarityRetriever(Retriever):
    """
    Similarity sampling like described in the paper 10.48550/arXiv.2305.14264

    Follows the KATE method proposed in 10.18653/v1/2022.deelio-1.10

    The retriever uses cosine similarity to select the k most similar
    responses from the given pool of messages.

    Args:
        k: Number of examples to be selected.
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
        super().__init__(
            k=k,
            pool=pool,
            text_variable=text_variable,
            label_variable=label_variable,
            model_name=model_name,
            seed=seed,
            **kwargs,
        )

    def select(self, query: Optional[str] = None) -> pd.DataFrame:
        """
        Selects the most similar responses from the given pool of messages.

        Args:
            query: The query sentence to select the most similar responses for.

        Returns:
            The k most similar responses from the given pool of messages.
        """
        if query is None:
            raise ValueError("Query must not be None")

        if self.embeddings is None:
            raise ValueError("Embeddings must not be None")

        # Compute embeddings for the query sentence
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        # Compute cosine similarities between the query and pool embeddings
        cos_similarities = util.pytorch_cos_sim(
            query_embedding,
            self.embeddings,
        )

        # Get k-nearest neighbors in descending order
        k_nearest_neighbors = cos_similarities.topk(self.k).indices[0]

        return self.pool.iloc[k_nearest_neighbors.tolist()]
