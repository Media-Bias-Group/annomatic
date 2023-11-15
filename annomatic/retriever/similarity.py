from typing import Optional

import numpy as np
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
        model_name: str = "all-mpnet-base-v2",
        seed: int = 42,
    ):
        super().__init__(k=k, model_name=model_name, seed=seed)

    def select(self, pool, query: Optional[str] = None) -> list:
        """
        Selects the most similar responses from the given pool of messages.

        Args:
            pool: List of messages to select the best response from.
            query: Query to select the best response for.

        Returns:
            The k most similar responses from the given pool of messages.
        """
        if query is None:
            raise ValueError("Query must not be None")

        self._compute_embeddings(pool)

        if self.embeddings is None:
            raise ValueError("Embeddings must not be None")

        # Compute embeddings for the query sentence
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            convert_to_tensor=True,
        )

        # Compute cosine similarities between the query and pool embeddings
        cos_similarities = [
            util.pytorch_cos_sim(query_embedding, embedding)
            for embedding in self.embeddings
        ]
        # Convert to numpy array
        cos_similarities_np = [
            cos_sim.numpy().flatten()[0] for cos_sim in cos_similarities
        ]

        # Get k-nearest neighbors in descending order
        k_nearest_neighbors = np.array(cos_similarities_np).argsort()[
            -self.k :
        ][::-1]

        return [pool[i] for i in k_nearest_neighbors]
