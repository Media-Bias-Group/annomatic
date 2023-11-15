from typing import Optional

import numpy as np
from sklearn.cluster import KMeans

from .base import Retriever


class DiversityRetriever(Retriever):
    """
    A retriever that selects diverse responses from the given pool of messages.

    Diversity sampling like described in the paper 10.48550/arXiv.2305.14264

    The retriever uses KMeans clustering to cluster the messages into
    k clusters and then selects the most diverse response from each cluster.

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

    def _compute_cluster(self, pool: list) -> np.ndarray:
        """
        Computes the clusters for the given pool of messages.

        Computes the clusters for the given pool of messages using the
        KMeans algorithm.

        Args:
            pool: List of messages to compute the clusters for.

        Returns:
            The cluster labels for the given pool of messages.
        """
        self._compute_embeddings(pool)

        kmeans = KMeans(
            n_clusters=self.k,
            init="k-means++",
            n_init="auto",
            random_state=self.seed,
        )
        kmeans.fit(self.embeddings)

        return kmeans.labels_

    def select(self, pool, query: Optional[str] = None) -> list:
        """
        Selects diverse response from the given pool of messages.

        Args:
            pool: List of messages to select the best response from.
            query: Not used in this retriever.
        """
        if query is not None:
            raise ValueError("Query must be None")

        clusters = self._compute_cluster(pool)
        selected = []

        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            # shuffle the indices with given seed
            np.random.seed(self.seed)
            np.random.shuffle(cluster_indices)

            selected_indices = cluster_indices[: min(len(cluster_indices), 1)]
            selected.extend([pool[i] for i in selected_indices])

        return selected
