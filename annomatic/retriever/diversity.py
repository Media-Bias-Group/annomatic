from typing import List, Optional, Union

import numpy as np
import pandas as pd
from haystack.lazy_imports import LazyImport

with LazyImport("Run 'pip install scikit-learn'") as lazy_import:
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
        pool: pd.DataFrame,
        text_variable: str = "text",
        label_variable: str = "label",
        model_name: str = "BAAI/llm-embedder",
        seed: int = 42,
        **kwargs,
    ):
        lazy_import.check()
        super().__init__(
            k=k,
            pool=pool,
            text_variable=text_variable,
            label_variable=label_variable,
            model_name=model_name,
            seed=seed,
            **kwargs,
        )
        self.cluster = self._compute_cluster()
        # list of indices retrieved from the pool
        self.examples_indices: Optional[list] = None

    def _compute_cluster(self) -> np.ndarray:
        """
        Computes the clusters for the given pool of messages.

        Computes the clusters for the given pool of messages using the
        KMeans algorithm.

        Returns:
            The cluster labels for the given pool of messages.
        """
        kmeans = KMeans(
            n_clusters=self.k,
            init="k-means++",
            n_init="auto",
            random_state=self.seed,
        )
        kmeans.fit(self.embeddings.cpu().numpy())

        return kmeans.labels_

    def select(self, query: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Selects diverse response from the given pool of messages.

        Args:
            query: Not used in this retriever.

        Returns:
            The k most diverse responses from the given pool of messages.
        """

        if self.examples_indices:
            return self.pool.iloc[self.examples_indices]

        selected_indices: List[int] = []
        for cluster_id in np.unique(self.cluster):
            cluster_indices = np.where(self.cluster == cluster_id)[0]
            # shuffle the indices with given seed
            np.random.seed(self.seed)
            np.random.shuffle(cluster_indices)

            selected_indices.extend(
                cluster_indices[: min(len(cluster_indices), 1)],
            )

        self.examples_indices = selected_indices
        return self.pool.iloc[selected_indices]
