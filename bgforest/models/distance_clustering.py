"""
Bayesian Distance Clustering (Duan & Dunson, 2021, JMLR)
=========================================================
Implements non-parametric Bayesian clustering based directly on pairwise
distance matrices D_ij without distributional/shape assumptions on raw coordinates.
"""

from typing import Optional, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans


class BayesianDistanceClustering:
    """
    Experimental distance-only clustering estimator.

    This lightweight pairwise-distance prototype is not a complete
    reproduction of the integrated Bayesian Distance Clustering sampler.

    Parameters
    ----------
    n_clusters : int, default=2
        Target number of clusters.
    n_iter : int, default=300
        Number of MCMC sampling iterations.
    burn_in : int, default=100
        Number of burn-in iterations.
    metric : str, default="euclidean"
        Distance metric for pairwise distances ("euclidean", "cityblock", "cosine").
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        n_iter: int = 300,
        burn_in: int = 100,
        metric: str = "euclidean",
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        self.n_clusters = int(n_clusters)
        self.n_iter = int(n_iter)
        self.burn_in = int(burn_in)
        self.metric = metric
        self.random_state = random_state

        if isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)

        self.labels_: Optional[np.ndarray] = None
        self.distance_matrix_: Optional[np.ndarray] = None
        self.co_clustering_matrix_: Optional[np.ndarray] = None

        if self.n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1.")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative.")

    def compute_distance_log_likelihood(self, D: np.ndarray, partition: np.ndarray) -> float:
        """
        Compute marginal log-likelihood P(D | C) over pairwise distances.
        """
        n = D.shape[0]
        same_mask = partition[:, None] == partition[None, :]
        diff_mask = ~same_mask

        # Extract intra-cluster and inter-cluster distances
        intra_dists = D[same_mask & ~np.eye(n, dtype=bool)]
        inter_dists = D[diff_mask]

        if len(intra_dists) == 0 or len(inter_dists) == 0:
            return -np.inf

        # Parameterize Exponential / Gamma distributions
        lambda_intra = 1.0 / max(1e-4, np.mean(intra_dists))
        lambda_inter = 1.0 / max(1e-4, np.mean(inter_dists))

        log_lik_intra = np.sum(np.log(lambda_intra) - lambda_intra * intra_dists)
        log_lik_inter = np.sum(np.log(lambda_inter) - lambda_inter * inter_dists)

        return float(log_lik_intra + log_lik_inter)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BayesianDistanceClustering":
        """
        Fit Bayesian Distance Clustering model to data or precomputed pairwise distance matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            Input coordinate data matrix or precomputed distance matrix.

        Returns
        -------
        self : BayesianDistanceClustering
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a non-empty 2D array containing finite values.")
        n = X.shape[0]
        if not 1 <= self.n_clusters <= n:
            raise ValueError("n_clusters must be between 1 and n_samples.")

        if X.shape[0] == X.shape[1] and np.allclose(X, X.T) and np.all(np.diag(X) == 0.0):
            D = np.copy(X)
        else:
            D = squareform(pdist(X, metric=self.metric))

        self.distance_matrix_ = D

        # Build similarity W from distance D for MCMC proposals
        sigma = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        W = np.exp(-(D**2) / (2.0 * (sigma**2)))
        np.fill_diagonal(W, 0.0)

        # Initial partition via Spectral Clustering on W
        from sklearn.cluster import SpectralClustering

        try:
            sc = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity="precomputed",
                random_state=self.rng.randint(0, 10000),
            )
            curr_partition = sc.fit_predict(W)
        except Exception:
            curr_partition = self.rng.randint(0, self.n_clusters, size=n)

        curr_log_lik = self.compute_distance_log_likelihood(D, curr_partition)

        samples = []
        effective_burn_in = min(self.burn_in, self.n_iter - 1)
        for it in range(self.n_iter):
            # Propose relocate move
            node = self.rng.randint(0, n)
            spatial_nbrs = np.argsort(D[node])[1:10]
            diff_nbrs = [j for j in spatial_nbrs if curr_partition[j] != curr_partition[node]]

            if len(diff_nbrs) > 0:
                tn = self.rng.choice(diff_nbrs)
                prop = np.copy(curr_partition)
                old_c = prop[node]
                new_c = prop[tn]

                if np.sum(prop == old_c) > 1:
                    prop[node] = new_c
                    prop_log_lik = self.compute_distance_log_likelihood(D, prop)

                    if not np.isneginf(prop_log_lik):
                        delta = prop_log_lik - curr_log_lik
                        if np.log(self.rng.uniform(0.0, 1.0)) < delta:
                            curr_partition = prop
                            curr_log_lik = prop_log_lik

            if it >= effective_burn_in:
                samples.append(np.copy(curr_partition))

        # Compute co-clustering matrix and final labels
        P = np.zeros((n, n), dtype=np.float64)
        for s in samples:
            P += (s[:, None] == s[None, :]).astype(np.float64)
        P /= float(max(1, len(samples)))

        self.co_clustering_matrix_ = P

        if self.n_clusters == 1:
            self.labels_ = np.zeros(n, dtype=np.int64)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(P)
            embedding = eigenvectors[:, np.argsort(eigenvalues)[::-1][: self.n_clusters]]
            self.labels_ = KMeans(
                n_clusters=self.n_clusters, n_init=10, random_state=self.random_state
            ).fit_predict(embedding)
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit model and return cluster label predictions.
        """
        self.fit(X, y)
        return self.labels_
