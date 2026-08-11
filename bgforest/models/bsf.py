"""
Bayesian Spanning Forest Scikit-Learn Estimator
==============================================
Provides the `BayesianSpanningForest` estimator class conforming to
Scikit-Learn cluster API standards for robust graphical model-based clustering.
"""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from bgforest.core.graph import (
    build_knn_similarity,
    build_rbf_similarity,
    connect_knn_components,
)
from bgforest.mcmc.sampler import BSFMCMCSampler
from bgforest.models.forest_process import ForestProcess


class BayesianSpanningForest(BaseEstimator, ClusterMixin):
    """
    Bayesian Spanning Forest (BSF) Cluster Estimator.

    Parameters
    ----------
    n_clusters : int, optional
        Target number of clusters for final partition extraction.
        If None, automatically inferred from posterior model mode.
    graph_type : str, default='knn'
        Graph similarity matrix builder ('rbf' or 'knn').
    gamma : float, optional
        RBF kernel scale coefficient gamma = 1 / (2 * sigma^2).
    sigma : float, optional
        RBF Gaussian standard deviation bandwidth.
    adaptive_bandwidth : bool, default=False
        If True, uses local self-tuning bandwidth (Zelnik-Manor & Perona, 2004).
    n_neighbors : int, default=10
        Number of nearest neighbors for k-NN graph construction.
    alpha : float, default=1.0
        Forest Process concentration parameter (> 0).
    beta : float, default=0.0
        Forest Process discount parameter in [0, 1).
    theta : float, default=1.0
        Spanning tree weight exponent.
    n_iter : int, default=1000
        Number of MCMC iterations.
    burn_in : int, default=300
        Number of MCMC burn-in iterations.
    thinning : int, default=1
        Thinning interval for MCMC posterior sampling.
    sigma_likelihood : Optional[float], default=None
        Likelihood variance standard deviation. If None, estimated adaptively from data.
    scale_features : bool, default=True
        If True, applies StandardScaler to features prior to graph construction.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster assignments for input dataset.
    co_clustering_matrix_ : np.ndarray of shape (n_samples, n_samples)
        Posterior co-clustering matrix P_ij = P(nodes i and j in same cluster).
    spectral_eigenvectors_ : np.ndarray of shape (n_samples, n_components)
        Leading eigenvectors extracted from co-clustering matrix P.
    spectral_eigenvalues_ : np.ndarray of shape (n_components,)
        Leading eigenvalues of co-clustering matrix P.
    log_posterior_trace_ : List[float]
        Log-posterior values recorded across MCMC iterations.
    n_clusters_inferred_ : int
        Number of clusters inferred.
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        graph_type: str = "knn",
        gamma: Optional[float] = None,
        sigma: Optional[float] = None,
        adaptive_bandwidth: bool = False,
        n_neighbors: int = 10,
        alpha: float = 1.0,
        beta: float = 0.0,
        theta: float = 0.1,
        n_iter: int = 1000,
        burn_in: int = 300,
        thinning: int = 1,
        sigma_likelihood: Optional[float] = None,
        scale_features: bool = True,
        ensure_connected: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.graph_type = graph_type
        self.gamma = gamma
        self.sigma = sigma
        self.adaptive_bandwidth = adaptive_bandwidth
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thinning = thinning
        self.sigma_likelihood = sigma_likelihood
        self.scale_features = scale_features
        self.ensure_connected = ensure_connected
        self.random_state = random_state

    @staticmethod
    def _infer_n_clusters(W: np.ndarray) -> int:
        """Choose K with a transparent normalized-Laplacian eigengap heuristic."""
        n = W.shape[0]
        if n <= 2:
            return 1
        degree = np.sum(W, axis=1)
        inv_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
        L = np.eye(n) - (inv_sqrt[:, None] * W * inv_sqrt[None, :])
        eigenvalues = np.linalg.eigvalsh(L)
        max_k = min(8, n - 1)
        gaps = np.diff(eigenvalues[: max_k + 1])
        return int(np.argmax(gaps) + 1)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        initial_partition: Optional[np.ndarray] = None,
    ) -> "BayesianSpanningForest":
        """
        Fit Bayesian Spanning Forest model to input dataset X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data matrix.
        y : Ignored
        initial_partition : np.ndarray of shape (n_samples,), optional
            User-provided initial cluster assignments for MCMC initialization.

        Returns
        -------
        self : BayesianSpanningForest
            Fitted estimator instance.
        """
        X_raw = check_array(X, ensure_min_samples=1, ensure_min_features=1, dtype=np.float64)
        n_samples = X_raw.shape[0]
        if self.n_clusters is not None and not 1 <= self.n_clusters <= n_samples:
            raise ValueError("n_clusters must be between 1 and n_samples.")
        if self.graph_type not in {"rbf", "knn"}:
            raise ValueError("graph_type must be either 'rbf' or 'knn'.")

        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_proc = self.scaler_.fit_transform(X_raw)
        else:
            X_proc = X_raw

        # 1. Build Similarity Graph Adjacency W
        if self.graph_type == "rbf":
            W = build_rbf_similarity(
                X_proc,
                gamma=self.gamma,
                sigma=self.sigma,
                adaptive_bandwidth=self.adaptive_bandwidth,
            )
        else:
            W = build_knn_similarity(
                X_proc, n_neighbors=self.n_neighbors, mode="distance", symmetric=True
            )
            if self.ensure_connected:
                W = connect_knn_components(W, X_proc)
        self.W_ = W

        target_k = self.n_clusters if self.n_clusters is not None else self._infer_n_clusters(W)

        # 2. Configure Forest Process Prior & MCMC Sampler
        fp = ForestProcess(alpha=self.alpha, beta=self.beta, theta=self.theta)
        sampler = BSFMCMCSampler(
            forest_process=fp,
            n_iter=self.n_iter,
            burn_in=self.burn_in,
            thinning=self.thinning,
            sigma_likelihood=self.sigma_likelihood,
            random_state=self.random_state,
        )
        self.mcmc_sampler_ = sampler

        # Initialize partition if not explicitly supplied
        if initial_partition is not None:
            initial_part = np.asarray(initial_partition, dtype=np.int64)
        elif target_k > 1:
            from sklearn.cluster import SpectralClustering

            try:
                sc = SpectralClustering(
                    n_clusters=target_k,
                    affinity="precomputed",
                    n_init=10,
                    random_state=self.random_state,
                )
                initial_part = sc.fit_predict(W)
            except Exception:
                initial_part = (
                    np.random.RandomState(self.random_state).permutation(np.arange(n_samples))
                    % target_k
                )
        else:
            initial_part = np.zeros(n_samples, dtype=np.int64)

        # 3. Run MCMC Sampling
        sampler.sample(
            X_proc,
            W,
            initial_partition=initial_part,
            target_n_clusters=target_k,
        )
        self.log_posterior_trace_ = sampler.traces_
        self.acceptance_rate_ = sampler.acceptance_rate_

        # 4. Compute Posterior Co-Clustering Matrix P_ij
        P = sampler.compute_co_clustering_matrix()
        self.co_clustering_matrix_ = P

        # 5. Extract Spectral Eigenvectors & Cluster Partition
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        k = target_k

        self.n_clusters_inferred_ = k
        self.spectral_eigenvalues_ = eigenvalues[:k]
        self.spectral_eigenvectors_ = eigenvectors[:, :k]

        if k == 1:
            self.labels_ = np.zeros(n_samples, dtype=np.int64)
        else:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            self.labels_ = km.fit_predict(self.spectral_eigenvectors_)

        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit estimator and return cluster labels."""
        self.fit(X, y)
        return self.labels_

    def predict_proba(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return soft cluster membership probabilities derived from posterior co-clustering.
        """
        check_is_fitted(self, ["labels_", "co_clustering_matrix_"])
        if X is not None:
            X = check_array(X, dtype=np.float64)
            if X.shape[0] != len(self.labels_):
                raise ValueError("predict_proba currently supports the fitted observations only.")

        n_samples = len(self.labels_)
        k = self.n_clusters_inferred_
        proba = np.zeros((n_samples, k), dtype=np.float64)

        for c in range(k):
            cluster_nodes = np.where(self.labels_ == c)[0]
            if len(cluster_nodes) > 0:
                proba[:, c] = np.mean(self.co_clustering_matrix_[:, cluster_nodes], axis=1)

        row_sums = np.sum(proba, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-12)
        proba /= row_sums

        return proba
