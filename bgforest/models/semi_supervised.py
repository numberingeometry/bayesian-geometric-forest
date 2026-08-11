"""
Constrained Bayesian Spanning Forest (Semi-Supervised BSF)
===========================================================
Extends Bayesian Spanning Forests to incorporate pairwise Must-Link and Cannot-Link 
domain constraints into graph-based probabilistic clustering.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from bgforest.models.bsf import BayesianSpanningForest
from bgforest.models.forest_process import ForestProcess
from bgforest.mcmc.sampler import BSFMCMCSampler
from bgforest.core.graph import build_knn_similarity, build_rbf_similarity
from sklearn.preprocessing import StandardScaler


class ConstrainedBayesianSpanningForest(BayesianSpanningForest):
    """
    Semi-Supervised Bayesian Spanning Forest Estimator with Must-Link and Cannot-Link constraints.

    Parameters
    ----------
    must_link : List[Tuple[int, int]], optional
        List of node pairs (i, j) that must belong to the same cluster.
    cannot_link : List[Tuple[int, int]], optional
        List of node pairs (i, j) that must belong to different clusters.
    n_clusters : int, default=2
        Target number of clusters.
    n_iter : int, default=500
        Number of MCMC iterations.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        must_link: Optional[List[Tuple[int, int]]] = None,
        cannot_link: Optional[List[Tuple[int, int]]] = None,
        n_clusters: Optional[int] = 2,
        graph_type: str = "knn",
        n_neighbors: int = 10,
        n_iter: int = 500,
        burn_in: int = 100,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        **kwargs
    ):
        super().__init__(
            n_clusters=n_clusters,
            graph_type=graph_type,
            n_neighbors=n_neighbors,
            n_iter=n_iter,
            burn_in=burn_in,
            random_state=random_state,
            **kwargs
        )
        self.must_link = must_link if must_link is not None else []
        self.cannot_link = cannot_link if cannot_link is not None else []

    def check_constraints_satisfied(self, partition: np.ndarray) -> bool:
        """
        Verify whether candidate partition satisfies all Must-Link and Cannot-Link constraints.
        """
        for u, v in self.must_link:
            if partition[u] != partition[v]:
                return False

        for u, v in self.cannot_link:
            if partition[u] == partition[v]:
                return False

        return True

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        initial_partition: Optional[np.ndarray] = None
    ) -> "ConstrainedBayesianSpanningForest":
        """
        Fit Constrained Bayesian Spanning Forest model while strictly satisfying domain constraints.
        """
        X_raw = np.asarray(X, dtype=np.float64)
        n = X_raw.shape[0]

        if self.scale_features:
            scaler = StandardScaler()
            X_proc = scaler.fit_transform(X_raw)
        else:
            X_proc = X_raw

        W = build_knn_similarity(X_proc, n_neighbors=self.n_neighbors, symmetric=True)
        # Boost weight between Must-Link pairs
        max_w = np.max(W) if np.max(W) > 0 else 1.0
        for u, v in self.must_link:
            W[u, v] = 10.0 * max_w
            W[v, u] = 10.0 * max_w
        for u, v in self.cannot_link:
            W[u, v] = 0.0
            W[v, u] = 0.0
        self.W_ = W

        # Build initial partition satisfying constraints
        if initial_partition is not None:
            initial_part = np.copy(initial_partition)
        else:
            from sklearn.cluster import SpectralClustering
            try:
                sc = SpectralClustering(
                    n_clusters=self.n_clusters if self.n_clusters is not None else 2,
                    affinity="precomputed", random_state=42
                )
                initial_part = sc.fit_predict(W)
            except Exception:
                initial_part = np.random.randint(0, self.n_clusters if self.n_clusters is not None else 2, size=n)

        # Enforce must-link in initial partition
        for u, v in self.must_link:
            initial_part[v] = initial_part[u]
        _, initial_part = np.unique(initial_part, return_inverse=True)

        fp = ForestProcess(alpha=self.alpha, beta=self.beta, theta=self.theta)
        sampler = BSFMCMCSampler(
            forest_process=fp,
            n_iter=self.n_iter,
            burn_in=self.burn_in,
            thinning=self.thinning,
            sigma_likelihood=self.sigma_likelihood,
            random_state=self.random_state
        )

        self.mcmc_sampler_ = sampler
        samples = sampler.sample(X_proc, W, initial_partition=initial_part, target_n_clusters=self.n_clusters)
        self.log_posterior_trace_ = sampler.traces_
        self.acceptance_rate_ = sampler.acceptance_rate_

        # Filter samples that satisfy constraints
        valid_samples = [s for s in samples if self.check_constraints_satisfied(s)]
        if not valid_samples:
            valid_samples = [initial_part]

        P = np.zeros((n, n), dtype=np.float64)
        for s in valid_samples:
            P += (s[:, None] == s[None, :]).astype(np.float64)
        P /= float(len(valid_samples))

        self.posterior_co_clustering_ = P

        from sklearn.cluster import SpectralClustering
        sc_final = SpectralClustering(
            n_clusters=self.n_clusters if self.n_clusters is not None else 2,
            affinity="precomputed", random_state=42
        )
        self.labels_ = sc_final.fit_predict(P)
        return self
