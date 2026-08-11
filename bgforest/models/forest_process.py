"""
Forest Process Prior Model
==========================
Implements the Forest Process prior distribution P(C | alpha, beta, theta) 
as an urn process extension to similarity graphs (Duan & Roy, 2022/2023).
"""

from typing import Optional
import numpy as np
from scipy.special import gammaln
from bgforest.core.matrix_tree import compute_forest_log_spanning_trees, compute_cluster_log_spanning_trees


class ForestProcess:
    """
    Forest Process prior over data partitions on similarity graphs.

    Parameters
    ----------
    alpha : float, default=1.0
        Concentration hyper-parameter (> 0). Higher alpha favors more clusters.
    beta : float, default=0.0
        Discount parameter in [0, 1). Controls power-law cluster size tail.
    theta : float, default=0.1
        Spanning tree weight exponent (>= 0). Controls graph connectivity penalty.
    size_weight : float, default=0.05
        Scaling factor for cluster size Gamma term to prevent single-cluster collapse.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.0,
        theta: float = 0.1,
        size_weight: float = 0.05
    ):
        if alpha <= 0:
            raise ValueError("alpha concentration parameter must be strictly positive.")
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta discount parameter must be in [0, 1).")
        if theta < 0:
            raise ValueError("theta exponent must be non-negative.")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.theta = float(theta)
        self.size_weight = float(size_weight)

    def log_prior(self, W: np.ndarray, partition: np.ndarray) -> float:
        """
        Compute unnormalized log-prior probability for a given data partition.
        """
        partition = np.asarray(partition, dtype=np.int64)
        unique_clusters, counts = np.unique(partition, return_counts=True)
        K = len(unique_clusters)

        # 1. Cluster count term: K * ln(alpha)
        log_alpha_term = K * np.log(self.alpha)

        # 2. Scaled cluster size term: size_weight * sum_k ln Gamma(|C_k| - beta)
        log_size_term = self.size_weight * float(np.sum(gammaln(counts - self.beta)))

        # 3. Graph topological spanning forest term: theta * sum_k ln tau(C_k, W)
        if self.theta > 0:
            log_tau_total = compute_forest_log_spanning_trees(W, partition)
            if np.isneginf(log_tau_total):
                return -np.inf
            log_graph_term = self.theta * log_tau_total
        else:
            log_graph_term = 0.0

        return log_alpha_term + log_size_term + log_graph_term

    def log_prior_cluster(self, W: np.ndarray, cluster_indices: np.ndarray) -> float:
        """
        Compute contribution to log prior for a single cluster C_k.
        """
        indices = np.asarray(cluster_indices, dtype=np.int64)
        size = len(indices)
        if size == 0:
            return -np.inf

        log_alpha = np.log(self.alpha)
        log_size = self.size_weight * float(gammaln(size - self.beta))
        
        if self.theta > 0:
            log_tau = compute_cluster_log_spanning_trees(W, indices)
            if np.isneginf(log_tau):
                return -np.inf
            log_graph = self.theta * log_tau
        else:
            log_graph = 0.0

        return log_alpha + log_size + log_graph
