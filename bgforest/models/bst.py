"""
Bayesian Spanning Tree (BST) for Feature Dependence Network Backbone Estimation
=================================================================================
Implements Bayesian Spanning Tree graph estimation (Duan & Dunson, 2024, JMLR)
for discovering the primary backbone structure of high-dimensional variable 
dependence graphs without full precision matrix inversion.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from bgforest.core.matrix_tree import compute_forest_log_spanning_trees
from bgforest.samplers.wilson import WilsonLERWSampler


class BayesianSpanningTree:
    """
    Bayesian Spanning Tree (BST) Estimator for variable dependence graph backbone discovery.

    Parameters
    ----------
    sigma_scale : float, default=1.0
        Bandwidth scaling parameter for feature pair similarity.
    n_samples_tree : int, default=100
        Number of posterior spanning tree samples to generate via Wilson's algorithm.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        sigma_scale: float = 1.0,
        n_samples_tree: int = 100,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        self.sigma_scale = float(sigma_scale)
        self.n_samples_tree = int(n_samples_tree)
        self.random_state = random_state

        if isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)

        self.W_feature_ : Optional[np.ndarray] = None
        self.mst_edges_ : List[Tuple[int, int]] = []
        self.backbone_adjacency_ : Optional[np.ndarray] = None
        self.posterior_edge_probabilities_ : Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "BayesianSpanningTree":
        """
        Fit Bayesian Spanning Tree model to feature matrix X of shape (n_samples, n_features).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, p_features)
            Input data matrix.

        Returns
        -------
        self : BayesianSpanningTree
        """
        X = np.asarray(X, dtype=np.float64)
        n, p = X.shape

        if p < 2:
            raise ValueError("Feature dimension p must be >= 2 for dependency backbone estimation.")

        # 1. Compute Pairwise Feature Distance / Dissimilarity S_uv
        # Standardize features
        X_std = (X - np.mean(X, axis=0)) / np.maximum(1e-8, np.std(X, axis=0))
        
        # Pairwise distance between features: S_uv = ||X_u - X_v||^2 / n
        cov_matrix = np.corrcoef(X_std, rowvar=False)
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)
        
        # Dissimilarity: d(u, v)^2 = 2 * (1 - corr(u, v))
        D_feat = np.maximum(0.0, 2.0 * (1.0 - cov_matrix))
        np.fill_diagonal(D_feat, 0.0)

        # 2. Build Feature Weight Matrix W
        sigma_sq = self.sigma_scale * (np.median(D_feat[D_feat > 0]) if np.any(D_feat > 0) else 1.0)
        sigma_sq = max(1e-4, sigma_sq)
        
        W = np.exp(- D_feat / (2.0 * sigma_sq))
        np.fill_diagonal(W, 0.0)
        self.W_feature_ = W

        # 3. Extract Maximum Spanning Tree (MST) Backbone
        # Scipy minimum_spanning_tree minimizes cost -> use -W or D_feat as cost
        mst_sparse = minimum_spanning_tree(D_feat)
        mst_dense = mst_sparse.toarray()
        
        edges = []
        backbone_adj = np.zeros((p, p), dtype=np.float64)
        for u in range(p):
            for v in range(p):
                if mst_dense[u, v] > 0 or mst_dense[v, u] > 0:
                    w_val = W[u, v]
                    backbone_adj[u, v] = w_val
                    backbone_adj[v, u] = w_val
                    if u < v:
                        edges.append((u, v))

        self.mst_edges_ = edges
        self.backbone_adjacency_ = backbone_adj

        # 4. Sample Spanning Trees via Wilson's LERW to compute Posterior Edge Inclusion Probabilities
        sampler = WilsonLERWSampler(random_state=self.rng)
        edge_counts = np.zeros((p, p), dtype=np.float64)

        for _ in range(self.n_samples_tree):
            sampled_edges = sampler.sample_spanning_tree(W)
            for u, v in sampled_edges:
                edge_counts[u, v] += 1.0
                edge_counts[v, u] += 1.0

        self.posterior_edge_probabilities_ = edge_counts / float(self.n_samples_tree)
        return self

    def get_backbone_network(self) -> np.ndarray:
        """
        Return binary backbone adjacency matrix of the maximum spanning tree.
        """
        if self.backbone_adjacency_ is None:
            raise RuntimeError("Model is not fitted. Call `fit()` first.")
        return (self.backbone_adjacency_ > 0).astype(np.int64)
