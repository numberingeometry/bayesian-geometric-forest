"""
Spanning-tree backbone of a feature dependence graph
====================================================

A descriptive tool, inspired by Bayesian spanning tree estimation (Duan & Dunson,
JMLR 2023) but not a reproduction of that model.

* ``backbone_edges_``: the spanning tree of the features that minimises total
  dissimilarity 1 - |r_uv|. For Gaussian data the mutual information of a pair,
  -log(1 - r^2) / 2, is increasing in |r|, so this is the Chow-Liu tree.
* ``edge_probability_``: the probability that each feature pair is an edge of a
  random spanning tree drawn with Pr(T) proportional to prod_{e in T} w_e on the
  kernel w_uv = exp(-(1 - |r_uv|) / bandwidth). It is computed exactly as
  w_e times the effective resistance of e (Kirchhoff), and ``sample_trees`` draws
  such trees exactly with Wilson's algorithm.
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from bgforest.samplers.wilson import RandomLike, WilsonSampler


class SpanningTreeBackbone:
    """Backbone tree and edge-inclusion probabilities for the features of X.

    Parameters
    ----------
    bandwidth : float, optional
        Kernel bandwidth on the dissimilarity 1 - |r|. Defaults to the median
        off-diagonal dissimilarity.
    random_state : int or numpy Generator, optional
        Seed for :meth:`sample_trees`.
    """

    def __init__(self, bandwidth: Optional[float] = None, random_state: RandomLike = None):
        self.bandwidth = bandwidth
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "SpanningTreeBackbone":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("X must be (n_samples, n_features) with at least two features.")
        p = X.shape[1]
        R = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)
        D = 1.0 - np.abs(R)
        np.fill_diagonal(D, 0.0)
        off = D[~np.eye(p, dtype=bool)]
        h = float(self.bandwidth) if self.bandwidth is not None else float(np.median(off))
        h = max(h, 1e-6)
        W = np.exp(-D / h)
        np.fill_diagonal(W, 0.0)

        cost = D.copy()
        cost[(cost == 0) & ~np.eye(p, dtype=bool)] = 1e-12  # csgraph reads 0 as "no edge"
        mst = minimum_spanning_tree(cost).toarray()
        self.backbone_edges_: List[Tuple[int, int]] = sorted(
            (min(u, v), max(u, v)) for u, v in zip(*np.nonzero(mst))
        )
        L = np.diag(W.sum(axis=1)) - W
        G = np.linalg.pinv(L)
        g = np.diag(G)
        self.edge_probability_ = W * (g[:, None] + g[None, :] - 2.0 * G)
        np.fill_diagonal(self.edge_probability_, 0.0)
        self.correlation_ = R
        self.weights_ = W
        self.bandwidth_ = h
        return self

    def sample_trees(self, n_trees: int) -> List[List[Tuple[int, int]]]:
        """Exact draws of spanning trees from the kernel, as (child, parent) edge lists."""
        if not hasattr(self, "weights_"):
            raise RuntimeError("call fit() first.")
        sampler = WilsonSampler(self.random_state)
        return [sampler.sample_spanning_tree(self.weights_) for _ in range(int(n_trees))]
