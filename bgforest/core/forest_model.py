"""
The Bayesian spanning forest as one weighted spanning tree
==========================================================

Duan & Roy (JASA 2024) model each cluster as a tree grown from a root drawn from a
diffuse density r, every other point drawn from a leaf density f centred on its
parent. Joining every root to an auxiliary node 0 turns the forest into a single
spanning tree on n + 1 nodes, and with the prior Pi(T) proportional to lambda^K the
posterior over that tree is a weighted spanning-tree law:

    Pi(T | y)  proportional to  prod_{(i,j) in T} A_ij,
    A_ij = f(y_i | y_j),   A_0i = lambda * r(y_i).

Everything in this module works with log-weights. The node 0 of the paper is stored
at index n ("aux").
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.special import gammaln, logsumexp


def gaussian_leaf_logweights(
    X: np.ndarray, scales: np.ndarray, sq_dists: Optional[np.ndarray] = None
) -> np.ndarray:
    """Log leaf density log f(y_i | y_j) for a Gaussian with variance s_i * s_j.

    This is equation (5) of Duan & Roy (2024): f is N(y_j, sigma_ij I_p) with the
    local parameterisation sigma_ij = s_i s_j. The diagonal is -inf (no self loops).
    Pass precomputed squared distances to avoid recomputing them.
    """
    X = np.asarray(X, dtype=np.float64)
    s = np.asarray(scales, dtype=np.float64)
    if X.ndim != 2 or s.shape != (X.shape[0],):
        raise ValueError("X must be (n, p) and scales must have length n.")
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("scales must be finite and strictly positive.")
    p = X.shape[1]
    sq = cdist(X, X, metric="sqeuclidean") if sq_dists is None else sq_dists
    var = np.outer(s, s)
    log_w = -0.5 * p * np.log(2.0 * np.pi * var) - sq / (2.0 * var)
    np.fill_diagonal(log_w, -np.inf)
    return log_w


def cauchy_root_logdensity(X: np.ndarray, rho: float = 1.0, center: Optional[np.ndarray] = None) -> np.ndarray:
    """Log density of the multivariate Cauchy root distribution r(y; center, rho)."""
    X = np.asarray(X, dtype=np.float64)
    if rho <= 0:
        raise ValueError("rho must be strictly positive.")
    p = X.shape[1]
    mu = np.zeros(p) if center is None else np.asarray(center, dtype=np.float64)
    sq = np.sum((X - mu) ** 2, axis=1)
    return (
        gammaln((1.0 + p) / 2.0)
        - 0.5 * (1.0 + p) * np.log(np.pi)
        - p * np.log(rho)
        - 0.5 * (1.0 + p) * np.log1p(sq / rho**2)
    )


def local_scales(X: np.ndarray, n_neighbors: int = 7) -> np.ndarray:
    """Distance to the k-th nearest neighbour, the self-tuning scale of Zelnik-Manor & Perona.

    Duan & Roy note that this heuristic behaves well and give a model-based version
    of it; here it is used directly to fix s_i, so that sigma_ij = d_i(k) d_j(k).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 2:
        return np.ones(n)
    k = int(min(max(1, n_neighbors), n - 1))
    d = np.sort(cdist(X, X), axis=1)[:, k]
    positive = d[d > 0]
    floor = float(np.min(positive)) if positive.size else 1.0
    return np.maximum(d, floor)


@dataclass
class ForestKernel:
    """Augmented-graph weights of a Bayesian spanning forest.

    Attributes
    ----------
    log_leaf : (n, n) array
        log A_ij = log f(y_i | y_j), symmetric, -inf on the diagonal.
    log_root : (n,) array
        log A_0i = log lambda + log r(y_i).
    """

    log_leaf: np.ndarray
    log_root: np.ndarray

    def __post_init__(self) -> None:
        self.log_leaf = np.asarray(self.log_leaf, dtype=np.float64)
        self.log_root = np.asarray(self.log_root, dtype=np.float64)
        n = self.log_root.shape[0]
        if self.log_leaf.shape != (n, n):
            raise ValueError("log_leaf must be (n, n) with n = len(log_root).")
        if not np.all(np.isfinite(self.log_root)):
            raise ValueError("every root weight must be finite and positive.")
        off = ~np.eye(n, dtype=bool)
        if not np.allclose(self.log_leaf[off], self.log_leaf.T[off], rtol=1e-12, atol=1e-12):
            raise ValueError("log_leaf must be symmetric.")

    @classmethod
    def from_data(
        cls,
        X: np.ndarray,
        scales: np.ndarray,
        rho: float = 1.0,
        lam: float = 0.5,
        center: Optional[np.ndarray] = None,
        sq_dists: Optional[np.ndarray] = None,
    ) -> "ForestKernel":
        """Gaussian leaves and a Cauchy root, as in Duan & Roy (2024), eq. (5).

        ``rho`` is the Cauchy root scale (gamma in the paper) and ``lam`` the prior
        weight per cluster, Pi(T) proportional to lam^K (the paper uses 0.5).
        """
        if lam <= 0:
            raise ValueError("lam must be strictly positive.")
        log_leaf = gaussian_leaf_logweights(X, scales, sq_dists)
        log_root = np.log(lam) + cauchy_root_logdensity(X, rho=rho, center=center)
        return cls(log_leaf, log_root)

    @property
    def n(self) -> int:
        return self.log_root.shape[0]

    def _shift(self) -> float:
        """A common log-scale used to keep exp() in range; it cancels in every ratio."""
        finite = self.log_leaf[np.isfinite(self.log_leaf)]
        top = max(float(np.max(self.log_root)), float(np.max(finite)) if finite.size else -np.inf)
        return top

    def augmented_log_weights(self) -> np.ndarray:
        """(n+1, n+1) log-weight matrix with the auxiliary node at index n."""
        n = self.n
        W = np.full((n + 1, n + 1), -np.inf)
        W[:n, :n] = self.log_leaf
        W[:n, n] = self.log_root
        W[n, :n] = self.log_root
        return W

    def grounded_laplacian(self) -> Tuple[np.ndarray, float]:
        """Laplacian of the augmented graph with node 0 removed, scaled by exp(-shift).

        L_g = diag(sum_j A_ij + A_0i) - A. It is symmetric positive definite whenever
        every root weight is positive. Returns (L_g, shift).
        """
        c = self._shift()
        A = np.exp(self.log_leaf - c)
        a = np.exp(self.log_root - c)
        L = np.diag(A.sum(axis=1) + a) - A
        return L, c

    def edge_inclusion(self) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior probability that each edge is in the augmented tree, in closed form.

        For a weighted spanning-tree law, Pr[e in T] = w_e * R_eff(e) (Kirchhoff). With
        G = L_g^{-1}:  M_ij = A_ij (G_ii + G_jj - 2 G_ij)  and  M_0i = A_0i G_ii.
        M_0i is the probability that point i is a cluster root, so sum_i M_0i = E[K | y].
        """
        L, c = self.grounded_laplacian()
        G = cho_solve(cho_factor(L, lower=True), np.eye(self.n))
        g = np.diag(G)
        A = np.exp(self.log_leaf - c)
        a = np.exp(self.log_root - c)
        R = g[:, None] + g[None, :] - 2.0 * G
        M = A * R
        np.fill_diagonal(M, 0.0)
        return M, a * g

    def expected_n_clusters(self) -> float:
        """E[K | y], the sum of the root-inclusion probabilities."""
        return float(np.sum(self.edge_inclusion()[1]))

    def log_tree_weight_sum(self, nodes: np.ndarray) -> float:
        """log tau(V) = log sum over spanning trees of V of prod A_ij (matrix-tree theorem)."""
        nodes = np.asarray(nodes, dtype=np.int64)
        m = nodes.size
        if m == 0:
            raise ValueError("a block must contain at least one node.")
        if m == 1:
            return 0.0
        sub = self.log_leaf[np.ix_(nodes, nodes)]
        finite = sub[np.isfinite(sub)]
        if finite.size == 0:
            return -np.inf
        c = float(np.max(finite))
        A = np.exp(sub - c)
        L = np.diag(A.sum(axis=1)) - A
        sign, logdet = np.linalg.slogdet(L[1:, 1:])
        if sign <= 0:
            return -np.inf
        return float(logdet + (m - 1) * c)

    def log_marginal(self, partition: np.ndarray) -> float:
        """Unnormalised log Pi(V | y): trees and roots summed out exactly.

        log Pi(V | y) = sum_k [ log sum_{i in V_k} A_0i + log tau(V_k) ] + const,
        where the lambda^K prior is already inside A_0i.
        """
        partition = np.asarray(partition)
        if partition.shape != (self.n,):
            raise ValueError("partition must assign one label to every point.")
        total = 0.0
        for label in np.unique(partition):
            nodes = np.flatnonzero(partition == label)
            total += float(logsumexp(self.log_root[nodes])) + self.log_tree_weight_sum(nodes)
        return total
