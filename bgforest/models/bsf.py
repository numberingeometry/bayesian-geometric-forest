"""
Bayesian spanning forest clustering (Duan & Roy, JASA 2024)
===========================================================

Each cluster is a tree: a root drawn from a diffuse Cauchy density, every other point
drawn from a Gaussian centred on its parent. The trees are latent and are summed
over; the partition, the number of clusters and their uncertainty come out of the
posterior.

Two ways to set the Gaussian leaf scales sigma_ij = s_i s_j:

* ``scales="adaptive"`` (default): the paper's hierarchical inverse-gamma prior on
  s_i and its Gibbs sampler (Sections 2.4 and 3.1). Each sweep draws the whole forest
  exactly given the scales, then updates the scales given the forest.
* ``scales="local"`` or an array: the scales are fixed (``"local"`` uses the distance
  to the ``n_neighbors``-th neighbour). The posterior over forests is then a weighted
  spanning-tree law, and every draw is exact and independent: no burn-in, no mixing
  diagnostics. Edge-inclusion probabilities are also available in closed form.
"""

from typing import Optional, Union

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from bgforest.core.forest_model import ForestKernel, local_scales
from bgforest.samplers.wilson import RandomLike, WilsonSampler, _as_generator, forest_partition

# Hyper-prior constants of Duan & Roy (2024), Section 2.4.
A_SIGMA, XI_SIGMA, B_SIGMA = 100.0, 1.0, 10.0
ROOT_SHAPE, ROOT_RATE = 2.0, 1.0


def _inv_gamma(rng: np.random.Generator, shape, rate):
    """Draw from Inverse-Gamma(shape, rate): the reciprocal of Gamma(shape, rate)."""
    shape, rate = np.broadcast_arrays(np.asarray(shape, dtype=np.float64), np.asarray(rate, dtype=np.float64))
    return rate / rng.gamma(shape)


def _monte_carlo_inclusion(forests: np.ndarray):
    """Edge- and root-inclusion frequencies over drawn forests (parent pointers)."""
    n_draws, n = forests.shape
    E = np.zeros((n, n))
    roots = np.zeros(n)
    for parent in forests:
        child = np.flatnonzero(parent >= 0)
        np.add.at(E, (child, parent[child]), 1.0)
        roots += parent < 0
    return (E + E.T) / n_draws, roots / n_draws


def _depth_parity(parent: np.ndarray) -> np.ndarray:
    """0/1 depth parity of each node in a rooted forest (roots have depth 0)."""
    n = parent.shape[0]
    parity = np.full(n, -1, dtype=np.int64)
    for v in range(n):
        path = []
        u = v
        while parity[u] < 0 and parent[u] >= 0:
            path.append(u)
            u = parent[u]
        if parity[u] < 0:
            parity[u] = 0
        p = parity[u]
        for w in reversed(path):
            p ^= 1
            parity[w] = p
    return parity


class BayesianSpanningForest(BaseEstimator, ClusterMixin):
    """Bayesian spanning forest clustering with posterior uncertainty.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters in the point estimate ``labels_``. The posterior itself is
        over all K; by default the point estimate uses the posterior mode of K.
    scales : {"adaptive", "local"} or array of shape (n_samples,), default="adaptive"
        How the leaf scales s_i (sigma_ij = s_i s_j) are set; see the module docstring.
    n_neighbors : int, default=7
        Neighbour rank for the local scales, and for initialising adaptive scales.
    lam : float, default=0.5
        Prior weight per cluster, Pi(T) proportional to lam^K (the paper's value).
    root_scale : float, default=1.0
        Cauchy root scale for fixed-scale fits. In adaptive mode it is sampled under
        the paper's Inverse-Gamma(2, 1) prior on its square and this is the start value.
    n_draws : int, default=200
        Number of retained posterior draws.
    burn_in : int, default=200
        Discarded sweeps in adaptive mode (ignored for fixed scales: draws are exact).
    standardize : bool, default=True
        Standardise features first, as the paper assumes.
    random_state : int, numpy Generator or RandomState, optional

    Attributes
    ----------
    labels_ : (n_samples,) array
        Point estimate: rank-K eigen-factorisation of the co-clustering matrix, then
        k-means (the paper uses a symmetric non-negative factorisation here).
    partitions_ : (n_draws, n_samples) array
        Posterior draws of the partition.
    forests_ : (n_draws, n_samples) array
        Parent pointers of each drawn forest (-1 marks a root).
    n_clusters_draws_ : (n_draws,) array
        Number of clusters in each draw.
    co_clustering_matrix_ : (n_samples, n_samples) array
        Pr(i and j in the same cluster | data), estimated from the draws.
    edge_inclusion_ : (n_samples, n_samples) array
        Pr(edge (i, j) in the forest | data): closed form for fixed scales, Monte Carlo
        for adaptive scales.
    root_probability_ : (n_samples,) array
        Pr(i is a cluster root | data); its sum is E[K | data].
    kernel_ : ForestKernel
        The fixed kernel, or the posterior-mean-scale kernel in adaptive mode.
    scales_ : (n_samples,) array
        Leaf scales used (posterior mean in adaptive mode).
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        scales: Union[str, np.ndarray] = "adaptive",
        n_neighbors: int = 7,
        lam: float = 0.5,
        root_scale: float = 1.0,
        n_draws: int = 200,
        burn_in: int = 200,
        standardize: bool = True,
        random_state: RandomLike = None,
    ):
        self.n_clusters = n_clusters
        self.scales = scales
        self.n_neighbors = n_neighbors
        self.lam = lam
        self.root_scale = root_scale
        self.n_draws = n_draws
        self.burn_in = burn_in
        self.standardize = standardize
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting
    def fit(self, X: np.ndarray, y=None) -> "BayesianSpanningForest":
        X = check_array(X, dtype=np.float64, ensure_min_samples=2)
        n = X.shape[0]
        if self.n_clusters is not None and not 1 <= self.n_clusters <= n:
            raise ValueError("n_clusters must be between 1 and n_samples.")
        if self.n_draws < 1 or self.burn_in < 0:
            raise ValueError("n_draws must be positive and burn_in non-negative.")
        if self.lam <= 0 or self.root_scale <= 0:
            raise ValueError("lam and root_scale must be strictly positive.")
        self.scaler_ = StandardScaler().fit(X) if self.standardize else None
        Z = self.scaler_.transform(X) if self.standardize else X
        self.X_ = Z
        rng = _as_generator(self.random_state)
        sampler = WilsonSampler(rng)

        if isinstance(self.scales, str) and self.scales == "adaptive":
            self._fit_adaptive(Z, sampler, rng)
        else:
            if isinstance(self.scales, str):
                if self.scales != "local":
                    raise ValueError('scales must be "adaptive", "local" or an array.')
                s = local_scales(Z, self.n_neighbors)
            else:
                s = np.asarray(self.scales, dtype=np.float64)
                if s.shape != (n,):
                    raise ValueError("a scales array must have one entry per sample.")
            self.scales_ = s
            self.root_scale_ = float(self.root_scale)
            self.kernel_ = ForestKernel.from_data(Z, s, rho=self.root_scale, lam=self.lam)
            self.forests_ = self._draw_forests(self.kernel_, sampler, self.n_draws)
            if self._is_constrained():
                self.edge_inclusion_, self.root_probability_ = _monte_carlo_inclusion(self.forests_)
            else:
                self.edge_inclusion_, self.root_probability_ = self.kernel_.edge_inclusion()

        self.partitions_ = np.array([forest_partition(p) for p in self.forests_])
        self.n_clusters_draws_ = self.partitions_.max(axis=1) + 1
        P = np.zeros((n, n))
        for lab in self.partitions_:
            P += lab[:, None] == lab[None, :]
        self.co_clustering_matrix_ = P / len(self.partitions_)
        self._point_estimate(rng)
        return self

    def _fit_adaptive(self, Z: np.ndarray, sampler: WilsonSampler, rng: np.random.Generator) -> None:
        """Gibbs sampler of Duan & Roy (2024), Section 3.1, with an exact forest step."""
        n, p = Z.shape
        sq = cdist(Z, Z, metric="sqeuclidean")
        norm_sq = np.sum(Z**2, axis=1)  # the root centre is fixed at 0 (standardised data)
        s = local_scales(Z, self.n_neighbors)
        beta = (B_SIGMA - 1.0) * float(np.mean(s))
        eta = beta
        gamma = float(self.root_scale)

        n_sweeps = self.burn_in + self.n_draws
        forests = np.empty((self.n_draws, n), dtype=np.int64)
        scale_sum = np.zeros(n)
        self.n_clusters_trace_ = np.empty(n_sweeps, dtype=np.int64)
        self.root_scale_trace_ = np.empty(n_sweeps)
        self.mean_scale_trace_ = np.empty(n_sweeps)

        for t in range(n_sweeps):
            kernel = ForestKernel.from_data(Z, s, rho=gamma, lam=self.lam, sq_dists=sq)
            parent = self._draw_forests(kernel, sampler, 1)[0]
            roots = parent < 0
            K = int(roots.sum())

            # Root scale: Cauchy as a scale mixture of normals over u ~ IG(1/2, 1/2).
            u = _inv_gamma(rng, (1.0 + p) / 2.0, 0.5 + norm_sq[roots] / (2.0 * gamma**2))
            gamma = float(np.sqrt(_inv_gamma(
                rng, ROOT_SHAPE + K * p / 2.0, ROOT_RATE + np.sum(norm_sq[roots] / (2.0 * u))
            )))

            # Leaf scales: eta | beta, beta | s, then s_i | forest in two conditionally
            # independent blocks (a forest is bipartite by depth parity).
            eta = float(_inv_gamma(rng, 1.0 + A_SIGMA, beta + XI_SIGMA))
            beta = float(rng.gamma(1.0 + n * B_SIGMA, 1.0 / (np.sum(1.0 / s) + 1.0 / eta)))
            child = np.flatnonzero(~roots)
            par = parent[child]
            degree = np.bincount(child, minlength=n) + np.bincount(par, minlength=n)
            parity = _depth_parity(parent)
            for colour in (0, 1):
                load = np.zeros(n)
                np.add.at(load, child, sq[child, par] / (2.0 * s[par]))
                np.add.at(load, par, sq[child, par] / (2.0 * s[child]))
                block = parity == colour
                s_new = _inv_gamma(rng, p * degree[block] / 2.0 + B_SIGMA, load[block] + beta)
                s[block] = np.maximum(s_new, 1e-12)

            self.n_clusters_trace_[t] = K
            self.root_scale_trace_[t] = gamma
            self.mean_scale_trace_[t] = float(np.mean(s))
            if t >= self.burn_in:
                d = t - self.burn_in
                forests[d] = parent
                scale_sum += s

        self.forests_ = forests
        self.edge_inclusion_, self.root_probability_ = _monte_carlo_inclusion(forests)
        self.scales_ = scale_sum / self.n_draws
        self.root_scale_ = float(np.mean(self.root_scale_trace_[self.burn_in:]))
        self.kernel_ = ForestKernel.from_data(Z, self.scales_, rho=self.root_scale_, lam=self.lam, sq_dists=sq)

    def _is_constrained(self) -> bool:
        return False

    def _draw_forests(self, kernel: ForestKernel, sampler: WilsonSampler, k: int) -> np.ndarray:
        """k exact draws of the forest given the kernel (subclasses may condition them)."""
        return sampler.sample_forests(kernel, k)

    def _point_estimate(self, rng: np.random.Generator) -> None:
        values, counts = np.unique(self.n_clusters_draws_, return_counts=True)
        self.n_clusters_posterior_ = dict(zip(values.tolist(), (counts / counts.sum()).tolist()))
        k = int(self.n_clusters) if self.n_clusters is not None else int(values[np.argmax(counts)])
        self.n_clusters_ = k
        evals, evecs = np.linalg.eigh(self.co_clustering_matrix_)
        order = np.argsort(evals)[::-1][:k]
        self.spectral_eigenvalues_ = evals[order]
        self.spectral_eigenvectors_ = evecs[:, order]
        if k == 1:
            self.labels_ = np.zeros(self.X_.shape[0], dtype=np.int64)
            return
        seed = int(rng.integers(2**31 - 1))
        self.labels_ = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
            self.spectral_eigenvectors_
        )

    # --------------------------------------------------------------- summaries
    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X).labels_

    def predict_proba(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Soft membership of each fitted point in each point-estimate cluster.

        Entry (i, c) is the average co-clustering probability between i and the
        members of cluster c, normalised over c. Only the fitted points are supported.
        """
        check_is_fitted(self, ["labels_", "co_clustering_matrix_"])
        if X is not None and np.asarray(X).shape[0] != self.labels_.shape[0]:
            raise ValueError("predict_proba supports the fitted observations only.")
        k = self.n_clusters_
        proba = np.zeros((self.labels_.shape[0], k))
        for c in range(k):
            members = self.labels_ == c
            if np.any(members):
                proba[:, c] = self.co_clustering_matrix_[:, members].mean(axis=1)
        return proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)

    def assignment_uncertainty(self) -> np.ndarray:
        """Posterior probability that each point is NOT with its point-estimate cluster.

        1 minus the average co-clustering probability with the other members of its
        cluster: 0 for points the posterior never separates from their cluster.
        """
        check_is_fitted(self, ["labels_", "co_clustering_matrix_"])
        out = np.empty(self.labels_.shape[0])
        for i, c in enumerate(self.labels_):
            members = np.flatnonzero(self.labels_ == c)
            members = members[members != i]
            out[i] = 0.0 if members.size == 0 else 1.0 - self.co_clustering_matrix_[i, members].mean()
        return out
