"""
Tempered spanning trees on a point cloud
========================================

For points x_1, ..., x_n and an inverse temperature lam > 0, the tree law is

    Pr_lam(T)  is proportional to  exp(-lam * sum_{(i,j) in T} |x_i - x_j|^2)

over the spanning trees T of the complete graph. It is the posterior over trees
when the cloud was grown by Gaussian diffusion along T, with step variance
sigma^2 = 1 / (2 lam) and a flat prior over trees (see ``tree_diffusion``).

* lam -> 0: the uniform spanning tree (every labelled tree equally likely).
* lam -> infinity: the minimum spanning tree, whose cuts are single linkage.

Edge probabilities follow Kirchhoff: pi_e = w_e * R_eff(e). They are computed by a
Kron (Schur) elimination carried out entirely in log space with positive-only
updates, so the law stays exact at temperatures where a floating-point Laplacian
inverse has no significant digits.
"""

from typing import Tuple

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.special import logsumexp

from bgforest.samplers.wilson import RandomLike, _as_generator, _cumulative_from_log, wilson_trees

LOG_HALF = float(np.log(0.5))


def squared_distances(X: np.ndarray) -> np.ndarray:
    """Matrix of squared Euclidean distances, exactly symmetric with a zero diagonal."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be an (n_points, n_features) array.")
    diff = X[:, None, :] - X[None, :, :]
    C = np.einsum("ijk,ijk->ij", diff, diff)
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 0.0)
    return C


def kron_log_resistance(log_w: np.ndarray) -> Tuple[float, np.ndarray]:
    """Log spanning-tree weight and log effective resistances of a weighted graph.

    Eliminates vertices 0, ..., n-2 one at a time. Each step is a Kron reduction,
    ``w'_ab = w_ab + w_ak w_kb / d_k`` with ``d_k = sum_a w_ak``, which only adds
    positive numbers and is evaluated with ``logaddexp``. The product of the pivots
    ``d_k`` is the weighted number of spanning trees (matrix-tree theorem). Effective
    resistances are restored in reverse order from

        R(k, j) = 1/d_k + sum_a p_a R(a, j) - (1/2) p' R p,      p_a = w_ak / d_k,

    which is the energy of the unit flow from k to j after k is eliminated; it is at
    least 1/d_k, so the one subtraction cannot cancel catastrophically.

    Parameters
    ----------
    log_w : (n, n) array
        Symmetric log edge weights; ``-inf`` marks a missing edge. The diagonal is
        ignored.

    Returns
    -------
    log_tau : float
        log of sum over spanning trees T of prod_{e in T} w_e.
    log_R : (n, n) array
        log effective resistances; the diagonal is ``-inf``.
    """
    lw = np.array(log_w, dtype=np.float64, copy=True)
    if lw.ndim != 2 or lw.shape[0] != lw.shape[1]:
        raise ValueError("log_w must be a square matrix.")
    if np.isnan(lw).any() or np.isposinf(lw).any():
        raise ValueError("log_w must not contain NaN or +inf.")
    n = lw.shape[0]
    log_R = np.full((n, n), -np.inf)
    if n <= 1:
        return 0.0, log_R
    lw = np.maximum(lw, lw.T)  # tolerate round-off asymmetry; exact inputs are unchanged
    np.fill_diagonal(lw, -np.inf)

    rows = []
    pivots = np.empty(n - 1)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for k in range(n - 1):
            r = lw[k, k + 1 :].copy()
            d = float(logsumexp(r))
            if not np.isfinite(d):
                raise ValueError("the graph is disconnected; its tree law is undefined.")
            rows.append(r)
            pivots[k] = d
            rest = lw[k + 1 :, k + 1 :]
            np.logaddexp(rest, r[:, None] + r[None, :] - d, out=rest)
            idx = np.arange(rest.shape[0])
            rest[idx, idx] = -np.inf

        for k in range(n - 2, -1, -1):
            lp = rows[k] - pivots[k]
            rest = log_R[k + 1 :, k + 1 :]
            lq = logsumexp(lp[:, None] + rest, axis=0)  # log sum_a p_a R(a, j)
            lpRp = logsumexp(lp + lq)  # log p' R p
            head = np.logaddexp(-pivots[k], lq)  # log (1/d_k + sum_a p_a R(a, j))
            ratio = np.minimum(LOG_HALF + lpRp - head, -1e-300)
            col = head + np.log1p(-np.exp(ratio))
            log_R[k, k + 1 :] = col
            log_R[k + 1 :, k] = col
    return float(pivots.sum()), log_R


def mst_edges(C: np.ndarray) -> np.ndarray:
    """Edges (i < j) of a minimum spanning tree of the complete graph with costs C."""
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    cost = C.copy()
    off = ~np.eye(n, dtype=bool)
    cost[(cost == 0) & off] = np.finfo(float).tiny  # csgraph reads 0 as "no edge"
    tree = minimum_spanning_tree(cost).tocoo()
    edges = np.sort(np.c_[tree.row, tree.col], axis=1)
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))]


class TreeLaw:
    """The tempered tree law of a point cloud, exact at any temperature.

    ``Pr(T)`` is proportional to ``exp(-lam * sum_{e in T} c_e)`` with ``c_e`` the
    squared length of edge e. Build it from points (``TreeLaw(X, lam)``) or from any
    symmetric cost matrix (``TreeLaw.from_costs(C, lam)``).

    Parameters
    ----------
    X : (n, p) array
        Points.
    lam : float
        Inverse temperature, ``1 / (2 sigma^2)`` for diffusion steps of variance sigma^2.
    """

    def __init__(self, X: np.ndarray, lam: float):
        self.X = np.asarray(X, dtype=np.float64)
        self._init(squared_distances(self.X), lam)

    @classmethod
    def from_costs(cls, C: np.ndarray, lam: float) -> "TreeLaw":
        """Tree law with arbitrary symmetric nonnegative edge costs ``C`` (``inf`` = no edge)."""
        law = cls.__new__(cls)
        law.X = None
        C = np.asarray(C, dtype=np.float64)
        if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.allclose(C, C.T, equal_nan=False):
            raise ValueError("C must be a symmetric square matrix.")
        law._init(C, lam)
        return law

    def _init(self, C: np.ndarray, lam: float) -> None:
        lam = float(lam)
        if not lam >= 0 or not np.isfinite(lam):
            raise ValueError("lam must be a finite nonnegative number.")
        self.costs = C
        self.lam = lam
        self.n = C.shape[0]
        with np.errstate(invalid="ignore"):
            self.log_weights = -lam * C
        self.log_weights[np.isinf(C)] = -np.inf
        np.fill_diagonal(self.log_weights, -np.inf)
        self._solved = None

    def _solve(self):
        if self._solved is None:
            self._solved = kron_log_resistance(self.log_weights)
        return self._solved

    # ------------------------------------------------------------------ exact
    def log_partition(self) -> float:
        """log Z, with Z = sum_T exp(-lam * S_T) (matrix-tree theorem)."""
        return self._solve()[0]

    def log_resistance(self) -> np.ndarray:
        """log effective resistances of the network with conductances exp(-lam c_e)."""
        return self._solve()[1]

    def edge_probability(self) -> np.ndarray:
        """Pr(e in T) for every pair, as an (n, n) matrix: pi_e = w_e R_eff(e) (Kirchhoff)."""
        with np.errstate(invalid="ignore"):
            P = np.exp(self.log_weights + self.log_resistance())
        P = np.nan_to_num(P, nan=0.0)
        np.fill_diagonal(P, 0.0)
        return P

    def trace_error(self) -> float:
        """|sum_e pi_e - (n - 1)| / (n - 1): every tree has n - 1 edges, so this gates accuracy."""
        if self.n <= 1:
            return 0.0
        return float(abs(np.triu(self.edge_probability(), 1).sum() - (self.n - 1)) / (self.n - 1))

    def expected_length(self) -> float:
        """E[S_T] = sum_e pi_e c_e, the posterior mean of the total squared edge length."""
        P = np.triu(self.edge_probability(), 1)
        mask = P > 0
        return float(np.sum(P[mask] * self.costs[mask]))

    def free_energy(self) -> float:
        """F = -log(Z) / lam: a smooth minimum spanning tree length.

        ``0 <= S_MST - F <= (n - 2) log(n) / lam`` on the complete graph.
        """
        if self.lam == 0:
            raise ValueError("the free energy needs lam > 0.")
        return -self.log_partition() / self.lam

    def free_energy_grad(self) -> np.ndarray:
        """Gradient of F with respect to the points: 2 L_pi X, with L_pi the Laplacian of pi."""
        if self.X is None:
            raise ValueError("the gradient in the points needs a TreeLaw built from points.")
        P = self.edge_probability()
        L = np.diag(P.sum(axis=1)) - P
        return 2.0 * L @ self.X

    # -------------------------------------------------------------- zero temperature
    def mst(self) -> np.ndarray:
        """Edges of a minimum spanning tree: the lam -> infinity limit of the law."""
        return mst_edges(self.costs)

    def mst_overlap(self) -> float:
        """Expected fraction of tree edges that belong to the minimum spanning tree."""
        e = self.mst()
        return float(self.edge_probability()[e[:, 0], e[:, 1]].sum() / max(self.n - 1, 1))

    # ------------------------------------------------------------------ sampling
    def sample(self, n_trees: int = 1, random_state: RandomLike = None, method: str = "wilson") -> np.ndarray:
        """Exact independent trees, as an (n_trees, n) parent array (-1 at the root).

        ``method="wilson"`` runs Wilson's loop-erased random walks: fast when hot, but at
        low temperature a walk can be trapped for about exp(lam * gap) steps between
        nearly isolated points. ``method="chain"`` uses :func:`chain_rule_tree`, whose cost
        does not depend on the temperature.
        """
        rng = _as_generator(random_state)
        out = np.empty((int(n_trees), self.n), dtype=np.int64)
        if method == "chain":
            from bgforest.ultrametric import reroot

            for t in range(int(n_trees)):
                out[t] = reroot(chain_rule_tree(self.log_weights, rng), self.n, int(rng.integers(self.n)))
            return out
        if method != "wilson":
            raise ValueError("method must be 'wilson' or 'chain'.")
        cum = _cumulative_from_log(self.log_weights)
        for t in range(int(n_trees)):
            roots = np.zeros(self.n, dtype=bool)
            roots[int(rng.integers(self.n))] = True
            out[t] = wilson_trees(cum, roots, rng, 1)[0]
        return out


def chain_rule_tree(log_w: np.ndarray, random_state: RandomLike = None) -> list:
    """One exact draw from the weighted spanning-tree law, at any temperature.

    The chain rule with exact marginals (in the manner of Kulkarni, 1990): keep groups of
    contracted vertices and the log of the total weight between groups. Repeatedly take
    the pair of groups most likely to be joined in the tree, with probability
    w_AB R_eff(A, B) from the log-space Kron solver. With that probability join them,
    choosing the original edge in proportion to its weight; otherwise delete every edge
    between them. Each step conditions the law exactly. Costs at most about 2n solves.

    Returns the n - 1 edges as (i, j) pairs.
    """
    rng = _as_generator(random_state)
    log_w = np.asarray(log_w, dtype=np.float64)
    n = log_w.shape[0]
    alive = np.maximum(log_w, log_w.T)
    np.fill_diagonal(alive, -np.inf)
    L = alive.copy()
    groups = [[i] for i in range(n)]
    edges = []
    while len(groups) > 1:
        _, log_R = kron_log_resistance(L)
        with np.errstate(invalid="ignore"):
            log_pi = L + log_R
        np.fill_diagonal(log_pi, -np.inf)
        a, b = np.unravel_index(int(np.argmax(log_pi)), log_pi.shape)
        a, b = int(min(a, b)), int(max(a, b))
        if rng.random() < np.exp(min(float(log_pi[a, b]), 0.0)):
            sub = alive[np.ix_(groups[a], groups[b])]
            p = np.exp(sub - logsumexp(sub)).ravel()
            k = int(rng.choice(len(p), p=p / p.sum()))
            edges.append((groups[a][k // len(groups[b])], groups[b][k % len(groups[b])]))
            merged = np.logaddexp(L[a], L[b])
            L[a], L[:, a] = merged, merged
            L[a, a] = -np.inf
            L = np.delete(np.delete(L, b, 0), b, 1)
            groups[a] = groups[a] + groups[b]
            del groups[b]
        else:
            L[a, b] = L[b, a] = -np.inf
            alive[np.ix_(groups[a], groups[b])] = -np.inf
            alive[np.ix_(groups[b], groups[a])] = -np.inf
    return edges


def edge_set(parent: np.ndarray) -> set:
    """Undirected edges {(min, max)} of a tree given as a parent array (-1 at the root)."""
    return {(min(v, int(p)), max(v, int(p))) for v, p in enumerate(parent) if p >= 0}


def edge_overlap(parent_a: np.ndarray, parent_b: np.ndarray) -> float:
    """Fraction of edges two spanning trees share: q = |T_a & T_b| / (n - 1)."""
    a, b = edge_set(parent_a), edge_set(parent_b)
    return len(a & b) / max(len(a), 1)


def lam_from_scale(C: np.ndarray, kappa: float) -> float:
    """lam = kappa / (mean squared MST edge length): a dimensionless temperature dial."""
    e = mst_edges(C)
    scale = float(np.mean(np.asarray(C)[e[:, 0], e[:, 1]]))
    if scale <= 0:
        raise ValueError("the minimum spanning tree has zero length.")
    return float(kappa) / scale


__all__ = [
    "TreeLaw",
    "chain_rule_tree",
    "edge_overlap",
    "edge_set",
    "kron_log_resistance",
    "lam_from_scale",
    "mst_edges",
    "squared_distances",
]

