"""
Wilson's algorithm: exact weighted spanning trees and rooted forests
====================================================================
Wilson (1996) draws a spanning tree T with probability proportional to
prod_{e in T} w_e by running loop-erased random walks until they hit the growing
tree. There is no burn-in and no mixing question: every call returns an exact,
independent draw.

`sample_forest` applies it to the augmented graph of a Bayesian spanning forest
(Duan & Roy, JASA 2024), rooting the walks at the auxiliary node, so each call is an
exact posterior draw of (K, partition, trees) for a fixed kernel. The paper itself
uses the Aldous-Broder covering walk, which targets the same law; the fast-forwarded
cover algorithm of Tam, Dunson & Duan (Biometrika 2025) is a third exact sampler and
is not implemented here.
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.special import logsumexp

from bgforest.core.forest_model import ForestKernel
from bgforest.core.graph import validate_similarity_matrix

RandomLike = Optional[Union[int, np.random.RandomState, np.random.Generator]]


def _as_generator(random_state: RandomLike) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, np.random.RandomState):
        return np.random.Generator(random_state._bit_generator)
    return np.random.default_rng(random_state)


def _cumulative_from_log(log_W: np.ndarray) -> np.ndarray:
    """Row-wise cumulative transition probabilities of the walk P = W / deg."""
    log_deg = logsumexp(log_W, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        P = np.exp(log_W - log_deg)
    P = np.nan_to_num(P, nan=0.0)
    cum = np.cumsum(P, axis=1)
    cum[:, -1] = np.maximum(cum[:, -1], 1.0)  # guard the last bin against round-off
    return cum


def wilson_trees(
    cum: np.ndarray, roots: np.ndarray, rng: np.random.Generator, n_draws: int = 1
) -> np.ndarray:
    """Core loop of Wilson's algorithm, repeated for independent draws.

    Parameters
    ----------
    cum : (N, N) array
        Row-wise cumulative transition probabilities of the walk.
    roots : (N,) bool array
        Nodes that start in the tree.
    rng : numpy Generator
    n_draws : int

    Returns
    -------
    parents : (n_draws, N) int array
        parents[d, v] is the next node on v's path to the roots, -1 for a root.
    """
    N = cum.shape[0]
    rows = [cum[u] for u in range(N)]  # row views: cheaper to index in the hot loop
    out = np.full((n_draws, N), -1, dtype=np.int64)
    buffer = rng.random(8192).tolist()
    pos = 0
    for d in range(n_draws):
        in_tree = roots.tolist()
        nxt = [-1] * N
        parent = out[d]
        for start in rng.permutation(N).tolist():
            u = start
            while not in_tree[u]:
                if pos == len(buffer):
                    buffer = rng.random(8192).tolist()
                    pos = 0
                v = int(np.searchsorted(rows[u], buffer[pos], side="right"))
                pos += 1
                v = v if v < N else N - 1
                nxt[u] = v
                u = v
            u = start
            while not in_tree[u]:
                in_tree[u] = True
                parent[u] = nxt[u]
                u = nxt[u]
    return out


def wilson_tree(cum: np.ndarray, in_tree: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw of Wilson's algorithm; ``in_tree`` marks the starting roots."""
    return wilson_trees(cum, np.asarray(in_tree, dtype=bool), rng, 1)[0]


def forest_partition(parent: np.ndarray) -> np.ndarray:
    """Component labels of a rooted forest given parent pointers (-1 marks a root).

    Labels are 0..K-1 in order of first appearance, so equal partitions get equal arrays.
    """
    n = parent.shape[0]
    root_of = np.full(n, -1, dtype=np.int64)
    for v in range(n):
        path = []
        u = v
        while root_of[u] < 0 and parent[u] >= 0:
            path.append(u)
            u = parent[u]
        r = root_of[u] if root_of[u] >= 0 else u
        root_of[u] = r
        for w in path:
            root_of[w] = r
    _, labels = np.unique(root_of, return_inverse=True)
    first = {}
    out = np.empty(n, dtype=np.int64)
    for i, lab in enumerate(labels):
        out[i] = first.setdefault(int(lab), len(first))
    return out


class WilsonSampler:
    """Exact sampler for weighted spanning trees, rooted forests and forest posteriors."""

    def __init__(self, random_state: RandomLike = None):
        self.rng = _as_generator(random_state)

    def sample_spanning_tree(self, W: np.ndarray, root: Optional[int] = None) -> List[Tuple[int, int]]:
        """Draw T with Pr(T) proportional to prod_{e in T} W_e on a connected graph.

        Returns the n - 1 edges as (child, parent) pairs directed toward ``root``.
        """
        W = validate_similarity_matrix(W, require_connected=True)
        n = W.shape[0]
        if n <= 1:
            return []
        if root is not None and not 0 <= root < n:
            raise IndexError("root must be a valid node index.")
        r = int(self.rng.integers(n)) if root is None else int(root)
        with np.errstate(divide="ignore"):
            cum = _cumulative_from_log(np.log(W))
        in_tree = np.zeros(n, dtype=bool)
        in_tree[r] = True
        parent = wilson_tree(cum, in_tree, self.rng)
        return [(v, int(parent[v])) for v in range(n) if parent[v] >= 0]

    def sample_spanning_forest(
        self, W: np.ndarray, roots: Sequence[int]
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Draw a forest with the given roots, Pr(F) proportional to prod_{e in F} W_e.

        Returns (edges, labels) where labels[v] is the index (in sorted ``roots``) of
        the tree containing v.
        """
        W = validate_similarity_matrix(W)
        n = W.shape[0]
        roots = sorted(set(int(r) for r in roots))
        if not roots:
            raise ValueError("roots must contain at least one node.")
        if any(not 0 <= r < n for r in roots):
            raise IndexError("roots must contain only valid node indices.")
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        _, comp = connected_components(csr_matrix(W > 0), directed=False)
        if len({comp[r] for r in roots}) != len(set(comp)):
            raise ValueError("roots must include at least one node from every connected component of W.")
        with np.errstate(divide="ignore"):
            cum = _cumulative_from_log(np.log(W))
        in_tree = np.zeros(n, dtype=bool)
        in_tree[roots] = True
        parent = wilson_tree(cum, in_tree, self.rng)
        edges = [(v, int(parent[v])) for v in range(n) if parent[v] >= 0]
        root_index = {r: k for k, r in enumerate(roots)}
        labels = np.empty(n, dtype=np.int64)
        for v in range(n):
            u = v
            while parent[u] >= 0:
                u = parent[u]
            labels[v] = root_index[u]
        return edges, labels

    def sample_forests(self, kernel: ForestKernel, n_draws: int) -> np.ndarray:
        """Independent exact draws from the Bayesian spanning forest posterior (fixed kernel).

        Returns an (n_draws, n) array of parent pointers over the data points:
        parent[i] = j means the edge (i, j) is in the forest, -1 means i is a root.
        """
        n = kernel.n
        cum = _cumulative_from_log(kernel.augmented_log_weights())
        roots = np.zeros(n + 1, dtype=bool)
        roots[n] = True
        parents = wilson_trees(cum, roots, self.rng, int(n_draws))[:, :n]
        parents[parents == n] = -1
        return parents

    def sample_forest(self, kernel: ForestKernel) -> np.ndarray:
        """One exact posterior draw; see :meth:`sample_forests`."""
        return self.sample_forests(kernel, 1)[0]


# Backwards-compatible name.
WilsonLERWSampler = WilsonSampler
