"""
Brute-force enumeration for small graphs
========================================
Ground truth for the verification suite. Every quantity here is computed by listing
objects one by one (all labelled trees via Pruefer sequences, all set partitions),
with no matrix-tree theorem and no sampling, so it can check both.

Only practical for about n <= 8 nodes: there are N^(N-2) labelled trees on N nodes.
"""

import itertools
from typing import Dict, Iterator, List, Tuple

import numpy as np
from scipy.special import logsumexp

from bgforest.core.forest_model import ForestKernel

Edge = Tuple[int, int]


def prufer_decode(seq: Tuple[int, ...], N: int) -> List[Edge]:
    """The labelled tree on nodes 0..N-1 encoded by a Pruefer sequence of length N-2."""
    degree = [1] * N
    for x in seq:
        degree[x] += 1
    edges = []
    for x in seq:
        leaf = min(i for i in range(N) if degree[i] == 1)
        edges.append((leaf, x))
        degree[leaf] -= 1
        degree[x] -= 1
    u, v = [i for i in range(N) if degree[i] == 1]
    edges.append((u, v))
    return edges


def all_labelled_trees(N: int) -> Iterator[List[Edge]]:
    """Every spanning tree of the complete graph on N nodes (Cayley: N^(N-2) of them)."""
    if N == 1:
        yield []
        return
    if N == 2:
        yield [(0, 1)]
        return
    for seq in itertools.product(range(N), repeat=N - 2):
        yield prufer_decode(seq, N)


def all_set_partitions(n: int) -> Iterator[Tuple[int, ...]]:
    """Every partition of {0..n-1} as canonical labels (first-appearance order)."""

    def grow(prefix: List[int], k: int) -> Iterator[Tuple[int, ...]]:
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for label in range(k + 1):
            yield from grow(prefix + [label], max(k, label + 1))

    yield from grow([], 0)


def canonical(labels: np.ndarray) -> Tuple[int, ...]:
    first: Dict[int, int] = {}
    return tuple(first.setdefault(int(x), len(first)) for x in labels)


def log_tree_weight_sum_bruteforce(log_W: np.ndarray) -> float:
    """log of the sum over all spanning trees of prod of edge weights (no matrix-tree theorem)."""
    m = log_W.shape[0]
    if m == 1:
        return 0.0
    terms = [sum(log_W[i, j] for i, j in tree) for tree in all_labelled_trees(m)]
    return float(logsumexp(terms))


def forest_posterior_bruteforce(kernel: ForestKernel):
    """Exact posterior over partitions, edge inclusion and K by listing augmented trees.

    Returns
    -------
    partition_prob : dict mapping canonical partition -> probability
    M : (n, n) array of Pr[(i, j) in forest | y]
    root_prob : (n,) array of Pr[i is a root | y]
    """
    n = kernel.n
    W = kernel.augmented_log_weights()
    aux = n
    trees = list(all_labelled_trees(n + 1))
    log_w = np.array([sum(W[i, j] for i, j in t) for t in trees])
    prob = np.exp(log_w - logsumexp(log_w))

    partition_prob: Dict[Tuple[int, ...], float] = {}
    M = np.zeros((n, n))
    root_prob = np.zeros(n)
    for t, p in zip(trees, prob):
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in t:
            if i == aux or j == aux:
                root_prob[j if i == aux else i] += p
                continue
            M[i, j] += p
            M[j, i] += p
            parent[find(i)] = find(j)
        key = canonical(np.array([find(i) for i in range(n)]))
        partition_prob[key] = partition_prob.get(key, 0.0) + p
    return partition_prob, M, root_prob
