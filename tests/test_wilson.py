"""Wilson's algorithm draws exactly from the weighted spanning-tree law."""

import numpy as np
import pytest
from scipy.stats import chisquare

from bgforest.core.forest_model import ForestKernel, local_scales
from bgforest.exact import all_labelled_trees, canonical, forest_posterior_bruteforce
from bgforest.samplers.wilson import WilsonSampler, forest_partition


def _chi2_pvalue(observed: dict, expected: dict, n: int) -> float:
    keys = [k for k, p in expected.items() if p * n >= 5]
    obs = np.array([observed.get(k, 0) for k in keys] + [n - sum(observed.get(k, 0) for k in keys)])
    exp = np.array([expected[k] * n for k in keys] + [n * (1 - sum(expected[k] for k in keys))])
    keep = exp > 0
    return chisquare(obs[keep], exp[keep] * obs[keep].sum() / exp[keep].sum()).pvalue


def test_spanning_tree_law_is_exact_on_a_weighted_graph():
    W = np.array([[0, 1.0, 0.5, 2.0], [1.0, 0, 1.0, 0.2], [0.5, 1.0, 0, 1.0], [2.0, 0.2, 1.0, 0]])
    trees = list(all_labelled_trees(4))
    weights = {}
    for t in trees:
        w = np.prod([W[i, j] for i, j in t])
        if w > 0:
            weights[frozenset(frozenset(e) for e in t)] = w
    total = sum(weights.values())
    expected = {k: v / total for k, v in weights.items()}
    sampler = WilsonSampler(7)
    n = 20000
    observed = {}
    for _ in range(n):
        key = frozenset(frozenset(e) for e in sampler.sample_spanning_tree(W))
        observed[key] = observed.get(key, 0) + 1
    assert set(observed) <= set(expected)
    assert _chi2_pvalue(observed, expected, n) > 1e-3


def test_forest_posterior_draws_are_exact():
    rng = np.random.default_rng(3)
    X = np.vstack([rng.normal(0, 0.4, (3, 2)), rng.normal(1.5, 0.4, (2, 2))])
    kernel = ForestKernel.from_data(X, local_scales(X, 2), lam=0.5)
    expected, _, _ = forest_posterior_bruteforce(kernel)
    n = 20000
    parents = WilsonSampler(11).sample_forests(kernel, n)
    observed = {}
    for p in parents:
        key = canonical(forest_partition(p))
        observed[key] = observed.get(key, 0) + 1
    assert _chi2_pvalue(observed, expected, n) > 1e-3


def test_forest_partition_labels_components():
    parent = np.array([-1, 0, 1, -1, 3, 0])
    assert forest_partition(parent).tolist() == [0, 0, 0, 1, 1, 0]


def test_spanning_forest_with_given_roots():
    W = np.array([[0, 1.0, 0.1, 0], [1.0, 0, 0.1, 0], [0.1, 0.1, 0, 1.0], [0, 0, 1.0, 0]])
    edges, labels = WilsonSampler(0).sample_spanning_forest(W, roots=[0, 3])
    assert len(edges) == 2 and labels[0] != labels[3]


def test_disconnected_graph_is_rejected():
    W = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="connected"):
        WilsonSampler(0).sample_spanning_tree(W)
