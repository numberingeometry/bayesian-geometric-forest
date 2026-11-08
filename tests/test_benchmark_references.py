"""Independent finite-state checks of the new reference tools."""

import numpy as np

from bgforest.benchmarks import aldous_broder_forests, sampling_kernel
from bgforest.exact import all_labelled_trees, forest_posterior_bruteforce


def test_root_count_distribution_matches_enumerated_partition_law():
    kernel = sampling_kernel(5, "geometric", 0.5)
    partitions, _, roots = forest_posterior_bruteforce(kernel)
    expected = np.zeros(6)
    for labels, p in partitions.items():
        expected[max(labels) + 1] += p
    assert np.allclose(kernel.root_count_distribution(), expected, atol=1e-12)
    assert np.isclose(np.arange(6) @ expected, roots.sum())
    attribution = kernel.root_assignment()
    assert np.allclose(attribution.sum(axis=1), 1.0)
    assert np.allclose(np.diag(attribution), roots)


def test_cover_walk_matches_enumerated_weighted_marginals():
    kernel = sampling_kernel(4, "path", 0.5)
    _, expected_edges, expected_roots = forest_posterior_bruteforce(kernel)
    draws, steps = aldous_broder_forests(kernel, 12000, 843)
    edges, roots = np.zeros((4, 4)), np.zeros(4)
    for parent in draws:
        child = np.flatnonzero(parent >= 0)
        edges[child, parent[child]] += 1
        roots += parent < 0
    assert steps >= 4 * len(draws)
    assert np.max(np.abs((edges + edges.T) / len(draws) - expected_edges)) < 0.018
    assert np.max(np.abs(roots / len(draws) - expected_roots)) < 0.018


def test_root_attribution_matches_independent_tree_enumeration():
    kernel = sampling_kernel(4, "geometric", 0.15)
    kernel.log_root += np.log([0.4, 1.2, 2.0, 0.7])
    log_w = kernel.augmented_log_weights()
    total, expected = 0.0, np.zeros((4, 4))
    for tree in all_labelled_trees(5):
        weight = np.exp(sum(log_w[i, j] for i, j in tree))
        groups = list(range(4))

        def component(i):
            while groups[i] != i:
                i = groups[i]
            return i

        roots = []
        for i, j in tree:
            if i == 4 or j == 4:
                roots.append(j if i == 4 else i)
            else:
                groups[component(i)] = component(j)
        for i in range(4):
            for root in roots:
                if component(i) == component(root):
                    expected[i, root] += weight
        total += weight
    assert np.allclose(kernel.root_assignment(), expected / total, atol=1e-12)
