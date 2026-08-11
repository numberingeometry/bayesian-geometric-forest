"""
Unit tests for Wilson's Loop-Erased Random Walk (LERW) sampler.
"""

import numpy as np
from bgforest.samplers.wilson import WilsonLERWSampler


def test_wilson_spanning_tree_sampling():
    W = np.array([
        [0.0, 1.0, 0.5, 0.0],
        [1.0, 0.0, 1.0, 0.2],
        [0.5, 1.0, 0.0, 1.0],
        [0.0, 0.2, 1.0, 0.0]
    ])

    sampler = WilsonLERWSampler(random_state=42)
    edges = sampler.sample_spanning_tree(W, root=0)

    # A spanning tree on 4 nodes must have exactly 3 edges
    assert len(edges) == 3

    # Check connected nodes
    visited = set([0])
    for u, v in edges:
        visited.add(u)
        visited.add(v)
    assert len(visited) == 4


def test_wilson_spanning_forest_sampling():
    W = np.array([
        [0.0, 1.0, 0.1, 0.0],
        [1.0, 0.0, 0.1, 0.0],
        [0.1, 0.1, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ])

    sampler = WilsonLERWSampler(random_state=42)
    edges, partition = sampler.sample_spanning_forest(W, roots=[0, 3])

    assert len(roots_visited := np.unique(partition)) == 2
    assert partition[0] != partition[3]
    assert len(edges) == 2
