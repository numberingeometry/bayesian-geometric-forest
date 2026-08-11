"""
Unit Tests for Matrix Tree Theorem Log-Determinant Solvers
"""

import numpy as np
import pytest
from bgforest.core.graph import compute_laplacian
from bgforest.core.matrix_tree import (
    log_spanning_trees_count,
    compute_cluster_log_spanning_trees,
    compute_forest_log_spanning_trees,
)


def test_cayley_formula_complete_graphs():
    """Verify Matrix Tree Theorem against Cayley's formula: n^(n-2) for Complete Graph K_n."""
    for n in [3, 4, 5, 6]:
        # Complete graph K_n adjacency
        W = np.ones((n, n)) - np.eye(n)
        L = compute_laplacian(W)
        
        log_tau = log_spanning_trees_count(L)
        expected_log_tau = (n - 2) * np.log(n)
        
        assert np.isclose(log_tau, expected_log_tau, rtol=1e-5), f"Failed for K_{n}"


def test_cycle_graph_spanning_trees():
    """Cycle graph C_n has exactly n spanning trees."""
    for n in [3, 4, 5, 10]:
        W = np.zeros((n, n))
        for i in range(n):
            W[i, (i + 1) % n] = 1.0
            W[(i + 1) % n, i] = 1.0
        
        L = compute_laplacian(W)
        log_tau = log_spanning_trees_count(L)
        assert np.isclose(log_tau, np.log(n), rtol=1e-5), f"Failed for C_{n}"


def test_single_node_spanning_tree():
    W = np.array([[0.0]])
    L = compute_laplacian(W)
    log_tau = log_spanning_trees_count(L)
    assert log_tau == 0.0


def test_compute_forest_log_spanning_trees():
    # 2 separate components: K_3 and K_4
    W = np.zeros((7, 7))
    # K_3 on nodes 0,1,2
    W[0:3, 0:3] = 1.0 - np.eye(3)
    # K_4 on nodes 3,4,5,6
    W[3:7, 3:7] = 1.0 - np.eye(4)

    partition = np.array([0, 0, 0, 1, 1, 1, 1])
    total_log_tau = compute_forest_log_spanning_trees(W, partition)
    
    expected = (3 - 2) * np.log(3) + (4 - 2) * np.log(4)
    assert np.isclose(total_log_tau, expected)
