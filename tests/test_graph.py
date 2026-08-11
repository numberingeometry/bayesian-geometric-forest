"""
Unit Tests for Graph Construction and Laplacians
"""

import numpy as np
import pytest
from bgforest.core.graph import (
    compute_pairwise_distances,
    build_rbf_similarity,
    build_knn_similarity,
    compute_laplacian,
    extract_cluster_submatrix,
)


def test_compute_pairwise_distances():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    D = compute_pairwise_distances(X)
    assert D.shape == (3, 3)
    assert np.isclose(D[0, 1], 5.0)
    assert np.isclose(D[0, 2], 0.0)
    assert np.isclose(D[1, 2], 5.0)


def test_build_rbf_similarity():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    W = build_rbf_similarity(X, sigma=1.0)
    assert W.shape == (3, 3)
    assert np.all(np.diag(W) == 0.0)
    assert W[0, 1] > 0.0
    assert np.isclose(W[0, 1], W[1, 0])
    assert np.isclose(W[0, 1], np.exp(-0.5))


def test_build_rbf_adaptive_bandwidth():
    X = np.random.randn(20, 3)
    W = build_rbf_similarity(X, adaptive_bandwidth=True, k_adaptive=5)
    assert W.shape == (20, 20)
    assert np.all(np.diag(W) == 0.0)
    assert np.all(W >= 0.0)
    assert np.allclose(W, W.T)


def test_build_knn_similarity():
    X = np.random.randn(15, 2)
    W = build_knn_similarity(X, n_neighbors=4, symmetric=True)
    assert W.shape == (15, 15)
    assert np.all(np.diag(W) == 0.0)
    assert np.allclose(W, W.T)


def test_compute_laplacian():
    W = np.array([
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 1.0],
        [2.0, 1.0, 0.0]
    ])
    L = compute_laplacian(W, normed=False)
    # Row sums of unnormalized Laplacian must be zero
    assert np.allclose(np.sum(L, axis=1), 0.0)
    assert np.isclose(L[0, 0], 3.0)
    assert np.isclose(L[0, 1], -1.0)

    L_normed = compute_laplacian(W, normed=True)
    assert L_normed.shape == (3, 3)
    assert np.allclose(L_normed, L_normed.T)


def test_extract_cluster_submatrix():
    W = np.arange(16).reshape(4, 4)
    sub = extract_cluster_submatrix(W, np.array([0, 2]))
    assert sub.shape == (2, 2)
    assert sub[0, 0] == W[0, 0]
    assert sub[0, 1] == W[0, 2]
    assert sub[1, 0] == W[2, 0]
    assert sub[1, 1] == W[2, 2]
