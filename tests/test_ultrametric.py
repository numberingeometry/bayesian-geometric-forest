import itertools

import numpy as np
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import pdist, squareform

from bgforest.treelaw import squared_distances
from bgforest.ultrametric import (
    cantor_dust,
    is_ultrametric,
    log_resistance_ultrametric,
    resistance_bound,
    single_linkage_ultrametric,
    tree_covariance,
    tree_diffusion,
)


def test_single_linkage_ultrametric_is_scipy_cophenetic():
    X = np.random.default_rng(0).normal(size=(30, 3))
    U = single_linkage_ultrametric(squareform(pdist(X)))
    assert np.allclose(U, squareform(cophenet(linkage(pdist(X), "single"))))
    assert is_ultrametric(U)


def test_single_linkage_is_the_largest_ultrametric_below_the_data():
    D = squareform(pdist(np.random.default_rng(1).normal(size=(12, 2))))
    U = single_linkage_ultrametric(D)
    assert np.all(U <= D + 1e-12)
    # any ultrametric below D is below U: check against complete-linkage heights, scaled
    # down until they stay below D (scaling keeps them ultrametric)
    V = squareform(cophenet(linkage(squareform(D), "complete")))
    V = V * np.min(D[D > 0] / V[D > 0])
    assert is_ultrametric(V) and np.all(V <= D + 1e-12)
    assert np.all(V <= U + 1e-12)


def test_resistance_melts_into_single_linkage():
    C = squared_distances(np.random.default_rng(2).uniform(size=(25, 2)))
    U = single_linkage_ultrametric(C)
    off = ~np.eye(25, dtype=bool)
    gaps = []
    for lam in (1.0, 10.0, 100.0, 1e3, 1e4):
        gap = np.abs(log_resistance_ultrametric(C, lam) - U)[off].max()
        assert gap <= resistance_bound(25, lam)
        gaps.append(gap)
    assert gaps[-1] < 1e-3 * gaps[0]


def test_brownian_tree_covariance():
    X, parent = tree_diffusion(9, sigma=0.5, random_state=4)
    G = tree_covariance(parent)
    root = int(np.flatnonzero(parent < 0)[0])
    keep = [i for i in range(9) if i != root]
    L = np.zeros((9, 9))
    for v, p in enumerate(parent):
        if p >= 0:
            L[[v, p], [v, p]] += 1
            L[v, p] -= 1
            L[p, v] -= 1
    Gk = G[np.ix_(keep, keep)]
    assert np.allclose(Gk, np.linalg.inv(L[np.ix_(keep, keep)]))
    assert np.isclose(np.linalg.det(Gk), 1.0)
    for i, j, k in itertools.product(range(8), repeat=3):
        assert Gk[i, j] >= min(Gk[i, k], Gk[k, j]) - 1e-12
    Y = (X - X[root])[keep]
    S = sum(np.sum((X[v] - X[p]) ** 2) for v, p in enumerate(parent) if p >= 0)
    assert np.isclose(np.trace(Y.T @ np.linalg.solve(Gk, Y)), S)


def test_tree_diffusion_step_variance():
    X, parent = tree_diffusion(4000, sigma=0.3, dim=2, random_state=5)
    steps = np.array([X[v] - X[p] for v, p in enumerate(parent) if p >= 0])
    assert abs(steps.var() / 0.09 - 1) < 0.05


def test_cantor_dust_is_hierarchical():
    P = cantor_dust(3)
    assert P.shape == (64, 2)
    D = squareform(pdist(P))
    U = single_linkage_ultrametric(D)
    off = ~np.eye(64, dtype=bool)
    assert (D[off] / U[off]).max() < 4.0  # bounded distortion: bi-Lipschitz to an ultrametric
