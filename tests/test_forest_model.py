"""The augmented-graph identities, checked against brute-force enumeration."""

import numpy as np
import pytest
from scipy.special import logsumexp

from bgforest.core.forest_model import (
    ForestKernel,
    cauchy_root_logdensity,
    gaussian_leaf_logweights,
    local_scales,
)
from bgforest.exact import (
    all_labelled_trees,
    all_set_partitions,
    forest_posterior_bruteforce,
    log_tree_weight_sum_bruteforce,
)


def small_kernel(n=5, seed=0, lam=0.5):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 0.4, (n // 2, 2)), rng.normal(1.5, 0.4, (n - n // 2, 2))])
    return ForestKernel.from_data(X, local_scales(X, 2), rho=1.0, lam=lam)


def test_counts_of_enumerated_objects():
    assert sum(1 for _ in all_labelled_trees(5)) == 5**3  # Cayley
    assert sum(1 for _ in all_set_partitions(6)) == 203  # Bell number


def test_matrix_tree_theorem_matches_enumeration():
    rng = np.random.default_rng(1)
    for m in (2, 3, 5, 6):
        W = rng.uniform(0.1, 2.0, (m, m))
        W = (W + W.T) / 2
        log_W = np.log(W)
        np.fill_diagonal(log_W, -np.inf)
        kernel = ForestKernel(log_W, np.zeros(m))
        assert np.isclose(kernel.log_tree_weight_sum(np.arange(m)), log_tree_weight_sum_bruteforce(log_W))


def test_marginal_partition_posterior_matches_enumeration():
    kernel = small_kernel(5)
    exact, _, _ = forest_posterior_bruteforce(kernel)
    parts = list(all_set_partitions(5))
    log_m = np.array([kernel.log_marginal(np.array(p)) for p in parts])
    prob = np.exp(log_m - logsumexp(log_m))
    for p, q in zip(parts, prob):
        assert np.isclose(q, exact.get(p, 0.0), atol=1e-12)


@pytest.mark.parametrize("lam", [0.05, 0.5, 5.0])
def test_edge_inclusion_closed_form_matches_enumeration(lam):
    kernel = small_kernel(5, seed=2, lam=lam)
    _, M_exact, root_exact = forest_posterior_bruteforce(kernel)
    M, roots = kernel.edge_inclusion()
    assert np.allclose(M, M_exact, atol=1e-12)
    assert np.allclose(roots, root_exact, atol=1e-12)
    # A spanning tree on n + 1 nodes has n edges.
    assert np.isclose(M.sum() / 2 + roots.sum(), kernel.n)
    assert np.isclose(kernel.expected_n_clusters(), roots.sum())


def test_lambda_moves_the_number_of_clusters():
    assert small_kernel(6, lam=5.0).expected_n_clusters() > small_kernel(6, lam=0.05).expected_n_clusters()


def _trapezoid(y, x):
    return float(np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2)


def test_densities_are_normalised():
    # Gaussian leaf: integrate over a grid in 1D.
    grid = np.linspace(-12, 12, 4001)[:, None]
    X = np.vstack([[0.3], grid])
    s = np.full(X.shape[0], 0.8)
    log_f = gaussian_leaf_logweights(X, s)[0, 1:]
    assert np.isclose(_trapezoid(np.exp(log_f), grid[:, 0]), 1.0, atol=1e-6)
    # Cauchy root in 2D: radial integral.
    r = np.linspace(0, 4000, 400001)
    pts = np.c_[r, np.zeros_like(r)]
    dens = np.exp(cauchy_root_logdensity(pts, rho=1.3))
    assert np.isclose(_trapezoid(2 * np.pi * r * dens, r), 1.0, atol=1e-3)


def test_kernel_validation():
    with pytest.raises(ValueError):
        ForestKernel(np.zeros((3, 3)), np.zeros(2))
    with pytest.raises(ValueError):
        gaussian_leaf_logweights(np.zeros((3, 2)), np.array([1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        ForestKernel.from_data(np.zeros((3, 2)) + np.arange(3)[:, None], np.ones(3), lam=0.0)
