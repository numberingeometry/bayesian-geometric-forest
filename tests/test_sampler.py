"""
Unit Tests for MCMC Sampler and Convergence Diagnostics
"""

import numpy as np

from bgforest.core.graph import build_rbf_similarity
from bgforest.mcmc.diagnostics import compute_effective_sample_size, compute_gelman_rubin_rhat
from bgforest.mcmc.sampler import BSFMCMCSampler


def test_bsf_mcmc_sampler():
    X = np.random.randn(20, 2)
    W = build_rbf_similarity(X, sigma=1.0)

    sampler = BSFMCMCSampler(n_iter=50, burn_in=10, random_state=42)
    samples = sampler.sample(X, W)

    assert len(samples) > 0
    assert len(sampler.traces_) == 50
    assert 0.0 <= sampler.acceptance_rate_ <= 1.0

    P = sampler.compute_co_clustering_matrix()
    assert P.shape == (20, 20)
    assert np.allclose(P, P.T)
    assert np.all(P >= 0.0) and np.all(P <= 1.0)
    assert np.allclose(np.diag(P), 1.0)


def test_mcmc_diagnostics():
    chain1 = [1.0, 1.2, 1.1, 1.3, 1.25, 1.22]
    chain2 = [1.05, 1.15, 1.18, 1.21, 1.24, 1.19]

    rhat = compute_gelman_rubin_rhat([chain1, chain2])
    assert rhat >= 1.0

    ess = compute_effective_sample_size(chain1)
    assert ess >= 1.0
