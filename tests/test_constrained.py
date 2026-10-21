import numpy as np
import pytest
from scipy.stats import chisquare

from bgforest import ConstrainedBayesianSpanningForest
from bgforest.core.forest_model import ForestKernel, local_scales
from bgforest.datasets import make_two_moons
from bgforest.exact import canonical, forest_posterior_bruteforce
from bgforest.samplers.wilson import WilsonSampler


def test_every_draw_and_the_point_estimate_satisfy_the_constraints():
    X, _ = make_two_moons(n_samples=60, noise=0.06, random_state=5)
    model = ConstrainedBayesianSpanningForest(
        must_link=[(0, 1)], cannot_link=[(0, 20)], n_draws=30, burn_in=30, random_state=5
    ).fit(X)
    for lab in model.partitions_:
        assert lab[0] == lab[1] and lab[0] != lab[20]
    assert model.check_constraints_satisfied(model.labels_)
    assert 0 < model.acceptance_rate_ <= 1


def test_contradictory_constraints_are_rejected():
    X, _ = make_two_moons(n_samples=12, random_state=3)
    model = ConstrainedBayesianSpanningForest(must_link=[(0, 1), (1, 2)], cannot_link=[(0, 2)])
    with pytest.raises(ValueError, match="cannot-link"):
        model.fit(X)


def test_rejection_draws_the_exact_conditional_posterior():
    rng = np.random.default_rng(8)
    X = np.vstack([rng.normal(0, 0.4, (3, 2)), rng.normal(1.5, 0.4, (2, 2))])
    kernel = ForestKernel.from_data(X, local_scales(X, 2), lam=0.5)
    exact, _, _ = forest_posterior_bruteforce(kernel)
    ok = {k: v for k, v in exact.items() if k[0] == k[4] and k[1] != k[2]}
    total = sum(ok.values())
    expected = {k: v / total for k, v in ok.items()}

    model = ConstrainedBayesianSpanningForest(must_link=[(0, 4)], cannot_link=[(1, 2)])
    model._attempts = model._accepted = 0
    n = 8000
    draws = model._draw_forests(kernel, WilsonSampler(4), n)
    from bgforest.samplers.wilson import forest_partition

    observed = {}
    for p in draws:
        key = canonical(forest_partition(p))
        assert key in expected
        observed[key] = observed.get(key, 0) + 1
    keys = [k for k in expected if expected[k] * n >= 5]
    obs = np.array([observed.get(k, 0) for k in keys], dtype=float)
    exp = np.array([expected[k] * n for k in keys])
    assert chisquare(obs, exp * obs.sum() / exp.sum()).pvalue > 1e-3
    assert np.isclose(model._accepted / model._attempts, total, atol=0.03)
