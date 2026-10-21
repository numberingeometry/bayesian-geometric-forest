import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from bgforest import BayesianSpanningForest
from bgforest.datasets import make_two_moons


@pytest.fixture(scope="module")
def moons():
    return make_two_moons(n_samples=80, noise=0.06, random_state=4)


def test_adaptive_fit_recovers_two_moons(moons):
    X, y = moons
    m = BayesianSpanningForest(n_clusters=2, n_draws=60, burn_in=60, random_state=0).fit(X)
    assert adjusted_rand_score(y, m.labels_) > 0.95
    n = X.shape[0]
    assert m.partitions_.shape == (60, n) and m.forests_.shape == (60, n)
    assert np.allclose(np.diag(m.co_clustering_matrix_), 1.0)
    assert np.allclose(m.co_clustering_matrix_, m.co_clustering_matrix_.T)
    assert np.isclose(sum(m.n_clusters_posterior_.values()), 1.0)
    assert len(m.n_clusters_trace_) == 120
    # every draw is a forest: n - K edges
    assert np.all((m.forests_ >= 0).sum(axis=1) == n - m.n_clusters_draws_)


def test_fixed_scales_give_closed_form_inclusion(moons):
    X, _ = moons
    m = BayesianSpanningForest(scales="local", n_clusters=2, n_draws=50, random_state=1).fit(X)
    M, roots = m.kernel_.edge_inclusion()
    assert np.allclose(m.edge_inclusion_, M) and np.allclose(m.root_probability_, roots)
    assert m.n_clusters_ == 2 and len(np.unique(m.labels_)) == 2
    proba = m.predict_proba()
    assert proba.shape == (X.shape[0], 2) and np.allclose(proba.sum(axis=1), 1.0)
    u = m.assignment_uncertainty()
    assert np.all((u >= 0) & (u <= 1))


def test_fixed_scale_draws_match_closed_form_on_average(moons):
    X, _ = moons
    m = BayesianSpanningForest(scales="local", n_draws=400, random_state=2).fit(X)
    roots_mc = np.mean(m.forests_ < 0, axis=0)
    assert abs(roots_mc.sum() - m.root_probability_.sum()) < 0.25


def test_reproducible_with_seed(moons):
    X, _ = moons
    a = BayesianSpanningForest(n_draws=10, burn_in=10, random_state=5).fit(X)
    b = BayesianSpanningForest(n_draws=10, burn_in=10, random_state=5).fit(X)
    assert np.array_equal(a.partitions_, b.partitions_)


@pytest.mark.parametrize(
    "kwargs", [{"n_clusters": 0}, {"lam": 0.0}, {"scales": "bogus"}, {"n_draws": 0}, {"scales": np.ones(3)}]
)
def test_invalid_parameters(moons, kwargs):
    X, _ = moons
    with pytest.raises(ValueError):
        BayesianSpanningForest(**kwargs).fit(X)
