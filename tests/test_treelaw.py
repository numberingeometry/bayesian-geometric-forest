import numpy as np
import pytest
from scipy.special import logsumexp

from bgforest import TreeLaw
from bgforest.exact import all_labelled_trees
from bgforest.treelaw import edge_overlap, kron_log_resistance, lam_from_scale, mst_edges


def enumerate_law(C, lam):
    n = C.shape[0]
    trees = list(all_labelled_trees(n))
    logw = np.array([-lam * sum(C[i, j] for i, j in t) for t in trees])
    log_z = logsumexp(logw)
    P = np.zeros((n, n))
    for t, p in zip(trees, np.exp(logw - log_z)):
        for i, j in t:
            P[i, j] += p
            P[j, i] += p
    return log_z, P, trees, logw


@pytest.mark.parametrize("n", [2, 3, 5, 6])
@pytest.mark.parametrize("lam", [0.0, 0.3, 30.0, 3e3, 3e4])
def test_log_space_solver_matches_enumeration(n, lam):
    X = np.random.default_rng(n).uniform(size=(n, 2))
    law = TreeLaw(X, lam)
    log_z, P, _, _ = enumerate_law(law.costs, lam)
    assert abs(law.log_partition() - log_z) <= 1e-12 * max(1.0, abs(log_z))
    assert np.abs(law.edge_probability() - P).max() < 1e-10
    assert law.trace_error() < 1e-12


def test_cold_limit_is_the_minimum_spanning_tree():
    X = np.random.default_rng(3).uniform(size=(40, 2))
    law = TreeLaw(X, 1.0)
    cold = TreeLaw(X, lam_from_scale(law.costs, 1e5))
    e = mst_edges(law.costs)
    assert cold.mst_overlap() > 0.999
    assert cold.trace_error() < 1e-10  # inputs -lam*c carry relative rounding ~ eps * lam * c
    assert np.isclose(cold.edge_probability()[e[:, 0], e[:, 1]].sum(), 39, atol=0.05)


def test_hot_limit_is_the_uniform_tree():
    X = np.random.default_rng(4).uniform(size=(12, 3))
    P = TreeLaw(X, 0.0).edge_probability()
    off = ~np.eye(12, dtype=bool)
    assert np.allclose(P[off], 2 / 12)


def test_free_energy_bound_and_gradient():
    X = np.random.default_rng(5).uniform(size=(20, 2))
    for lam in (5.0, 50.0, 500.0):
        law = TreeLaw(X, lam)
        e = law.mst()
        gap = law.costs[e[:, 0], e[:, 1]].sum() - law.free_energy()
        assert -1e-12 <= gap <= 18 * np.log(20) / lam
        g = law.free_energy_grad()
        h = 1e-6
        for i, d in [(0, 0), (7, 1), (19, 0)]:
            Xp, Xm = X.copy(), X.copy()
            Xp[i, d] += h
            Xm[i, d] -= h
            num = (TreeLaw(Xp, lam).free_energy() - TreeLaw(Xm, lam).free_energy()) / (2 * h)
            assert abs(num - g[i, d]) < 1e-6 * max(1.0, abs(g).max())


def test_expected_length_is_minus_dlogz():
    X = np.random.default_rng(6).uniform(size=(15, 2))
    lam, h = 20.0, 1e-5
    num = -(TreeLaw(X, lam + h).log_partition() - TreeLaw(X, lam - h).log_partition()) / (2 * h)
    assert np.isclose(TreeLaw(X, lam).expected_length(), num, rtol=1e-7)


def test_wilson_draws_match_edge_probabilities():
    X = np.random.default_rng(7).uniform(size=(6, 2))
    law = TreeLaw(X, 3.0)
    parents = law.sample(6000, random_state=1)
    freq = np.zeros((6, 6))
    for par in parents:
        assert (par < 0).sum() == 1
        for v, p in enumerate(par):
            if p >= 0:
                freq[v, p] += 1
                freq[p, v] += 1
    P = law.edge_probability()
    se = np.sqrt(P * (1 - P) / len(parents)) + 1e-12
    assert np.max(np.abs(freq / len(parents) - P) / se) < 4.5


def test_overlap_mean_matches_enumeration():
    X = np.random.default_rng(8).uniform(size=(5, 2))
    _, P, trees, logw = enumerate_law(TreeLaw(X, 2.0).costs, 2.0)
    p = np.exp(logw - logsumexp(logw))
    sets = [{(min(a, b), max(a, b)) for a, b in t} for t in trees]
    mean = sum(p[a] * p[b] * len(sets[a] & sets[b]) for a in range(len(trees)) for b in range(len(trees)))
    assert np.isclose(mean, np.sum(np.triu(P, 1) ** 2), atol=1e-12)
    assert edge_overlap(np.array([-1, 0, 1, 2, 3]), np.array([-1, 0, 1, 2, 3])) == 1.0


def test_disconnected_graph_is_rejected():
    lw = np.full((3, 3), -np.inf)
    lw[0, 1] = lw[1, 0] = 0.0
    with pytest.raises(ValueError):
        kron_log_resistance(lw)


def test_chain_rule_draws_match_enumeration_when_hot_and_cold():
    from scipy.stats import chisquare

    X = np.random.default_rng(11).uniform(size=(5, 2))
    for kappa in (0.3, 3.0, 300.0):
        law = TreeLaw(X, lam_from_scale(TreeLaw(X, 1.0).costs, kappa))
        _, P, trees, logw = enumerate_law(law.costs, law.lam)
        p = np.exp(logw - logsumexp(logw))
        keys = [tuple(sorted((min(i, j), max(i, j)) for i, j in t)) for t in trees]
        draws = law.sample(3000, random_state=4, method="chain")
        counts = {}
        for par in draws:
            k = tuple(sorted((min(v, int(u)), max(v, int(u))) for v, u in enumerate(par) if u >= 0))
            counts[k] = counts.get(k, 0) + 1
        assert set(counts) <= set(keys)
        cells = [i for i, q in enumerate(p) if q * 3000 >= 5]
        if len(cells) > 1:
            obs = np.array([counts.get(keys[i], 0) for i in cells], float)
            exp = p[cells] * 3000
            assert chisquare(obs, exp * obs.sum() / exp.sum()).pvalue > 1e-3
        freq = np.zeros((5, 5))
        for par in draws:
            for v, u in enumerate(par):
                if u >= 0:
                    freq[v, u] += 1
                    freq[u, v] += 1
        se = np.sqrt(P * (1 - P) / 3000) + 1e-12
        assert np.max(np.abs(freq / 3000 - P) / se) < 4.5
