import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from bgforest import BayesianSpanningForest, SpanningTreeBackbone  # noqa: E402
from bgforest.datasets import make_two_moons  # noqa: E402
from bgforest.viz import plot_coclustering, plot_forest, plot_trace, plot_uncertainty  # noqa: E402


def test_backbone_recovers_a_chain():
    rng = np.random.default_rng(0)
    n, p = 2000, 6
    X = np.zeros((n, p))
    X[:, 0] = rng.normal(size=n)
    for j in range(1, p):
        X[:, j] = 0.8 * X[:, j - 1] + 0.6 * rng.normal(size=n)
    model = SpanningTreeBackbone(random_state=0).fit(X)
    assert model.backbone_edges_ == [(j, j + 1) for j in range(p - 1)]
    assert np.isclose(model.edge_probability_.sum() / 2, p - 1)
    trees = model.sample_trees(5)
    assert all(len(t) == p - 1 for t in trees)


def test_plots_render():
    X, _ = make_two_moons(n_samples=40, random_state=0)
    m = BayesianSpanningForest(n_draws=5, burn_in=5, random_state=0).fit(X)
    fig, axes = plt.subplots(1, 4)
    plot_forest(m.X_, m.forests_[-1], ax=axes[0])
    plot_coclustering(m.co_clustering_matrix_, m.labels_, ax=axes[1])
    plot_uncertainty(m.X_, m.assignment_uncertainty(), ax=axes[2])
    plot_trace(m.n_clusters_trace_, burn_in=5, ax=axes[3])
    plt.close(fig)
