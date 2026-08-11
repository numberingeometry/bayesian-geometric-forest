"""
Unit tests for Constrained Bayesian Spanning Forest estimator.
"""

import numpy as np

from bgforest.datasets.synthetic import make_two_moons
from bgforest.models.semi_supervised import ConstrainedBayesianSpanningForest


def test_constrained_bsf_estimator():
    X, y = make_two_moons(n_samples=50, noise=0.08, random_state=42)

    must_link = [(0, 1), (2, 3)]
    cannot_link = [(0, 25)]

    cbsf = ConstrainedBayesianSpanningForest(
        must_link=must_link,
        cannot_link=cannot_link,
        n_clusters=2,
        n_iter=50,
        burn_in=10,
        random_state=42,
    )
    labels = cbsf.fit_predict(X)

    assert len(labels) == 50
    assert len(np.unique(labels)) == 2
    assert cbsf.labels_[0] == cbsf.labels_[1]
    assert cbsf.labels_[2] == cbsf.labels_[3]
