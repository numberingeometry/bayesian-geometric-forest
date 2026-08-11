"""
Unit tests for Bayesian Distance Clustering estimator.
"""

import numpy as np

from bgforest.datasets.synthetic import make_anisotropic_blobs
from bgforest.models.distance_clustering import BayesianDistanceClustering


def test_bayesian_distance_clustering():
    X, y = make_anisotropic_blobs(n_samples=60, random_state=42)

    bdc = BayesianDistanceClustering(n_clusters=3, n_iter=50, burn_in=10, random_state=42)
    labels = bdc.fit_predict(X)

    assert len(labels) == 60
    assert len(np.unique(labels)) == 3
    assert bdc.distance_matrix_.shape == (60, 60)
    assert bdc.co_clustering_matrix_.shape == (60, 60)
