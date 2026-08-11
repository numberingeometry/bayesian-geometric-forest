"""
Unit Tests for Clustering Evaluation & Uncertainty Metrics
"""

import numpy as np
import pytest
from bgforest.metrics.evaluation import (
    compute_clustering_metrics,
    compute_uncertainty_entropy,
    compute_misclassification_bound,
)


def test_compute_clustering_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])

    metrics = compute_clustering_metrics(y_true, y_pred)
    assert np.isclose(metrics["ARI"], 1.0)
    assert np.isclose(metrics["NMI"], 1.0)


def test_compute_uncertainty_entropy():
    proba = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
        [0.9, 0.1]
    ])
    entropy = compute_uncertainty_entropy(proba)
    assert np.isclose(entropy[0], 0.0, atol=1e-5)
    assert entropy[1] > entropy[2] > entropy[0]


def test_compute_misclassification_bound():
    X = np.array([
        [0.0, 0.0], [0.1, 0.1],
        [10.0, 10.0], [10.1, 10.1]
    ])
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    err, bound = compute_misclassification_bound(X, y_true, y_pred, sigma=1.0)
    assert err == 0.0
    assert bound > 0.0
