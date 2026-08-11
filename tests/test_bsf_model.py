"""
Unit Tests for BayesianSpanningForest Scikit-Learn Estimator
"""

import numpy as np
import pytest
from bgforest.models.bsf import BayesianSpanningForest


def test_bsf_estimator_fit_predict():
    # Simple two cluster dataset
    X1 = np.random.randn(15, 2) + np.array([5.0, 5.0])
    X2 = np.random.randn(15, 2) + np.array([-5.0, -5.0])
    X = np.vstack([X1, X2])

    bsf = BayesianSpanningForest(
        n_clusters=2,
        n_iter=100,
        burn_in=20,
        random_state=42
    )

    labels = bsf.fit_predict(X)
    assert len(labels) == 30
    assert len(np.unique(labels)) == 2

    assert bsf.co_clustering_matrix_.shape == (30, 30)
    assert bsf.spectral_eigenvectors_.shape[0] == 30

    proba = bsf.predict_proba()
    assert proba.shape == (30, 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0)
