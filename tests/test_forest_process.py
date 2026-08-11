"""
Unit Tests for Forest Process Prior
"""

import numpy as np
import pytest
from bgforest.core.graph import build_rbf_similarity
from bgforest.models.forest_process import ForestProcess


def test_forest_process_init_validation():
    with pytest.raises(ValueError):
        ForestProcess(alpha=0.0)

    with pytest.raises(ValueError):
        ForestProcess(beta=1.5)

    with pytest.raises(ValueError):
        ForestProcess(theta=-0.5)


def test_forest_process_log_prior():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0], [11.0, 10.0]])
    W = build_rbf_similarity(X, sigma=1.5)

    fp = ForestProcess(alpha=1.0, beta=0.1, theta=1.0)
    partition = np.array([0, 0, 0, 1, 1])

    log_p = fp.log_prior(W, partition)
    assert np.isfinite(log_p)

    # Log prior for a single cluster contribution
    log_p_c0 = fp.log_prior_cluster(W, np.array([0, 1, 2]))
    assert np.isfinite(log_p_c0)
