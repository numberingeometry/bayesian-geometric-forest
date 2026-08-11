"""Regression tests for validation, constraint preservation, and public API fixes."""

import numpy as np
import pytest

from bgforest import BayesianSpanningForest, BayesianSpanningTree, WilsonLERWSampler
from bgforest.core.graph import (
    build_knn_similarity,
    connect_knn_components,
    validate_similarity_matrix,
)
from bgforest.datasets import make_two_moons
from bgforest.models.semi_supervised import ConstrainedBayesianSpanningForest


def test_knn_component_repair_returns_connected_graph():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    W = build_knn_similarity(X, n_neighbors=1)
    with pytest.raises(ValueError, match="connected"):
        validate_similarity_matrix(W, require_connected=True)
    validate_similarity_matrix(connect_knn_components(W, X), require_connected=True)


def test_wilson_rejects_disconnected_tree_graph():
    W = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="connected"):
        WilsonLERWSampler(0).sample_spanning_tree(W)


def test_constrained_final_partition_enforces_both_constraint_types():
    X, _ = make_two_moons(n_samples=40, noise=0.05, random_state=5)
    model = ConstrainedBayesianSpanningForest(
        must_link=[(0, 1)],
        cannot_link=[(0, 20)],
        n_clusters=2,
        n_iter=40,
        burn_in=10,
        random_state=5,
    ).fit(X)
    assert model.labels_[0] == model.labels_[1]
    assert model.labels_[0] != model.labels_[20]
    assert model.check_constraints_satisfied(model.labels_)


def test_contradictory_constraints_are_rejected():
    X, _ = make_two_moons(n_samples=12, random_state=3)
    model = ConstrainedBayesianSpanningForest(
        must_link=[(0, 1)], cannot_link=[(0, 1)], n_clusters=2, random_state=3
    )
    with pytest.raises(ValueError, match="cannot-link"):
        model.fit(X)


def test_readme_quick_start_uses_feature_tree_weights():
    X, _ = make_two_moons(n_samples=20, random_state=1)
    bsf = BayesianSpanningForest(n_clusters=2, n_iter=20, burn_in=5, random_state=1).fit(X)
    tree = BayesianSpanningTree(n_samples_tree=2, random_state=1).fit(X)
    edges = WilsonLERWSampler(1).sample_spanning_tree(tree.W_feature_)
    assert len(edges) == X.shape[1] - 1
    assert not hasattr(bsf, "W_feature_")
