"""
Unit tests for Bayesian Spanning Tree feature dependence backbone estimator.
"""

import numpy as np

from bgforest.models.bst import BayesianSpanningTree


def test_bayesian_spanning_tree_fit():
    np.random.seed(42)
    # Generate data with strong linear dependence between X0-X1 and X2-X3
    X0 = np.random.randn(100)
    X1 = X0 + 0.1 * np.random.randn(100)
    X2 = np.random.randn(100)
    X3 = X2 + 0.1 * np.random.randn(100)

    X = np.column_stack([X0, X1, X2, X3])

    bst = BayesianSpanningTree(n_samples_tree=30, random_state=42)
    bst.fit(X)

    assert bst.W_feature_ is not None
    assert bst.W_feature_.shape == (4, 4)
    assert len(bst.mst_edges_) == 3  # Spanning tree on 4 features has 3 edges

    adj = bst.get_backbone_network()
    assert adj.shape == (4, 4)
    assert np.all(adj == adj.T)
    assert bst.posterior_edge_probabilities_.shape == (4, 4)
