"""
Unit Tests for Visualization Functions
"""

import numpy as np
import pytest
from bgforest.core.graph import build_rbf_similarity
from bgforest.viz.graph_viz import plot_spanning_forest
from bgforest.viz.posterior_viz import (
    plot_co_clustering_matrix,
    plot_mcmc_trace,
    plot_spectral_eigenvectors,
)
from bgforest.viz.interactive import (
    create_interactive_forest_plot,
    create_interactive_scrna_visualizer,
)


def test_static_visualizers():
    X = np.random.randn(20, 2)
    labels = np.array([0]*10 + [1]*10)
    W = build_rbf_similarity(X, sigma=1.0)

    fig1, ax1 = plot_spanning_forest(X, labels, W)
    assert fig1 is not None

    P = np.eye(20)
    fig2, ax2 = plot_co_clustering_matrix(P, labels=labels)
    assert fig2 is not None

    trace = [1.0, 2.0, 3.0, 3.5, 3.8]
    fig3, ax3 = plot_mcmc_trace(trace, burn_in=2)
    assert fig3 is not None

    eigs = np.random.randn(20, 2)
    fig4, ax4 = plot_spectral_eigenvectors(eigs, labels=labels)
    assert fig4 is not None


def test_interactive_visualizers():
    X = np.random.randn(20, 2)
    labels = np.array([0]*10 + [1]*10)
    W = build_rbf_similarity(X, sigma=1.0)

    fig_plotly1 = create_interactive_forest_plot(X, labels, W)
    assert fig_plotly1 is not None

    fig_plotly2 = create_interactive_scrna_visualizer(
        X, labels, cell_type_names=["Cell A", "Cell B"]
    )
    assert fig_plotly2 is not None
