"""
Visualization Suite for Bayesian Geometric Forest (`bgforest.viz`)
===================================================================
"""

from bgforest.viz.graph_viz import plot_spanning_forest
from bgforest.viz.interactive import (
    create_interactive_forest_plot,
    create_interactive_scrna_visualizer,
)
from bgforest.viz.posterior_viz import (
    plot_co_clustering_matrix,
    plot_mcmc_trace,
    plot_spectral_eigenvectors,
)

__all__ = [
    "plot_spanning_forest",
    "plot_co_clustering_matrix",
    "plot_mcmc_trace",
    "plot_spectral_eigenvectors",
    "create_interactive_forest_plot",
    "create_interactive_scrna_visualizer",
]
