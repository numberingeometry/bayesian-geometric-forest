"""
Posterior Co-Clustering & MCMC Diagnostic Visualizers
=====================================================
Routines for plotting reordered co-clustering heatmaps, MCMC log-posterior
convergence traces, and spectral eigenvector embeddings.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_co_clustering_matrix(
    P: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Posterior Co-Clustering Matrix P_ij",
    figsize: Tuple[int, int] = (7, 6),
    cmap: str = "viridis",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot reordered posterior co-clustering matrix P_ij.

    Parameters
    ----------
    P : np.ndarray of shape (n, n)
        Symmetric co-clustering probability matrix.
    labels : np.ndarray of shape (n,), optional
        Cluster labels used to reorder matrix rows and columns.
    title : str, default='Posterior Co-Clustering Matrix P_ij'
        Plot title.
    figsize : Tuple[int, int], default=(7, 6)
        Figure dimensions.
    cmap : str, default='viridis'
        Colormap name.

    Returns
    -------
    fig, ax : Matplotlib Figure and Axes
    """
    if labels is not None:
        order = np.argsort(labels)
        P_plot = P[np.ix_(order, order)]
    else:
        P_plot = P

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(P_plot, cmap=cmap, origin="upper", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(node i and j in same cluster)", fontsize=11)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Reordered Node Index", fontsize=11)
    ax.set_ylabel("Reordered Node Index", fontsize=11)
    plt.tight_layout()

    return fig, ax


def plot_mcmc_trace(
    trace: List[float],
    burn_in: int = 0,
    title: str = "MCMC Log-Posterior Convergence Trace",
    figsize: Tuple[int, int] = (8, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot MCMC log-posterior trace with burn-in line and rolling average.

    Parameters
    ----------
    trace : List[float]
        Log-posterior values recorded across MCMC iterations.
    burn_in : int, default=0
        Burn-in iteration index threshold.
    title : str, default='MCMC Log-Posterior Convergence Trace'
        Plot title.
    figsize : Tuple[int, int], default=(8, 4)
        Figure size.

    Returns
    -------
    fig, ax : Matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    iters = np.arange(len(trace))

    ax.plot(iters, trace, color="#1f77b4", alpha=0.6, linewidth=1.0, label="Log-Posterior")

    # Compute moving average
    window = max(5, len(trace) // 20)
    if len(trace) >= window:
        moving_avg = np.convolve(trace, np.ones(window) / window, mode="valid")
        ax.plot(
            iters[window - 1 :],
            moving_avg,
            color="#d62728",
            linewidth=2.0,
            label=f"Moving Avg (w={window})",
        )

    if burn_in > 0:
        ax.axvline(burn_in, color="black", linestyle="--", linewidth=1.5, label="Burn-in Cutoff")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("MCMC Iteration", fontsize=11)
    ax.set_ylabel("Log-Posterior", fontsize=11)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig, ax


def plot_spectral_eigenvectors(
    eigenvectors: np.ndarray,
    labels: np.ndarray,
    title: str = "Leading Eigenvector Embedding of Posterior Matrix P",
    figsize: Tuple[int, int] = (7, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D scatter of leading eigenvectors extracted from posterior matrix P.

    Demonstrates theoretical equivalence to normalized spectral clustering (Duan & Roy, 2022).
    """
    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))

    for idx, cluster in enumerate(unique_labels):
        nodes = np.where(labels == cluster)[0]
        ax.scatter(
            eigenvectors[nodes, 0],
            eigenvectors[nodes, 1],
            c=[cmap(idx)],
            label=f"Cluster {cluster}",
            s=40,
            edgecolors="k",
            linewidth=0.5,
            alpha=0.9,
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Eigenvector 1", fontsize=11)
    ax.set_ylabel("Eigenvector 2", fontsize=11)
    ax.legend(frameon=True, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig, ax
