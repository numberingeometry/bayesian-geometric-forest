"""
Matplotlib Graph Spanning Forest Plotters
=========================================
Visualizes data points, graph edge weights, and topological spanning forest
trees overlaid on 2D embeddings.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree


def plot_spanning_forest(
    X: np.ndarray,
    labels: np.ndarray,
    W: np.ndarray,
    title: str = "Bayesian Spanning Forest Graph Partitioning",
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D data points and overlaid minimum spanning trees for each cluster.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        2D data coordinates.
    labels : np.ndarray of shape (n_samples,)
        Cluster assignments.
    W : np.ndarray of shape (n_samples, n_samples)
        Graph adjacency matrix.
    title : str, default='Bayesian Spanning Forest Graph Partitioning'
        Plot title.
    figsize : Tuple[int, int], default=(8, 6)
        Figure size.
    ax : plt.Axes, optional
        Pre-existing Matplotlib axes object.

    Returns
    -------
    fig, ax : Matplotlib Figure and Axes
    """
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA

        X = PCA(n_components=2).fit_transform(X)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10", len(unique_labels))

    # Draw spanning trees for each cluster
    for idx, cluster in enumerate(unique_labels):
        nodes = np.where(labels == cluster)[0]
        color = cmap(idx)

        # Plot cluster points
        ax.scatter(
            X[nodes, 0],
            X[nodes, 1],
            c=[color],
            label=f"Cluster {cluster}",
            s=40,
            edgecolors="k",
            linewidth=0.5,
            zorder=3,
        )

        if len(nodes) > 1:
            # Extract cluster sub-adjacency matrix
            sub_W = W[np.ix_(nodes, nodes)]
            # Invert weights for MST (max weight spanning tree)
            max_w = np.max(sub_W) if np.max(sub_W) > 0 else 1.0
            dist_sub = max_w - sub_W
            np.fill_diagonal(dist_sub, 0.0)

            mst = minimum_spanning_tree(dist_sub).toarray()

            # Draw tree edges
            for i_local in range(len(nodes)):
                for j_local in range(i_local + 1, len(nodes)):
                    if mst[i_local, j_local] > 0 or mst[j_local, i_local] > 0:
                        p1 = X[nodes[i_local]]
                        p2 = X[nodes[j_local]]
                        ax.plot(
                            [p1[0], p2[0]],
                            [p1[1], p2[1]],
                            c=color,
                            alpha=0.7,
                            linewidth=1.5,
                            zorder=2,
                        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Dimension 1", fontsize=11)
    ax.set_ylabel("Dimension 2", fontsize=11)
    ax.legend(frameon=True, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    return fig, ax
