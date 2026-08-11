"""
Interactive Plotly Visualizers
==============================
Generates interactive 2D/3D web-compatible graph figures, spanning forest
tree animations, and single-cell RNA-seq uncertainty visualizers.
"""

from typing import List, Optional

import numpy as np
import plotly.graph_objects as go
from scipy.sparse.csgraph import minimum_spanning_tree


def create_interactive_forest_plot(
    X: np.ndarray,
    labels: np.ndarray,
    W: np.ndarray,
    proba: Optional[np.ndarray] = None,
    title: str = "Bayesian Spanning Forest - Interactive Graph Visualizer",
) -> go.Figure:
    """
    Build interactive Plotly 2D scatter and spanning forest graph overlay.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
        2D data coordinates.
    labels : np.ndarray of shape (n_samples,)
        Cluster assignments.
    W : np.ndarray of shape (n_samples, n_samples)
        Weighted adjacency matrix.
    proba : np.ndarray of shape (n_samples, n_clusters), optional
        Soft assignment probabilities from `predict_proba()`.
    title : str, default='Bayesian Spanning Forest - Interactive Graph Visualizer'
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive Plotly Figure object.
    """
    if X.shape[1] > 2:
        from sklearn.decomposition import PCA

        X = PCA(n_components=2).fit_transform(X)

    fig = go.Figure()

    unique_clusters = np.unique(labels)

    # Palette colors for clusters
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, cluster in enumerate(unique_clusters):
        nodes = np.where(labels == cluster)[0]
        color = colors[idx % len(colors)]

        # 1. Spanning tree edges for this cluster
        if len(nodes) > 1:
            sub_W = W[np.ix_(nodes, nodes)]
            max_w = np.max(sub_W) if np.max(sub_W) > 0 else 1.0
            dist_sub = max_w - sub_W
            np.fill_diagonal(dist_sub, 0.0)

            mst = minimum_spanning_tree(dist_sub).toarray()

            edge_x, edge_y = [], []
            for i_local in range(len(nodes)):
                for j_local in range(i_local + 1, len(nodes)):
                    if mst[i_local, j_local] > 0 or mst[j_local, i_local] > 0:
                        p1 = X[nodes[i_local]]
                        p2 = X[nodes[j_local]]
                        edge_x.extend([p1[0], p2[0], None])
                        edge_y.extend([p1[1], p2[1], None])

            fig.add_trace(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    opacity=0.6,
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        # 2. Scatter nodes for this cluster
        hover_text = []
        for n_idx in nodes:
            txt = f"<b>Node ID:</b> {n_idx}<br><b>Cluster:</b> {cluster}"
            if proba is not None:
                p_val = proba[n_idx, cluster] if cluster < proba.shape[1] else 0.0
                txt += f"<br><b>Posterior Prob:</b> {p_val:.3f}"
            hover_text.append(txt)

        fig.add_trace(
            go.Scatter(
                x=X[nodes, 0],
                y=X[nodes, 1],
                mode="markers",
                marker=dict(size=10, color=color, line=dict(width=1, color="black")),
                name=f"Cluster {cluster}",
                text=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#111111")),
        xaxis=dict(title="Dimension 1", showgrid=True, gridcolor="#EAEAEA"),
        yaxis=dict(title="Dimension 2", showgrid=True, gridcolor="#EAEAEA"),
        plot_bgcolor="white",
        legend=dict(x=0.01, y=0.99, bordercolor="#DDDDDD", borderwidth=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig


def create_interactive_scrna_visualizer(
    embedding: np.ndarray,
    cell_labels: np.ndarray,
    cell_type_names: List[str],
    uncertainty_entropy: Optional[np.ndarray] = None,
    title: str = "Single-Cell RNA-Seq Cell-Type Clustering & Bayesian Uncertainty",
) -> go.Figure:
    """
    Build interactive Plotly visualizer for scRNA-seq cell types and uncertainty.

    Parameters
    ----------
    embedding : np.ndarray of shape (n_cells, 2)
        2D PCA/UMAP embedding coordinates.
    cell_labels : np.ndarray
        Cell cluster labels.
    cell_type_names : List[str]
        Biological cell type names.
    uncertainty_entropy : np.ndarray, optional
        Point-wise Bayesian assignment entropy values.
    title : str, default='Single-Cell RNA-Seq Cell-Type Clustering & Bayesian Uncertainty'
        Plot title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    fig = go.Figure()
    unique_types = np.unique(cell_labels)

    for idx, c_type in enumerate(unique_types):
        cells = np.where(cell_labels == c_type)[0]
        type_name = cell_type_names[c_type] if c_type < len(cell_type_names) else f"Type {c_type}"

        hover_text = []
        for cell_idx in cells:
            txt = f"<b>Cell ID:</b> {cell_idx}<br><b>Type:</b> {type_name}"
            if uncertainty_entropy is not None:
                txt += f"<br><b>Entropy Uncertainty:</b> {uncertainty_entropy[cell_idx]:.3f}"
            hover_text.append(txt)

        marker_dict = dict(size=8, line=dict(width=0.5, color="black"))
        if uncertainty_entropy is not None:
            # Scale size by uncertainty
            sizes = 6 + 10 * (uncertainty_entropy[cells] / (np.max(uncertainty_entropy) + 1e-8))
            marker_dict["size"] = sizes

        fig.add_trace(
            go.Scatter(
                x=embedding[cells, 0],
                y=embedding[cells, 1],
                mode="markers",
                marker=marker_dict,
                name=type_name,
                text=hover_text,
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#111111")),
        xaxis=dict(title="UMAP / PCA 1", showgrid=True, gridcolor="#EAEAEA"),
        yaxis=dict(title="UMAP / PCA 2", showgrid=True, gridcolor="#EAEAEA"),
        plot_bgcolor="white",
        legend=dict(x=0.01, y=0.99, bordercolor="#DDDDDD", borderwidth=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig
