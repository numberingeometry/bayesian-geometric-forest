"""Matplotlib helpers for forests, co-clustering matrices and uncertainty."""

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

PALETTE = ["#2563eb", "#e11d48", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4d7c0f"]
INK = "#1f2937"
MUTED = "#9ca3af"


def cluster_colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return np.array([PALETTE[int(c) % len(PALETTE)] for c in labels])


def plot_forest(
    X: np.ndarray,
    parent: np.ndarray,
    labels: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    point_size: float = 14.0,
    edge_width: float = 1.1,
    show_roots: bool = True,
) -> plt.Axes:
    """Draw one forest: tree edges coloured by cluster, roots as rings."""
    ax = ax or plt.gca()
    X = np.asarray(X)[:, :2]
    parent = np.asarray(parent)
    if labels is None:
        from bgforest.samplers.wilson import forest_partition

        labels = forest_partition(parent)
    colors = cluster_colors(labels)
    child = np.flatnonzero(parent >= 0)
    segments = np.stack([X[child], X[parent[child]]], axis=1)
    ax.add_collection(LineCollection(segments, colors=colors[child], linewidths=edge_width, alpha=0.85))
    ax.scatter(X[:, 0], X[:, 1], s=point_size, c=colors, linewidths=0, zorder=3)
    if show_roots:
        roots = parent < 0
        ax.scatter(X[roots, 0], X[roots, 1], s=point_size * 6, facecolors="white",
                   edgecolors=colors[roots], linewidths=1.6, zorder=4)
    _clean(ax)
    return ax


def plot_coclustering(P: np.ndarray, labels: np.ndarray, ax: Optional[plt.Axes] = None, cmap: str = "magma_r") -> plt.Axes:
    """Co-clustering probabilities with rows and columns grouped by cluster."""
    ax = ax or plt.gca()
    order = np.argsort(np.asarray(labels), kind="stable")
    ax.imshow(P[np.ix_(order, order)], cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(MUTED)
    return ax


def plot_uncertainty(X: np.ndarray, values: np.ndarray, ax: Optional[plt.Axes] = None,
                     point_size: float = 16.0, cmap: str = "magma_r", vmax: Optional[float] = None) -> plt.Axes:
    """Points coloured by a per-point uncertainty (most uncertain drawn on top)."""
    ax = ax or plt.gca()
    X = np.asarray(X)[:, :2]
    values = np.asarray(values)
    order = np.argsort(values)
    ax.scatter(X[order, 0], X[order, 1], c=values[order], s=point_size, cmap=cmap,
               vmin=0.0, vmax=vmax if vmax is not None else max(float(values.max()), 1e-9), linewidths=0)
    _clean(ax)
    return ax


def plot_trace(values: Sequence[float], burn_in: int = 0, ax: Optional[plt.Axes] = None, label: str = "") -> plt.Axes:
    """A Markov chain trace with the burn-in shaded."""
    ax = ax or plt.gca()
    values = np.asarray(values)
    if burn_in:
        ax.axvspan(0, burn_in, color="#f3f4f6", zorder=0)
    ax.plot(values, color=PALETTE[0], linewidth=0.9)
    ax.set_xlabel("sweep")
    if label:
        ax.set_ylabel(label)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return ax


def _clean(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
