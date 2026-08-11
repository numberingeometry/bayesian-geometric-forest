"""Generate a compact animation of a Bayesian Geometric Forest fit.

The GIF is derived from one reproducible model run: it reveals the sparse
similarity graph, a representative within-partition spanning forest, and the
posterior co-clustering matrix accumulated from retained Gibbs draws.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse.csgraph import minimum_spanning_tree

from bgforest import BayesianSpanningForest
from bgforest.datasets import make_two_moons

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figs" / "bayesian_geometric_forest_process.gif"


def _display_edges(W: np.ndarray, max_edges: int = 130):
    """Return a readable subset of the strongest graph edges."""
    upper_i, upper_j = np.triu_indices_from(W, k=1)
    weights = W[upper_i, upper_j]
    order = np.argsort(weights)[::-1]
    edges = [(upper_i[idx], upper_j[idx], weights[idx]) for idx in order if weights[idx] > 0]
    return edges[:max_edges]


def _forest_edges(W: np.ndarray, labels: np.ndarray):
    """Build a maximum-weight spanning tree within every fitted cluster."""
    edges = []
    for label in np.unique(labels):
        nodes = np.where(labels == label)[0]
        if len(nodes) < 2:
            continue
        tree = minimum_spanning_tree(-W[np.ix_(nodes, nodes)]).toarray()
        for i, j in zip(*np.where(tree != 0)):
            edges.append((nodes[i], nodes[j]))
    return edges


def generate_process_gif(output_path: Path = OUTPUT) -> Path:
    """Create the reproducible README animation and return its output path."""
    X, _ = make_two_moons(n_samples=100, noise=0.07, random_state=12)
    model = BayesianSpanningForest(
        n_clusters=2,
        graph_type="knn",
        n_neighbors=8,
        theta=0.1,
        n_iter=180,
        burn_in=40,
        random_state=12,
    ).fit(X)

    graph_edges = _display_edges(model.W_)
    forest_edges = _forest_edges(model.W_, model.labels_)
    retained = model.mcmc_sampler_.samples_
    colours = np.array(["#16a6b6", "#ff8d4a", "#8e6ad8", "#68b66b"])

    figure, (ax_graph, ax_posterior) = plt.subplots(
        1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    figure.patch.set_facecolor("#fbfaf7")

    def draw(frame: int):
        ax_graph.clear()
        ax_posterior.clear()
        ax_graph.set_facecolor("#fbfaf7")
        ax_posterior.set_facecolor("#fbfaf7")
        ax_graph.set_xticks([])
        ax_graph.set_yticks([])

        if frame < 10:
            phase = "1. Observations become a geometric graph"
            ax_graph.scatter(X[:, 0], X[:, 1], c="#334155", s=24, alpha=0.88)
            explanation = "Start with observations in feature space."
        elif frame < 28:
            phase = "2. Sparse similarity edges preserve local geometry"
            visible = int((frame - 9) / 18 * len(graph_edges))
            for i, j, weight in graph_edges[:visible]:
                ax_graph.plot(
                    X[[i, j], 0], X[[i, j], 1], color="#94a3b8", alpha=0.13 + 0.35 * weight, lw=0.55
                )
            ax_graph.scatter(X[:, 0], X[:, 1], c="#334155", s=22, zorder=3)
            explanation = "A k-NN graph supplies the weighted support."
        else:
            phase = "3. Posterior partitions favour connected forests"
            visible = min(len(graph_edges), int((frame - 27) / 16 * len(graph_edges)))
            for i, j, weight in graph_edges[:visible]:
                ax_graph.plot(
                    X[[i, j], 0], X[[i, j], 1], color="#cbd5e1", alpha=0.20 + 0.2 * weight, lw=0.5
                )
            forest_visible = min(len(forest_edges), int((frame - 27) / 20 * len(forest_edges)))
            for i, j in forest_edges[:forest_visible]:
                ax_graph.plot(X[[i, j], 0], X[[i, j], 1], color="#0f766e", lw=1.6, zorder=2)
            ax_graph.scatter(
                X[:, 0],
                X[:, 1],
                c=colours[model.labels_],
                s=25,
                edgecolor="white",
                lw=0.35,
                zorder=3,
            )
            explanation = "Tree-weighted partitions are sampled with fixed-K Gibbs updates."

        ax_graph.set_title(phase, fontsize=11, weight="bold", color="#172033", loc="left", pad=12)
        ax_graph.text(
            0.0, -0.10, explanation, transform=ax_graph.transAxes, fontsize=9, color="#475569"
        )

        if frame < 28:
            ax_posterior.axis("off")
            ax_posterior.text(
                0.08, 0.72, "Bayesian Geometric Forest", fontsize=16, weight="bold", color="#172033"
            )
            ax_posterior.text(
                0.08,
                0.48,
                "Similarity graph\n+→ spanning-forest prior\n+→ posterior co-clustering",
                fontsize=12,
                color="#475569",
                linespacing=1.55,
            )
        else:
            count = max(1, min(len(retained), int((frame - 27) / 35 * len(retained))))
            samples = np.asarray(retained[:count])
            posterior = np.mean(samples[:, :, None] == samples[:, None, :], axis=0)
            ax_posterior.imshow(posterior, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            ax_posterior.set_title(
                "Posterior co-clustering",
                fontsize=11,
                weight="bold",
                color="#172033",
                loc="left",
                pad=12,
            )
            ax_posterior.set_xlabel(
                f"{count} retained Gibbs draw{'s' if count != 1 else ''}",
                fontsize=9,
                color="#475569",
            )
            ax_posterior.set_xticks([])
            ax_posterior.set_yticks([])

        figure.tight_layout(pad=1.4)

    animation = FuncAnimation(figure, draw, frames=68, interval=70, repeat=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=14), dpi=105)
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    print(f"Writing {generate_process_gif()}")
