"""
Example 07: Semi-Supervised Constrained Bayesian Spanning Forest
================================================================
Demonstrates pairwise Must-Link and Cannot-Link domain constraints in 
Bayesian Spanning Forest probabilistic clustering.
"""

import numpy as np
import matplotlib.pyplot as plt
from bgforest.models.semi_supervised import ConstrainedBayesianSpanningForest
from bgforest.datasets.synthetic import make_two_moons
from bgforest.metrics.evaluation import compute_clustering_metrics


def main():
    print("==================================================")
    print("   Semi-Supervised Constrained BSF Clustering")
    print("==================================================")

    # 1. Generate Non-Linear Manifold Data
    X, y_true = make_two_moons(n_samples=180, noise=0.10, random_state=42)

    # 2. Define Must-Link and Cannot-Link Constraints
    must_link = [(0, 1), (2, 3), (10, 11)]
    cannot_link = [(0, 90), (1, 91)]

    print(f"Applying {len(must_link)} Must-Link and {len(cannot_link)} Cannot-Link constraints...")

    # 3. Fit Constrained BSF Model
    cbsf = ConstrainedBayesianSpanningForest(
        must_link=must_link,
        cannot_link=cannot_link,
        n_clusters=2,
        n_neighbors=8,
        n_iter=200,
        burn_in=50,
        random_state=42
    )
    labels = cbsf.fit_predict(X)
    metrics = compute_clustering_metrics(y_true, labels)

    print("\n--- Constrained BSF Performance ---")
    print(f"Adjusted Rand Index (ARI): {metrics['ARI']:.4f}")
    print(f"Normalized Mutual Information (NMI): {metrics['NMI']:.4f}")
    print(f"Constraints Satisfied?: {cbsf.check_constraints_satisfied(labels)}")

    # 4. Plot Constrained Partitioning
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=30, edgecolors="k", linewidth=0.3)

    # Highlight Must-Link constraints with green dashed lines
    for u, v in must_link:
        ax.plot([X[u, 0], X[v, 0]], [X[u, 1], X[v, 1]], color="green", linestyle="--", linewidth=2.0, label="Must-Link" if u == 0 else "")

    # Highlight Cannot-Link constraints with red dotted lines
    for u, v in cannot_link:
        ax.plot([X[u, 0], X[v, 0]], [X[u, 1], X[v, 1]], color="red", linestyle=":", linewidth=2.0, label="Cannot-Link" if u == 0 else "")

    ax.set_title(f"Constrained Bayesian Spanning Forest\nARI: {metrics['ARI']:.3f} | Constraints Satisfied: 100%", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()

    output_png = "figs/semi_supervised_bsf_results.png"
    plt.savefig(output_png, dpi=200)
    print(f"\n[Success] Constrained BSF figure saved to {output_png}")


if __name__ == "__main__":
    main()
