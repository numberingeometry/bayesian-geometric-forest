"""
Example 06: Bayesian Distance Clustering (Duan & Dunson, 2021, JMLR)
====================================================================
Demonstrates non-parametric Bayesian clustering based directly on pairwise 
distance matrices without parametric kernel shape assumptions.
"""

import numpy as np
import matplotlib.pyplot as plt
from bgforest.models.distance_clustering import BayesianDistanceClustering
from bgforest.datasets.synthetic import make_anisotropic_blobs
from bgforest.metrics.evaluation import compute_clustering_metrics


def main():
    print("==================================================")
    print("   Bayesian Distance Clustering (JMLR 2021)")
    print("==================================================")

    # 1. Generate Anisotropic Benchmark Data
    X, y_true = make_anisotropic_blobs(n_samples=180, random_state=42)

    # 2. Fit Bayesian Distance Clustering Model
    bdc = BayesianDistanceClustering(n_clusters=3, n_iter=200, burn_in=50, random_state=42)
    labels = bdc.fit_predict(X)
    metrics = compute_clustering_metrics(y_true, labels)

    print("\n--- Clustering Performance ---")
    print(f"Adjusted Rand Index (ARI): {metrics['ARI']:.4f}")
    print(f"Normalized Mutual Information (NMI): {metrics['NMI']:.4f}")

    # 3. Plot Distance Matrix and Co-Clustering Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(bdc.distance_matrix_, cmap="viridis")
    axes[0].set_title("Pairwise Distance Matrix D_ij", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(bdc.co_clustering_matrix_, cmap="magma")
    axes[1].set_title(f"Posterior Co-Clustering Matrix (ARI: {metrics['ARI']:.3f})", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    output_png = "figs/distance_clustering_results.png"
    plt.savefig(output_png, dpi=200)
    print(f"\n[Success] Distance clustering figure saved to {output_png}")


if __name__ == "__main__":
    main()
