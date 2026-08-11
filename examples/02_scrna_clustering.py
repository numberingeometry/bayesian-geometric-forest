"""
Example 02: Single-Cell RNA-Seq Cell-Type Clustering & Uncertainty Quantification
==================================================================================
Simulates single-cell gene expression matrix (cells x genes), applies standard 
preprocessing (log-normalization, HVG selection, PCA), fits Bayesian Spanning Forest,
and quantifies point-wise cell assignment uncertainty.
"""

import numpy as np
import matplotlib.pyplot as plt
from bgforest.models.bsf import BayesianSpanningForest
from bgforest.datasets.single_cell import simulate_scrna_data, preprocess_scrna_data
from bgforest.metrics.evaluation import compute_clustering_metrics, compute_uncertainty_entropy
from bgforest.viz.posterior_viz import plot_co_clustering_matrix


def main():
    print("==================================================")
    print("   scRNA-Seq Cell-Type Partitioning & Uncertainty")
    print("==================================================")

    # 1. Simulate Single-Cell RNA-Seq Expression Data
    n_cells, n_genes = 240, 400
    counts, y_true, cell_names = simulate_scrna_data(
        n_cells=n_cells, n_genes=n_genes, n_cell_types=3,
        cell_type_names=["T-Cell", "B-Cell", "Monocyte"],
        random_state=42
    )
    print(f"Simulated count matrix: {counts.shape} ({n_cells} cells, {n_genes} genes across 3 cell types)")

    # 2. Preprocess Data: Library Normalization, HVG Selection, PCA
    pca_embedding, hvg_data, pca_model = preprocess_scrna_data(
        counts, n_hvg=150, n_pcs=10, random_state=42
    )
    print(f"Extracted PCA low-dimensional embedding: {pca_embedding.shape}")

    # 3. Fit Bayesian Spanning Forest Model
    bsf = BayesianSpanningForest(
        n_clusters=3,
        graph_type="knn",
        n_neighbors=8,
        n_iter=400,
        burn_in=100,
        random_state=42
    )
    bsf_labels = bsf.fit_predict(pca_embedding)
    proba = bsf.predict_proba()

    # 4. Compute Metrics & Uncertainty Entropy
    metrics = compute_clustering_metrics(y_true, bsf_labels)
    entropy = compute_uncertainty_entropy(proba)

    print(f"Cell-Type Partition Performance -> ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f}")
    print(f"Average Cell Entropy Uncertainty: {np.mean(entropy):.4f} (min: {np.min(entropy):.4f}, max: {np.max(entropy):.4f})")

    # 5. Plot Figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # True cell types
    axes[0].scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=y_true, cmap="Set1", s=30, edgecolors="k", linewidth=0.3)
    axes[0].set_title("True Cell Types (Ground Truth)", fontsize=12, fontweight="bold")

    # BSF Predicted cell types
    axes[1].scatter(pca_embedding[:, 0], pca_embedding[:, 1], c=bsf_labels, cmap="Set1", s=30, edgecolors="k", linewidth=0.3)
    axes[1].set_title(f"BSF Predicted Cell Types\nARI: {metrics['ARI']:.3f} | NMI: {metrics['NMI']:.3f}", fontsize=12, fontweight="bold")

    # Uncertainty Sizing Scatter
    sizes = 15 + 80 * (entropy / np.max(entropy))
    sc = axes[2].scatter(
        pca_embedding[:, 0], pca_embedding[:, 1],
        c=entropy, cmap="viridis", s=sizes, edgecolors="k", linewidth=0.3
    )
    cbar = fig.colorbar(sc, ax=axes[2])
    cbar.set_label("Bayesian Entropy Uncertainty", fontsize=10)
    axes[2].set_title("Cell Assignment Uncertainty Entropy", fontsize=12, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("PCA Component 1", fontsize=10)
        ax.set_ylabel("PCA Component 2", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_png = "scrna_clustering_results.png"
    plt.savefig(output_png, dpi=200)
    print(f"\n[Success] scRNA-seq figure saved to {output_png}")

    # Plot co-clustering matrix
    fig_p, _ = plot_co_clustering_matrix(bsf.co_clustering_matrix_, labels=bsf_labels)
    plt.savefig("scrna_co_clustering_matrix.png", dpi=200)


if __name__ == "__main__":
    main()
