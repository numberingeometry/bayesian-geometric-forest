"""
Example 01: Synthetic Non-Linear Manifold & Misspecified Mixture Benchmarks
===========================================================================
Benchmarks Bayesian Spanning Forest (BSF) against K-Means and Spectral Clustering
across challenging synthetic datasets:
1. Interleaved Two Moons
2. Concentric Circles
3. Multi-Arm Spirals
4. Misspecified Heavy-Tailed Gaussian Mixture
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from bgforest.models.bsf import BayesianSpanningForest
from bgforest.datasets.synthetic import (
    make_two_moons,
    make_concentric_circles,
    make_spirals,
    make_misspecified_mixtures
)
from bgforest.metrics.evaluation import compute_clustering_metrics


def main():
    print("==================================================")
    print("   Bayesian Geometric Forest: Synthetic Benchmarks")
    print("==================================================")

    datasets = [
        ("Two Moons", make_two_moons(n_samples=180, noise=0.08, random_state=42), 2, 6),
        ("Concentric Circles", make_concentric_circles(n_samples=180, noise=0.05, random_state=42), 2, 6),
        ("Multi-Spirals", make_spirals(n_samples=180, n_arms=2, noise=0.05, random_state=42), 2, 6),
        ("Misspecified Mixture", make_misspecified_mixtures(n_samples=180, random_state=42), 3, 10),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))

    for idx, (name, (X, y_true), n_clusters, n_neighbors) in enumerate(datasets):
        # 1. Ground Truth
        axes[idx, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", s=25, edgecolors="k", linewidth=0.3)
        axes[idx, 0].set_title(f"{name}\n(Ground Truth)", fontsize=10, fontweight="bold")

        # 2. K-Means
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        km_labels = km.fit_predict(X)
        m_km = compute_clustering_metrics(y_true, km_labels)
        axes[idx, 1].scatter(X[:, 0], X[:, 1], c=km_labels, cmap="tab10", s=25, edgecolors="k", linewidth=0.3)
        axes[idx, 1].set_title(f"K-Means\nARI: {m_km['ARI']:.3f} | NMI: {m_km['NMI']:.3f}", fontsize=10)

        # 3. Spectral Clustering
        sc = SpectralClustering(n_clusters=n_clusters, affinity="nearest_neighbors", n_neighbors=n_neighbors, random_state=42)
        sc_labels = sc.fit_predict(X)
        m_sc = compute_clustering_metrics(y_true, sc_labels)
        axes[idx, 2].scatter(X[:, 0], X[:, 1], c=sc_labels, cmap="tab10", s=25, edgecolors="k", linewidth=0.3)
        axes[idx, 2].set_title(f"Spectral Clustering\nARI: {m_sc['ARI']:.3f} | NMI: {m_sc['NMI']:.3f}", fontsize=10)

        # 4. Bayesian Spanning Forest (BSF)
        bsf = BayesianSpanningForest(
            n_clusters=n_clusters, graph_type="knn", n_neighbors=n_neighbors,
            theta=1.0, sigma_likelihood=10.0, n_iter=200, burn_in=50, random_state=42
        )
        bsf_labels = bsf.fit_predict(X)
        m_bsf = compute_clustering_metrics(y_true, bsf_labels)
        axes[idx, 3].scatter(X[:, 0], X[:, 1], c=bsf_labels, cmap="tab10", s=25, edgecolors="k", linewidth=0.3)
        axes[idx, 3].set_title(f"Bayesian Spanning Forest\nARI: {m_bsf['ARI']:.3f} | NMI: {m_bsf['NMI']:.3f}", fontsize=10, fontweight="bold")

        print(f"[{name}] ARI -> K-Means: {m_km['ARI']:.3f} | Spectral: {m_sc['ARI']:.3f} | BSF: {m_bsf['ARI']:.3f}")

        for ax in axes[idx]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    output_png = "figs/synthetic_benchmark_results.png"
    plt.savefig(output_png, dpi=200)
    print(f"\n[Success] Benchmark figure saved to {output_png}")


if __name__ == "__main__":
    main()
