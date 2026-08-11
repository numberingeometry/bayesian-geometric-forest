"""
Example 04: Interactive Plotly Visualizers
==========================================
Generates interactive HTML visualizations for 2D Spanning Forest graph topology 
and single-cell RNA-seq cell-type uncertainty plots.
"""

from bgforest.models.bsf import BayesianSpanningForest
from bgforest.datasets.synthetic import make_two_moons
from bgforest.datasets.single_cell import simulate_scrna_data, preprocess_scrna_data
from bgforest.metrics.evaluation import compute_uncertainty_entropy
from bgforest.viz.interactive import (
    create_interactive_forest_plot,
    create_interactive_scrna_visualizer,
)


def main():
    print("==================================================")
    print("   Interactive Plotly Visualization Generator")
    print("==================================================")

    # 1. Synthetic Moons Interactive Spanning Forest
    X_moons, y_moons = make_two_moons(n_samples=160, noise=0.08, random_state=42)
    bsf_moons = BayesianSpanningForest(n_clusters=2, n_iter=250, random_state=42)
    bsf_moons.fit(X_moons)

    fig_forest = create_interactive_forest_plot(
        X_moons, bsf_moons.labels_, bsf_moons.W_, bsf_moons.predict_proba(),
        title="Interactive Bayesian Spanning Forest - Interleaved Two Moons"
    )
    html_forest = "figs/interactive_spanning_forest.html"
    fig_forest.write_html(html_forest)
    print(f"[Success] Interactive forest plot saved to {html_forest}")

    # 2. scRNA-Seq Interactive Uncertainty Scatter
    counts, y_cell, cell_names = simulate_scrna_data(n_cells=200, n_genes=300, random_state=42)
    pca_emb, _, _ = preprocess_scrna_data(counts, n_pcs=10, random_state=42)
    
    bsf_scrna = BayesianSpanningForest(n_clusters=3, n_iter=250, random_state=42)
    bsf_scrna.fit(pca_emb)
    proba = bsf_scrna.predict_proba()
    entropy = compute_uncertainty_entropy(proba)

    fig_scrna = create_interactive_scrna_visualizer(
        pca_emb, bsf_scrna.labels_, cell_names, uncertainty_entropy=entropy,
        title="Single-Cell RNA-Seq Interactive Bayesian Cell-Type Uncertainty"
    )
    html_scrna = "figs/interactive_scrna_uncertainty.html"
    fig_scrna.write_html(html_scrna)
    print(f"[Success] Interactive scRNA-seq plot saved to {html_scrna}")


if __name__ == "__main__":
    main()
