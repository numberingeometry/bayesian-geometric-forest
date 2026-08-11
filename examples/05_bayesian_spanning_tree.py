"""
Example 05: Bayesian Spanning Tree (BST) Variable Dependence Backbone Discovery
================================================================================
Demonstrates Bayesian Spanning Tree graph estimation (Duan & Dunson, 2024, JMLR)
for discovering the primary backbone structure of high-dimensional variable 
dependence networks.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from bgforest.models.bst import BayesianSpanningTree


def main():
    print("==================================================")
    print("   Bayesian Spanning Tree Backbone Discovery")
    print("==================================================")

    np.random.seed(42)
    n_samples = 150
    p_features = 6

    # 1. Generate Synthetic Data with Tree-Structured Dependence Graph
    print(f"Generating synthetic dataset ({n_samples} samples, {p_features} features)...")
    Z0 = np.random.randn(n_samples)
    Z1 = Z0 + 0.2 * np.random.randn(n_samples)
    Z2 = Z0 + 0.2 * np.random.randn(n_samples)
    Z3 = Z1 + 0.2 * np.random.randn(n_samples)
    Z4 = Z1 + 0.2 * np.random.randn(n_samples)
    Z5 = Z2 + 0.2 * np.random.randn(n_samples)

    X = np.column_stack([Z0, Z1, Z2, Z3, Z4, Z5])

    # 2. Fit Bayesian Spanning Tree Model
    bst = BayesianSpanningTree(n_samples_tree=100, random_state=42)
    bst.fit(X)

    backbone_adj = bst.get_backbone_network()
    prob_matrix = bst.posterior_edge_probabilities_

    print("\n--- Estimated Backbone Edges ---")
    for u, v in bst.mst_edges_:
        print(f"Edge ({u} <-> {v}): Posterior Inclusion Prob = {prob_matrix[u, v]:.3f}")

    # 3. Plot Estimated Backbone Graph Network
    G = nx.from_numpy_array(backbone_adj)
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_nodes(G, pos, node_color="#1f77b4", node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="white", font_size=11, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, width=2.5, edge_color="#d62728", ax=ax)

    ax.set_title("Bayesian Spanning Tree Variable Dependence Backbone", fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    output_png = "figs/bst_backbone_network.png"
    plt.savefig(output_png, dpi=200)
    print(f"\n[Success] Backbone network figure saved to {output_png}")


if __name__ == "__main__":
    main()
