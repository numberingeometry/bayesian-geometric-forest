"""
Example 00: Literature Taxonomy & Methodological Connections Diagram Generator
=============================================================================
Generates a clear visual graph showing the mathematical and methodological 
relationships between all 5 foundational papers in the Bayesian Geometric Forest 
literature.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def main():
    fig, ax = plt.subplots(figsize=(14, 9), dpi=200)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Set background color
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Title
    ax.text(
        7.0, 9.4,
        "Theoretical Framework & Literature Taxonomy of Bayesian Geometric Forests",
        fontsize=15, fontweight="bold", ha="center", va="center", color="#1a252c"
    )
    ax.text(
        7.0, 9.0,
        "Methodological Foundations, Theoretical Guarantees, Distance Likelihoods, and Exact Tree Samplers",
        fontsize=11, fontstyle="italic", ha="center", va="center", color="#4a5568"
    )

    # Box properties helper
    def draw_box(x, y, w, h, title, authors, journal, desc, box_color, border_color):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.2",
            linewidth=1.8, edgecolor=border_color, facecolor=box_color
        )
        ax.add_patch(rect)

        ax.text(x + w / 2, y + h - 0.35, title, fontsize=11, fontweight="bold", ha="center", color="#1a202c")
        ax.text(x + w / 2, y + h - 0.70, authors, fontsize=9.5, fontweight="semibold", ha="center", color="#2b6cb0")
        ax.text(x + w / 2, y + h - 1.00, journal, fontsize=8.5, fontstyle="italic", ha="center", color="#4a5568")

        # Description text
        ax.text(x + w / 2, y + h / 2 - 0.40, desc, fontsize=8.5, ha="center", va="center", color="#2d3748", multialignment="center")

    # 1. Core BSF Paper (Top Center)
    draw_box(
        4.3, 5.8, 5.4, 2.4,
        "1. Bayesian Spanning Forest & Forest Process",
        "Leo L. Duan & Arkaprava Roy",
        "JASA (Journal of the American Statistical Association), 2023",
        "Generative graphical model for spectral clustering.\nMatrix Tree Theorem: log τ(C_k, W) = det(L_C, (uu))\nSpectral Equivalence Theorem to Normalized Cuts",
        "#ebf8ff", "#3182ce"
    )

    # 2. Consistency Paper (Top Right)
    draw_box(
        10.2, 5.8, 3.4, 2.4,
        "2. Asymptotic Consistency",
        "Y. Zheng, L. L. Duan, A. Roy",
        "Bernoulli, 2024",
        "Posterior concentration P(C = C* | X) -> 1\nTheoretical robustness under weak separation\nand model misspecification.",
        "#f0fff4", "#38a169"
    )

    # 3. Spanning Tree Backbone Paper (Bottom Left)
    draw_box(
        0.4, 1.2, 3.8, 2.4,
        "3. Bayesian Spanning Tree",
        "Leo L. Duan & David B. Dunson",
        "JMLR (Journal of Machine Learning Research), 2024",
        "Estimates backbone of variable dependence\nnetworks across features p without full\nprecision matrix inversion.",
        "#fffaf0", "#dd6b20"
    )

    # 4. Bayesian Distance Clustering (Bottom Center)
    draw_box(
        4.8, 1.2, 4.4, 2.4,
        "4. Bayesian Distance Clustering",
        "Leo L. Duan & David B. Dunson",
        "JMLR (Journal of Machine Learning Research), 2021",
        "Non-parametric distance likelihood P(D_ij | C)\ndirectly on pairwise distance matrices D_ij.\nKernel-free shape robustness.",
        "#faf5ff", "#805ad5"
    )

    # 5. Exact Tree Sampler (Bottom Right)
    draw_box(
        9.7, 1.2, 3.9, 2.4,
        "5. Exact Sampler (Wilson LERW)",
        "E. Tam, D. B. Dunson, L. L. Duan",
        "Biometrika, 2025",
        "Exact, unbiased spanning tree sampling via\nWilson's Loop-Erased Random Walk (LERW)\nwithout MCMC mixing bottlenecks.",
        "#fff5f5", "#e53e3e"
    )

    # Draw Connector Arrows
    arrow_style = dict(arrowstyle="->", lw=2, color="#4a5568")
    
    # Paper 1 -> Paper 2 (Consistency)
    ax.annotate("", xy=(10.2, 7.0), xytext=(9.7, 7.0), arrowprops=arrow_style)
    ax.text(9.95, 7.25, "Consistency\nProof", fontsize=8, ha="center", color="#2b6cb0", fontweight="bold")

    # Paper 1 -> Paper 3 (BST Backbone)
    ax.annotate("", xy=(2.3, 3.6), xytext=(5.0, 5.8), arrowprops=arrow_style)
    ax.text(3.3, 4.9, "Variable Graph\nBackbone", fontsize=8, ha="center", color="#dd6b20", fontweight="bold")

    # Paper 1 -> Paper 4 (Distance Likelihood)
    ax.annotate("", xy=(7.0, 3.6), xytext=(7.0, 5.8), arrowprops=arrow_style)
    ax.text(7.0, 4.7, "Distance Matrix\nLikelihood", fontsize=8, ha="center", color="#805ad5", fontweight="bold")

    # Paper 1 -> Paper 5 (Exact Sampler)
    ax.annotate("", xy=(11.6, 3.6), xytext=(9.0, 5.8), arrowprops=arrow_style)
    ax.text(10.7, 4.9, "Exact LERW\nSampling", fontsize=8, ha="center", color="#e53e3e", fontweight="bold")

    plt.tight_layout()
    output_png = "figs/literature_taxonomy_graph.png"
    plt.savefig(output_png, dpi=200)
    print(f"[Success] Literature taxonomy graph saved to {output_png}")


if __name__ == "__main__":
    main()
