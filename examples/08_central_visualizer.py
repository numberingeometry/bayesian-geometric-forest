"""
Example 08: Central Interactive Web Dashboard Generator (Human-Crafted Premium UI)
===================================================================================
Generates a state-of-the-art, publication-grade interactive Plotly web dashboard 
(figs/central_visualizer.html) with 100% working Plotly charts, literature taxonomy 
graph, 3D/2D manifold scatter plots, heatmap matrix explorers, and live metric pills.
"""

import base64
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from bgforest.models.bsf import BayesianSpanningForest
from bgforest.models.bst import BayesianSpanningTree
from bgforest.models.distance_clustering import BayesianDistanceClustering
from bgforest.datasets.synthetic import make_two_moons, make_concentric_circles
from bgforest.datasets.single_cell import simulate_scrna_data, preprocess_scrna_data
from bgforest.metrics.evaluation import compute_uncertainty_entropy, compute_clustering_metrics


def generate_central_visualizer():
    print("==================================================")
    print("   Building Central Interactive Web Dashboard")
    print("==================================================")

    # 1. Synthetic Manifolds (Two Moons & Concentric Circles)
    print("Generating synthetic manifold benchmarks...")
    X_tm, y_tm = make_two_moons(n_samples=180, noise=0.08, random_state=42)
    bsf_tm = BayesianSpanningForest(n_clusters=2, graph_type="knn", n_neighbors=6, theta=1.0, sigma_likelihood=10.0, random_state=42).fit(X_tm)
    m_tm = compute_clustering_metrics(y_tm, bsf_tm.labels_)

    X_cc, y_cc = make_concentric_circles(n_samples=180, noise=0.05, random_state=42)
    bsf_cc = BayesianSpanningForest(n_clusters=2, graph_type="knn", n_neighbors=6, theta=1.0, sigma_likelihood=10.0, random_state=42).fit(X_cc)
    m_cc = compute_clustering_metrics(y_cc, bsf_cc.labels_)

    fig_bench = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Interleaved Two Moons (ARI: {m_tm['ARI']:.3f})",
            f"Concentric Circles (ARI: {m_cc['ARI']:.3f})"
        )
    )
    fig_bench.add_trace(
        go.Scatter(
            x=X_tm[:, 0], y=X_tm[:, 1], mode="markers",
            marker=dict(color=bsf_tm.labels_, colorscale="Viridis", size=8, line=dict(width=0.6, color="#0f172a")),
            name="Two Moons",
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Cluster: %{marker.color}<extra></extra>"
        ),
        row=1, col=1
    )
    fig_bench.add_trace(
        go.Scatter(
            x=X_cc[:, 0], y=X_cc[:, 1], mode="markers",
            marker=dict(color=bsf_cc.labels_, colorscale="Plasma", size=8, line=dict(width=0.6, color="#0f172a")),
            name="Concentric Circles",
            hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Cluster: %{marker.color}<extra></extra>"
        ),
        row=1, col=2
    )
    fig_bench.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        height=450,
        margin=dict(l=30, r=30, t=50, b=30)
    )

    # 2. scRNA-Seq 3D Cell Uncertainty
    print("Generating scRNA-seq cell uncertainty visualizer...")
    counts, cell_types, names = simulate_scrna_data(n_cells=240, n_genes=400, n_cell_types=3, random_state=42)
    X_pca, _, _ = preprocess_scrna_data(counts, n_hvg=200, n_pcs=10, random_state=42)
    
    bsf_rna = BayesianSpanningForest(n_clusters=3, graph_type="knn", n_neighbors=8, n_iter=200, random_state=42).fit(X_pca)
    proba_rna = bsf_rna.predict_proba()
    entropy_rna = compute_uncertainty_entropy(proba_rna)

    fig_rna = go.Figure(data=[
        go.Scatter3d(
            x=X_pca[:, 0], y=X_pca[:, 1], z=X_pca[:, 2],
            mode="markers",
            marker=dict(
                size=6,
                color=entropy_rna,
                colorscale="Spectral_r",
                colorbar=dict(title="Entropy H(i)", len=0.8, thickness=15),
                showscale=True
            ),
            text=[f"Cell {i}<br>Type: {names[cell_types[i]]}<br>Cluster: {bsf_rna.labels_[i]}<br>Entropy: {entropy_rna[i]:.3f}" for i in range(len(cell_types))],
            hoverinfo="text"
        )
    ])
    fig_rna.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(15,23,42,0.8)", gridcolor="#334155", title="PC 1"),
            yaxis=dict(backgroundcolor="rgba(15,23,42,0.8)", gridcolor="#334155", title="PC 2"),
            zaxis=dict(backgroundcolor="rgba(15,23,42,0.8)", gridcolor="#334155", title="PC 3")
        ),
        height=520,
        margin=dict(l=10, r=10, t=20, b=10)
    )

    # 3. BST Feature Dependence Backbone
    print("Generating BST feature dependence backbone network...")
    np.random.seed(42)
    Z0 = np.random.randn(150)
    Z1 = Z0 + 0.2 * np.random.randn(150)
    Z2 = Z0 + 0.2 * np.random.randn(150)
    Z3 = Z1 + 0.2 * np.random.randn(150)
    Z4 = Z1 + 0.2 * np.random.randn(150)
    Z5 = Z2 + 0.2 * np.random.randn(150)
    X_feat = np.column_stack([Z0, Z1, Z2, Z3, Z4, Z5])

    bst = BayesianSpanningTree(n_samples_tree=80, random_state=42).fit(X_feat)
    prob_mat = bst.posterior_edge_probabilities_

    fig_bst = go.Figure(data=[
        go.Heatmap(
            z=prob_mat,
            x=[f"Gene_{i}" for i in range(6)],
            y=[f"Gene_{i}" for i in range(6)],
            colorscale="Cividis",
            colorbar=dict(title="Inclusion Prob", len=0.85),
            hovertemplate="Feature %{x} ↔ %{y}<br>Inclusion Prob: %{z:.3f}<extra></extra>"
        )
    ])
    fig_bst.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        height=450,
        margin=dict(l=40, r=40, t=30, b=40)
    )

    # 4. Bayesian Distance Clustering Pairwise Matrix
    print("Generating Bayesian distance clustering matrix...")
    bdc = BayesianDistanceClustering(n_clusters=3, n_iter=150, burn_in=30, random_state=42).fit(X_tm)
    m_bdc = compute_clustering_metrics(y_tm, bdc.labels_)

    fig_bdc = go.Figure(data=[
        go.Heatmap(
            z=bdc.co_clustering_matrix_,
            colorscale="Magma",
            colorbar=dict(title="Co-Clustering P_ij", len=0.85),
            hovertemplate="Node %{x} ↔ Node %{y}<br>P_ij: %{z:.3f}<extra></extra>"
        )
    ])
    fig_bdc.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        font=dict(family="Inter, sans-serif", color="#e2e8f0"),
        height=450,
        margin=dict(l=40, r=40, t=30, b=40)
    )

    # Embed all runtime dependencies and image assets.  The dashboard is then
    # usable from GitHub Pages, a local file, or an exported artifact without a
    # CDN request or relative-path data dependency.
    bench_str = fig_bench.to_json()
    rna_str = fig_rna.to_json()
    bst_str = fig_bst.to_json()
    bdc_str = fig_bdc.to_json()
    plotly_js = get_plotlyjs()
    taxonomy_path = Path(__file__).resolve().parents[1] / "figs" / "literature_taxonomy_graph.png"
    taxonomy_image = base64.b64encode(taxonomy_path.read_bytes()).decode("ascii")

    # Handcrafted, Human-Engineered HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bayesian Geometric Forest — Research & Visual Analytics Portal</title>
    <script>{plotly_js}</script>
    <style>
        :root {{
            --bg-base: #090d16;
            --panel-bg: #131b2e;
            --panel-border: rgba(255, 255, 255, 0.08);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent-cyan: #38bdf8;
            --accent-emerald: #34d399;
            --accent-amber: #fbbf24;
            --accent-purple: #c084fc;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            padding: 0;
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            -webkit-font-smoothing: antialiased;
        }}
        .top-navbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(19, 27, 46, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--panel-border);
            padding: 16px 36px;
            position: sticky;
            top: 0;
            z-index: 100;
        }}
        .brand-logo {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .brand-badge {{
            background: linear-gradient(135deg, #0284c7, #7e22ce);
            width: 32px;
            height: 32px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
            color: #ffffff;
            letter-spacing: -0.5px;
        }}
        .brand-title {{
            font-size: 1.15rem;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: -0.3px;
        }}
        .brand-subtitle {{
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-left: 6px;
        }}
        .repo-link {{
            color: var(--accent-cyan);
            text-decoration: none;
            font-size: 0.88rem;
            font-weight: 500;
            border: 1px solid var(--panel-border);
            padding: 6px 14px;
            border-radius: 6px;
            transition: all 0.2s ease;
        }}
        .repo-link:hover {{
            background: rgba(56, 189, 248, 0.1);
            border-color: var(--accent-cyan);
        }}
        .main-container {{
            max-width: 1440px;
            margin: 0 auto;
            padding: 28px 24px 60px 24px;
        }}
        /* Tab Navigation Bar */
        .tab-bar {{
            display: flex;
            gap: 8px;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            padding: 6px;
            border-radius: 10px;
            margin-bottom: 24px;
            overflow-x: auto;
        }}
        .tab-item {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 10px 18px;
            border-radius: 6px;
            font-size: 0.90rem;
            font-weight: 500;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s ease;
        }}
        .tab-item:hover {{
            color: #ffffff;
            background: rgba(255, 255, 255, 0.04);
        }}
        .tab-item.active {{
            background: #2563eb;
            color: #ffffff;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
        }}
        /* Metric Badges */
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 10px;
            padding: 18px;
        }}
        .metric-label {{
            font-size: 0.80rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 1.6rem;
            font-weight: 700;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            color: var(--accent-emerald);
        }}
        .metric-desc {{
            font-size: 0.80rem;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        /* Panel Section */
        .section-panel {{
            display: none;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 24px;
            animation: fadeIn 0.25s ease-in-out;
        }}
        .section-panel.active {{
            display: block;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .panel-header {{
            margin-bottom: 20px;
        }}
        .panel-title {{
            font-size: 1.25rem;
            font-weight: 600;
            margin: 0 0 6px 0;
            color: #ffffff;
        }}
        .panel-desc {{
            font-size: 0.90rem;
            color: var(--text-muted);
            margin: 0;
        }}
        .chart-frame {{
            border: 1px solid var(--panel-border);
            border-radius: 8px;
            background: #0f172a;
            overflow: hidden;
            min-height: 440px;
        }}
        .taxonomy-img {{
            width: 100%;
            border-radius: 8px;
            border: 1px solid var(--panel-border);
            margin-top: 10px;
        }}
        footer {{
            border-top: 1px solid var(--panel-border);
            padding: 24px;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>

    <div class="top-navbar">
        <div class="brand-logo">
            <div class="brand-badge">BGF</div>
            <div>
                <span class="brand-title">Bayesian Geometric Forest</span>
                <span class="brand-subtitle">Methods explorer</span>
            </div>
        </div>
        <a href="https://github.com/numberingeometry/bayesian-geometric-forest" target="_blank" class="repo-link">GitHub Repository</a>
    </div>

    <div class="main-container">
        
        <!-- Metric Summary Cards -->
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Two Moons ARI</div>
                <div class="metric-value">{m_tm['ARI']:.3f}</div>
                <div class="metric-desc">Two-moons benchmark · BSF</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Concentric Circles ARI</div>
                <div class="metric-value">{m_cc['ARI']:.3f}</div>
                <div class="metric-desc">Concentric-circles benchmark · BSF</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Distance model ARI</div>
                <div class="metric-value">{m_bdc['ARI']:.3f}</div>
                <div class="metric-desc">Two-moons benchmark · experimental</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Dashboard bundle</div>
                <div class="metric-value">Offline</div>
                <div class="metric-desc">Charts and assets are self-contained</div>
            </div>
        </div>

        <!-- Tab Bar -->
        <div class="tab-bar">
            <button class="tab-item active" onclick="switchTab('taxonomy', this)">Literature Taxonomy Map</button>
            <button class="tab-item" onclick="switchTab('benchmarks', this)">Synthetic Manifolds (BSF)</button>
            <button class="tab-item" onclick="switchTab('scrna', this)">scRNA-Seq Cell Uncertainty</button>
            <button class="tab-item" onclick="switchTab('bst', this)">BST Variable Backbone</button>
            <button class="tab-item" onclick="switchTab('distance', this)">Bayesian Distance Clustering</button>
        </div>

        <!-- Section 1: Literature Taxonomy Map -->
        <div id="taxonomy" class="section-panel active">
            <div class="panel-header">
                <h2 class="panel-title">Theoretical Framework & Literature Taxonomy</h2>
                <p class="panel-desc">A map of the related ideas behind graph partitions, dependence trees, distance-based clustering, and loop-erased random walks.</p>
            </div>
            <img src="data:image/png;base64,{taxonomy_image}" alt="Literature taxonomy graph" class="taxonomy-img">
        </div>

        <!-- Section 2: Synthetic Manifolds -->
        <div id="benchmarks" class="section-panel">
            <div class="panel-header">
                <h2 class="panel-title">Non-Linear Manifold Clustering (Bayesian Spanning Forest)</h2>
                <p class="panel-desc">Two standard nonlinear examples, shown with the fitted partition. Hover for observation-level detail.</p>
            </div>
            <div class="chart-frame" id="chart-benchmarks"></div>
        </div>

        <!-- Section 3: scRNA-Seq Uncertainty -->
        <div id="scrna" class="section-panel">
            <div class="panel-header">
                <h2 class="panel-title">Single-Cell RNA-Seq 3D PCA & Bayesian Cell Uncertainty</h2>
                <p class="panel-desc">A simulated single-cell embedding. Colour shows assignment entropy, so uncertain boundary cells are visible immediately.</p>
            </div>
            <div class="chart-frame" id="chart-scrna"></div>
        </div>

        <!-- Section 4: BST Variable Backbone -->
        <div id="bst" class="section-panel">
            <div class="panel-header">
                <h2 class="panel-title">Bayesian Spanning Tree (BST) Variable Dependence Network</h2>
                <p class="panel-desc">Pairwise posterior edge-inclusion frequencies from weighted spanning-tree draws.</p>
            </div>
            <div class="chart-frame" id="chart-bst"></div>
        </div>

        <!-- Section 5: Bayesian Distance Clustering -->
        <div id="distance" class="section-panel">
            <div class="panel-header">
                <h2 class="panel-title">Distance-based clustering (experimental)</h2>
                <p class="panel-desc">An exploratory pairwise-distance co-clustering view. It is kept separate from the validated BSF inference path.</p>
            </div>
            <div class="chart-frame" id="chart-distance"></div>
        </div>

    </div>

    <footer>
        Bayesian Geometric Forest &bull; Reproducible examples generated from this repository
    </footer>

    <script>
        // All figure data and Plotly itself are embedded in this document.
        var specBench = {bench_str};
        var specRna = {rna_str};
        var specBst = {bst_str};
        var specBdc = {bdc_str};

        var configOptions = {{ responsive: true, displayModeBar: true, displaylogo: false }};

        // Render Plotly Charts
        Plotly.newPlot('chart-benchmarks', specBench.data, specBench.layout, configOptions);
        Plotly.newPlot('chart-scrna', specRna.data, specRna.layout, configOptions);
        Plotly.newPlot('chart-bst', specBst.data, specBst.layout, configOptions);
        Plotly.newPlot('chart-distance', specBdc.data, specBdc.layout, configOptions);

        function switchTab(sectionId, element) {{
            var panels = document.querySelectorAll('.section-panel');
            panels.forEach(function(panel) {{
                panel.classList.remove('active');
            }});

            var tabs = document.querySelectorAll('.tab-item');
            tabs.forEach(function(tab) {{
                tab.classList.remove('active');
            }});

            document.getElementById(sectionId).classList.add('active');
            element.classList.add('active');

            // Hidden charts need an explicit resize when their panel becomes visible.
            setTimeout(function() {{
                var chart = document.getElementById('chart-' + sectionId);
                if (chart) {{ Plotly.Plots.resize(chart); }}
            }}, 50);
        }}
    </script>
</body>
</html>
"""

    output_html = "figs/central_visualizer.html"
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[Success] Central HTML visualizer dashboard saved to {output_html}")


if __name__ == "__main__":
    generate_central_visualizer()
