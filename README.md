# Bayesian Geometric Forest (`bayesian-geometric-forest`)

A high-performance Python library for **graphical model-based robust clustering** using **Bayesian Spanning Forests** and **Forest Processes**.

This package implements the theories and algorithms introduced in:
1. **"Spectral Clustering, Bayesian Spanning Forest, and Forest Process"** (Duan & Roy, 2022/2023, *Journal of the American Statistical Association* / [arXiv:2202.00493](https://arxiv.org/abs/2202.00493))
2. **"Consistency of Graphical Model-based Clustering: Robust Clustering using Bayesian Spanning Forest"** (Zheng, Duan & Roy, 2024, *Bernoulli* / [arXiv:2409.19129](https://arxiv.org/abs/2409.19129))

---

## Overview

Traditional mixture models (such as Gaussian Mixture Models) rely on strong distributional assumptions and are vulnerable to model misspecification. Standard Spectral Clustering avoids intra-cluster distributional modeling by partitioning graphs via normalized Laplacian cut minimization, but lacks a formal probabilistic framework for quantifying assignment uncertainty.

**Bayesian Geometric Forest** bridges this gap:
- **Random Spanning Forest Generative Model**: Represents cluster topology as a disjoint union of rooted spanning trees.
- **Kirchhoff's Matrix Tree Theorem**: Computes partition functions and log-spanning tree weights $\tau(C_k, W) = \det(L_{C_k, (uu)})$ with numerical Cholesky log-determinants.
- **Forest Process Prior**: An urn-process prior over graph partitions $P(\mathcal{C} \mid \alpha, \beta, \theta) \propto \alpha^K \prod \Gamma(|C_k| - \beta) \, \tau(C_k, W)^\theta$.
- **Theoretical Consistency**: Proves posterior concentration $\mathbb{P}(\mathcal{C} = \mathcal{C}^* \mid X_1, \dots, X_n) \to 1$ under weak separation even when Gaussian assumptions fail.
- **Spectral Equivalence**: Proves that the leading eigenvectors of the MCMC posterior co-clustering matrix $P_{ij} = \mathbb{P}(i \sim j \mid X)$ match normalized spectral clustering.

---

## Quick Start

```python
import numpy as np
from bgforest import BayesianSpanningForest
from bgforest.datasets import make_two_moons
from bgforest.metrics import compute_clustering_metrics, compute_uncertainty_entropy

# 1. Generate Non-linear Manifold Data
X, y_true = make_two_moons(n_samples=200, noise=0.08, random_state=42)

# 2. Fit Scikit-Learn Compatible Estimator
bsf = BayesianSpanningForest(
    n_clusters=2,
    graph_type="knn",
    n_neighbors=10,
    n_iter=500,
    burn_in=100,
    random_state=42
)
labels = bsf.fit_predict(X)

# 3. Predict Soft Assignment Probabilities & Bayesian Uncertainty
proba = bsf.predict_proba()
entropy = compute_uncertainty_entropy(proba)

# 4. Evaluate Performance
metrics = compute_clustering_metrics(y_true, labels)
print(f"Adjusted Rand Index (ARI): {metrics['ARI']:.4f}")
print(f"Normalized Mutual Information (NMI): {metrics['NMI']:.4f}")
```

---

## Benchmarks & Empirical Performance

| Dataset | K-Means (ARI) | Spectral Clustering (ARI) | **Bayesian Spanning Forest (ARI)** |
| :--- | :---: | :---: | :---: |
| **Interleaved Two Moons** | 0.265 | 0.973 | **0.947** |
| **Concentric Circles** | -0.007 | 1.000 | **0.947** |
| **Heavy-Tailed Misspecified Mixture** | 0.934 | 0.983 | **0.967** |
| **Single-Cell RNA-Seq (3 Cell Types)** | 0.280 | 0.350 | **0.419** |

---

## Visualization

### Synthetic Manifold Benchmarks
![Synthetic Benchmark Results](synthetic_benchmark_results.png)

### Single-Cell RNA-Seq Cell-Type Partitioning & Uncertainty
![scRNA-Seq Clustering Results](scrna_clustering_results.png)

### Posterior Co-Clustering Probability Matrix
![scRNA Co-Clustering Matrix](scrna_co_clustering_matrix.png)

### Multi-Chain MCMC Convergence Diagnostics
![MCMC Convergence Diagnostics](mcmc_multi_chain_diagnostics.png)

### Interactive Web Visualizers (Plotly)

To view interactive 3D/2D Plotly visualizers directly in the browser:
- **Interactive Spanning Forest Graph**: [View Interactive HTML Plot](https://numberingeometry.github.io/bayesian-geometric-forest/interactive_spanning_forest.html) (Local File: [`interactive_spanning_forest.html`](interactive_spanning_forest.html))
- **Single-Cell RNA-Seq Bayesian Cell Uncertainty**: [View Interactive HTML Plot](https://numberingeometry.github.io/bayesian-geometric-forest/interactive_scrna_uncertainty.html) (Local File: [`interactive_scrna_uncertainty.html`](interactive_scrna_uncertainty.html))

*(Note: Enabling GitHub Pages under repository Settings -> Pages allows `.html` visualizer links to open live in any web browser.)*

---

## Repository Structure

```text
bayesian-geometric-forest/
├── bgforest/                      # Package source
│   ├── core/                      # Graph Laplacians & Matrix Tree solvers
│   ├── models/                    # BayesianSpanningForest & ForestProcess
│   ├── mcmc/                      # Metropolis-Hastings MCMC sampler & R-hat diagnostics
│   ├── datasets/                  # Synthetic benchmark & scRNA-seq expression simulator
│   ├── metrics/                   # ARI, NMI, uncertainty entropy & theoretical bounds
│   └── viz/                       # Matplotlib & Plotly interactive visualizers
├── examples/                      # Demo scripts & benchmarks
├── tests/                         # Complete pytest unit test suite
├── PLAN.md                        # Master theoretical & architecture plan
├── LICENSE                        # MIT License
├── pyproject.toml                 # Package setup
└── README.md                      # Documentation
```

---

## Citation

If you use `bayesian-geometric-forest` in your research, please cite the underlying papers:

```bibtex
@article{duan2023spectral,
  title={Spectral Clustering, Bayesian Spanning Forest, and Forest Process},
  author={Duan, Leo L and Roy, Arkaprava},
  journal={Journal of the American Statistical Association},
  year={2023}
}

@article{zheng2024consistency,
  title={Consistency of Graphical Model-based Clustering: Robust Clustering using Bayesian Spanning Forest},
  author={Zheng, Yu and Duan, L. L. and Roy, Arkaprava},
  journal={arXiv preprint arXiv:2409.19129},
  year={2024}
}
```
