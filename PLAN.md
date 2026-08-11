# Execution & Implementation Plan: `bayesian-geometric-forest`

## Executive Overview
`bayesian-geometric-forest` is a high-performance, modular Python library implementing the theories introduced in:
1. **"Spectral Clustering, Bayesian Spanning Forest, and Forest Process"** (Duan & Roy, 2022/2023, JASA / arXiv:2202.00493)
2. **"Consistency of Graphical Model-based Clustering: Robust Clustering using Bayesian Spanning Forest"** (Zheng, Duan & Roy, 2024, arXiv:2409.19129)

This package unifies **graph-based spectral methods** and **Bayesian nonparametrics**, modeling data partitions via random spanning forests while guaranteeing clustering consistency under model misspecification.

---

## Primary Goals & Portfolio Objectives
- **Theoretical Fidelity**: Faithfully implement the Matrix Tree Theorem, Forest Process urn prior, marginalized likelihood, and MCMC split-merge/edge-swap samplers.
- **Scikit-Learn Compatible API**: Provide `BayesianSpanningForest` with familiar `fit`, `predict`, `fit_predict`, and `predict_proba` methods.
- **Visualization Focus**: Deliver interactive visual tools (graph spanning forest animations, co-clustering posterior heatmaps, spectral eigenvector spectrum analysis, single-cell RNA-seq cell-type uncertainty plots).
- **Professional Standard**: Clean modular structure, comprehensive test suite (`pytest`), sphinx/mkdocs documentation, CI workflow, and conventional commits **without AI co-author signatures**.

---

## Mathematical Formulation & Core Algorithms

### 1. Graph Construction & Adjacency
Given data $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$, construct a similarity graph $G = (V, E, W)$:
- **k-Nearest Neighbors (k-NN)** graph or Gaussian RBF kernel: $W_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$ for $(i,j) \in E$.
- **Graph Laplacian**: Unnormalized $L = D - W$, Normalized $L_{sym} = D^{-1/2} L D^{-1/2}$.

### 2. Bayesian Spanning Forest (BSF) Model
- Data partition $\mathcal{C} = \{C_1, \dots, C_K\}$.
- For each cluster $C_k$, the topological connection is modeled as a **rooted spanning tree** $T_k \sim \text{SpanningTree}(C_k, W)$.
- **Matrix Tree Theorem (Kirchhoff's Theorem)**:
  The total weight of all spanning trees in a connected component $C_k$ is given by any cofactor of the component Laplacian matrix $L_{C_k}$:
  $$\tau(C_k, W) = \det(L_{C_k, (uu)})$$
  where $L_{C_k, (uu)}$ is the submatrix formed by removing the $u$-th row and column.

### 3. The Forest Process (Prior over Forests)
- Defines a prior $P(\mathcal{C})$ extending the classical Dirichlet Process / Chinese Restaurant Process to graphs.
- Prior distribution:
  $$P(\mathcal{C} \mid \alpha, \beta) \propto \alpha^K \prod_{k=1}^K \Gamma(|C_k| - \beta) \cdot \tau(C_k, W)^{\theta}$$
  where $\alpha > 0$ is the concentration parameter and $\beta \in [0, 1)$ governs cluster size discounting.

### 4. Marginalized Likelihood & Posterior Inference
- Marginalize latent edges $E_T$ within trees using log-determinants for numerical stability:
  $$\ln P(X \mid \mathcal{C}) = \sum_{k=1}^K \left[ \ln \det(L_{C_k, (uu)}) - \frac{|C_k|-1}{2}\ln(2\pi\sigma^2) - \text{loss}(C_k) \right]$$
- **MCMC Sampler**:
  - **Edge-Swap Proposal**: Select a tree $T_k$, replace edge $e \in T_k$ with non-tree edge $e'$, updating tree state via Metropolis-Hastings.
  - **Split-Merge Proposal**: Propose splitting a cluster into two trees or merging two adjacent trees with acceptance ratio evaluating the ratio of partition log-determinants.
  - **Posterior Co-Clustering Matrix**: $P_{ij} = \mathbb{P}(\text{node } i \text{ and node } j \text{ in same cluster})$.
  - **Eigen-decomposition of $P$**: Yields leading eigenvectors matching normalized spectral clustering.

### 5. Theoretical Consistency (Zheng et al. 2024)
- Demonstrates posterior concentration $\mathbb{P}(\mathcal{C} = \mathcal{C}^* \mid X_1, \dots, X_n) \to 1$ as $n \to \infty$ under weak cluster separation.
- Derives an explicit upper bound on expected misclassification rate $\mathbb{E}[R_n]$.

---

## Repository Architecture

```text
bayesian-geometric-forest/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Automated testing & linting
├── docs/                          # Package documentation
│   ├── theory.md                  # Detailed math & proofs breakdown
│   ├── tutorials/                 # Step-by-step notebooks & visual guides
│   └── api/                       # API documentation
├── examples/                      # Interactive scripts & demos
│   ├── 01_synthetic_benchmarks.py # Comparison on Two Moons, Circles, Spirals
│   ├── 02_scrna_clustering.py     # Single-cell RNA-seq cell-type partitioning
│   ├── 03_mcmc_diagnostics.py     # Convergence & trace plots
│   └── 04_interactive_viz.py      # Interactive Plotly graph visualizer
├── bgforest/                      # Core Package
│   ├── __init__.py
│   ├── core/                      # Fundamental graph & linear algebra routines
│   │   ├── __init__.py
│   │   ├── graph.py               # Adjacency, Laplacians, k-NN graph construction
│   │   └── matrix_tree.py         # Log-determinant Matrix Tree Theorem solvers
│   ├── models/                    # Model abstractions
│   │   ├── __init__.py
│   │   ├── bsf.py                 # BayesianSpanningForest estimator (sklearn compatible)
│   │   └── forest_process.py      # Prior distribution classes
│   ├── mcmc/                      # Sampler engines
│   │   ├── __init__.py
│   │   ├── sampler.py             # MCMC sampler (edge-swap & split-merge)
│   │   └── diagnostics.py         # Gelman-Rubin R-hat, ESS, trace analysis
│   ├── datasets/                  # Benchmark generators & datasets
│   │   ├── __init__.py
│   │   ├── synthetic.py           # Two Moons, Spirals, Concentric Circles, Blobs
│   │   └── single_cell.py         # scRNA-seq expression simulator & preprocessor
│   ├── metrics/                   # Performance & evaluation metrics
│   │   ├── __init__.py
│   │   └── evaluation.py          # ARI, NMI, misclassification bound, uncertainty entropy
│   └── viz/                       # Visualization suite
│       ├── __init__.py
│       ├── graph_viz.py           # 2D/3D Forest graph overlay plotters
│       ├── posterior_viz.py       # Co-clustering matrix & spectral eigenvector plots
│       └── interactive.py         # Plotly interactive web visualizers
├── tests/                         # Full unit & integration test suite
│   ├── test_graph.py
│   ├── test_matrix_tree.py
│   ├── test_bsf_model.py
│   ├── test_mcmc.py
│   ├── test_datasets.py
│   └── test_metrics.py
├── .gitignore
├── LICENSE                        # MIT License
├── README.md                      # Professional README with benchmarks & figures
└── pyproject.toml                 # Package setup & dependency specification
```

---

## Implementation Phases & Milestones

### Phase 1: Core Mathematical Engine & Graph Tools
- Implement robust graph builder (`bgforest.core.graph`) for RBF kernels, k-NN, and Laplacian matrices.
- Implement Matrix Tree Theorem log-determinant solver (`bgforest.core.matrix_tree`) using Cholesky / LU decomposition with high numerical stability for large matrices.
- Write unit tests verifying spanning tree counts against known analytical graph properties.

### Phase 2: Forest Process Prior & MCMC Sampler
- Implement `ForestProcess` prior distribution (`bgforest.models.forest_process`).
- Build MCMC sampler (`bgforest.mcmc.sampler`) supporting:
  1. Intra-cluster tree edge swaps.
  2. Inter-cluster split/merge Metropolis-Hastings proposals.
- Implement MCMC diagnostic functions (R-hat, Effective Sample Size, log-posterior trace).

### Phase 3: Scikit-Learn API Estimator (`BayesianSpanningForest`)
- Wrap MCMC sampler in `BayesianSpanningForest` estimator class conforming to Scikit-Learn interface (`fit`, `predict`, `fit_predict`, `predict_proba`).
- Compute posterior co-clustering matrix $P$ and extract leading eigenvectors to showcase theoretical equivalence with normalized spectral clustering.

### Phase 4: Datasets & Real-World Case Studies (scRNA-Seq)
- Build synthetic geometric dataset generator (`Two Moons`, `Concentric Circles`, `Spirals`, `High-dim Blobs`).
- Implement single-cell RNA-seq loader and simulated gene expression preprocessor (normalization, HVG selection, PCA embedding).

### Phase 5: Visualization & Interactive Suite
- Build static matplotlib visualizers for tree overlay and co-clustering heatmaps.
- Build interactive Plotly visualizer for exploring posterior tree states, clustering uncertainty, and spectral embeddings.

### Phase 6: Documentation, Tests & GitHub Release
- Write comprehensive README.md with math summary, benchmarks, visual assets, and quick-start guide.
- Complete full test suite (`pytest`) with >90% code coverage.
- Create automated GitHub Actions CI pipeline.

---

## Commit & Quality Guidelines
- **Conventional Commits**: Format every commit as `feat: ...`, `fix: ...`, `docs: ...`, `refactor: ...`, `test: ...`, or `chore: ...`.
- **Zero AI Signatures**: Do not append any `Co-authored-by: Antigravity` or AI metadata to git commits.
- **No Emojis**: Maintain a strictly clean, professional technical document tone.
- **Repository Scope**: All work kept strictly inside `C:\Users\15002\repos\bayesian-geometric-forest`.
