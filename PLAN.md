# Bayesian Geometric Forest (`bgforest`) — Comprehensive Methods Master Plan

This document outlines the theoretical architecture, literature survey, and step-by-step implementation plan for expanding `bayesian-geometric-forest` (`bgforest`) into a complete unified library covering all existing **Bayesian Spanning Forest**, **Forest Process**, **Bayesian Spanning Tree**, **Bayesian Distance Clustering**, and **Exact Tree Samplers** from the literature.

---

## 1. Literature Survey & Methodological Taxonomy

The literature on graphical model-based Bayesian spanning forest/tree methods comprises five foundational pillars:

```text
                               ┌───────────────────────────────────────────────────────────┐
                               │   Bayesian Geometric Spanning Forest & Graph Methods     │
                               └─────────────────────────────┬─────────────────────────────┘
                                                             │
         ┌───────────────────────────────┬───────────────────┼───────────────────┬───────────────────────────────┐
         │                               │                       │                   │                               │
┌────────┴────────┐             ┌────────┴────────┐     ┌────────┴────────┐ ┌────────┴────────┐             ┌────────┴────────┐
│ 1. Bayesian     │             │ 2. Bayesian     │     │ 3. Bayesian     │ │ 4. Exact Tree   │             │ 5. Semi-Super-  │
│ Spanning Forest │             │ Spanning Tree   │     │ Distance        │ │ Samplers        │             │ vised & Dynamic │
│ (BSF)           │             │ (BST Backbone)  │     │ Clustering      │ │ (Wilson LERW)   │             │ BSF             │
└────────┬────────┘             └────────┬────────┘     └────────┬────────┘ └────────┬────────┘             └────────┬────────┘
         │                               │                       │                   │                               │
  Duan & Roy (2023)              Duan & Dunson           Duan & Dunson       Tam, Dunson &                   Must-Link / 
  Zheng et al. (2024)            (2024, JMLR)            (2021, JMLR)        Duan (2025)                     Cannot-Link
  JASA / Bernoulli                                                                                           Constraints
```

### Pillar 1: Bayesian Spanning Forest (BSF) & Forest Process
- **Primary References**: 
  - Duan, L. L., & Roy, A. (2023). *Spectral Clustering, Bayesian Spanning Forest, and Forest Process*. Journal of the American Statistical Association. ([arXiv:2202.00493](https://arxiv.org/abs/2202.00493))
  - Zheng, Y., Duan, L. L., & Roy, A. (2024). *Consistency of Graphical Model-based Clustering: Robust Clustering using Bayesian Spanning Forest*. Bernoulli. ([arXiv:2409.19129](https://arxiv.org/abs/2409.19129))
- **Mathematical Formulation**:
  - A graph partition $\mathcal{C} = \{C_1, \dots, C_K\}$ is represented as a disjoint union of rooted spanning trees $T(C_k)$.
  - **Prior**: Forest Process $P(\mathcal{C} \mid \alpha, \beta, \theta) \propto \alpha^K \prod_{k=1}^K \Gamma(|C_k| - \beta) \, \tau(C_k, W)^\theta$
  - **Matrix Tree Partition Function**: $\tau(C_k, W) = \det(L_{C_k, (uu)})$ computed via LAPACK BLAS Cholesky decomposition.
  - **Likelihood**: Gaussian / Non-parametric density $P(X \mid \mathcal{C}) = \prod_{k=1}^K P(X_{C_k})$.
  - **Key Theorems**:
    - *Spectral Equivalence Theorem*: Leading eigenvectors of posterior co-clustering matrix $P_{ij} = \mathbb{P}(i \sim j \mid X)$ match normalized spectral clustering.
    - *Posterior Concentration Theorem*: $\mathbb{P}(\mathcal{C} = \mathcal{C}^* \mid X) \to 1$ as $n \to \infty$ under weak separation even when Gaussian assumptions fail.

### Pillar 2: Bayesian Spanning Tree (BST) for Network Backbone Estimation
- **Primary Reference**:
  - Duan, L. L., & Dunson, D. B. (2024). *Bayesian Spanning Tree: Estimating the Backbone of the Dependence Graph*. Journal of Machine Learning Research, 25(102), 1-35. ([arXiv:2012.11867](https://arxiv.org/abs/2012.11867))
- **Mathematical Formulation**:
  - Instead of estimating a full dense precision matrix $\Omega$ or Gaussian Graphical Model, BST models the dependence structure of variables as a spanning tree backbone over the variable graph.
  - Uses Kirchhoff's Matrix Tree theorem prior over tree configurations $T$ to yield exact closed-form marginalization of graph backbones without matrix inversion instability.

### Pillar 3: Bayesian Distance Clustering
- **Primary Reference**:
  - Duan, L. L., & Dunson, D. B. (2021). *Bayesian Distance Clustering*. Journal of Machine Learning Research, 22(224), 1-27. ([arXiv:1806.07542](https://arxiv.org/abs/1806.07542))
- **Mathematical Formulation**:
  - Models the likelihood of pairwise distance matrix $D_{ij} = d(x_i, x_j)$ rather than original raw coordinates $X$.
  - Uses Minimum Spanning Tree (MST) distance upper/lower bounds $d_{\text{MST}}(i, j)$ and distance likelihoods to achieve non-parametric clustering immune to parametric kernel shape assumptions.

### Pillar 4: Exact Tree Samplers (Wilson's LERW & Fast-Forwarded Cover)
- **Primary Reference**:
  - Tam, E., Dunson, D. B., & Duan, L. L. (2025). *Exact sampling of spanning trees via fast-forwarded random walks*. Biometrika, 112(2). ([arXiv:2305.10549](https://arxiv.org/abs/2305.10549))
- **Mathematical Formulation**:
  - Replaces traditional Metropolis-Hastings MCMC mixing bottlenecks with **Wilson's Loop-Erased Random Walk (LERW)** algorithm and **Fast-Forwarded Cover Time sampling**.
  - Generates exact, i.i.d. random spanning trees from the Uniform Spanning Tree (UST) or weighted spanning forest distribution in polynomial time $O(\tau_{\text{cover}})$.

### Pillar 5: Constrained & Semi-Supervised Bayesian Spanning Forest
- **Formulation**:
  - Incorporates pairwise domain knowledge:
    - **Must-Link Set $\mathcal{M}$**: $(i, j) \in \mathcal{M} \implies i \text{ and } j \text{ must belong to the same spanning tree component } C_k$.
    - **Cannot-Link Set $\mathcal{C}_{\text{cannot}}$**: $(i, j) \in \mathcal{C}_{\text{cannot}} \implies i \text{ and } j \text{ must belong to different components}$.
  - Constrained prior: $P(\mathcal{C} \mid \mathcal{M}, \mathcal{C}_{\text{cannot}}) = P(\mathcal{C}) \cdot \mathbb{I}(\mathcal{C} \text{ satisfies } \mathcal{M} \text{ and } \mathcal{C}_{\text{cannot}})$.

---

## 2. Package Architecture Roadmap

We expand `bgforest` into a modular suite:

```text
bgforest/
├── core/
│   ├── graph.py                   # Laplacians, RBF kernel, local adaptive bandwidth
│   └── matrix_tree.py             # Kirchhoff log-determinant log τ(C_k, W) Cholesky solver
├── models/
│   ├── bsf.py                     # BayesianSpanningForest (Unsupervised BSF Estimator)
│   ├── bst.py                     # BayesianSpanningTree (Dependence Backbone Estimator)
│   ├── distance_clustering.py     # BayesianDistanceClustering (Distance-based Model)
│   ├── semi_supervised.py         # ConstrainedBayesianSpanningForest (Must-Link/Cannot-Link BSF)
│   └── forest_process.py          # ForestProcess Prior Distribution Engine
├── samplers/
│   ├── mcmc.py                    # Metropolis-Hastings MCMC with Fiedler bisection & graph relocations
│   └── wilson.py                  # Wilson's Loop-Erased Random Walk (LERW) Exact Sampler
├── datasets/
│   ├── synthetic.py               # Two Moons, Concentric Circles, Spirals, Misspecified Mixtures
│   └── single_cell.py             # scRNA-seq expression simulator & PBMC 3k benchmark loader
├── metrics/
│   └── evaluation.py              # ARI, NMI, uncertainty entropy & theoretical spectral bounds
└── viz/
    ├── graph_viz.py               # Spanning Forest tree network visualizers
    ├── posterior_viz.py           # Co-clustering heatmaps & trace diagnostics
    └── interactive.py             # Plotly 3D/2D interactive HTML visualizers
```

---

## 3. Implementation Steps

### Step 1: Implement `WilsonLERWSampler` (`bgforest/samplers/wilson.py`)
- Implement Wilson's Loop-Erased Random Walk (LERW) algorithm for exact sampling of random spanning trees/forests.
- Support weighted graph transitions $P_{ij} = W_{ij} / D_{ii}$.

### Step 2: Implement `BayesianSpanningTree` (`bgforest/models/bst.py`)
- Build Scikit-Learn style estimator `BayesianSpanningTree` for variable dependence graph backbone estimation (Duan & Dunson 2024).

### Step 3: Implement `BayesianDistanceClustering` (`bgforest/models/distance_clustering.py`)
- Build pairwise distance-based non-parametric clustering estimator `BayesianDistanceClustering` (Duan & Dunson 2021).

### Step 4: Implement `ConstrainedBayesianSpanningForest` (`bgforest/models/semi_supervised.py`)
- Build semi-supervised BSF estimator supporting `must_link` and `cannot_link` constraints.

### Step 5: Complete Benchmark Scripts & Unit Tests
- Add examples `05_bayesian_spanning_tree.py`, `06_distance_clustering.py`, and `07_semi_supervised_bsf.py`.
- Expand unit test suite under `tests/` to cover all new models.

---

## 4. Verification & Quality Control
- Zero AI co-author signatures on git commits (`Bobby Zhang <chelseaandmadrid@gmail.com>`).
- Zero emojis anywhere in codebase or documentation.
- All unit tests pass cleanly with 100% test success rate.
