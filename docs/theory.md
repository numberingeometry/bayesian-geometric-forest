# Theoretical Foundations & Mathematical Proofs

## 1. Introduction & Problem Formulation

Clustering data $X = \{x_1, \dots, x_n\} \subset \mathbb{R}^d$ into $K$ components is classically modeled using mixture distributions $P(x) = \sum_{k=1}^K \pi_k f(x \mid \theta_k)$. However, when the true data-generating density deviates from standard parametric families (e.g., non-Gaussian heavy-tailed distributions or complex non-linear manifolds), traditional mixture models suffer from severe **model misspecification**.

The **Bayesian Spanning Forest (BSF)** model addresses this challenge by defining the statistical likelihood directly over the **graph topology** of data partitions, bypassing intra-cluster parametric assumptions.

---

## 2. Kirchhoff's Matrix Tree Theorem & Spanning Tree Weight Counting

Given a similarity graph $G = (V, E, W)$ with adjacency weights $W_{ij} \ge 0$, the unnormalized graph Laplacian for a cluster $C_k \subseteq V$ of size $|C_k| = m$ is defined as:

$$L_{C_k} = D_{C_k} - W_{C_k}$$

where $D_{C_k} = \text{diag}(\sum_{j} W_{ij})$.

### Theorem 1 (Kirchhoff's Matrix Tree Theorem)
The sum of products of edge weights over all spanning trees $T$ in a connected component $C_k$ is equal to any cofactor of the Laplacian matrix $L_{C_k}$:

$$\tau(C_k, W) = \sum_{T \in \text{SpanningTrees}(C_k)} \prod_{e \in T} w(e) = \det(L_{C_k, (uu)})$$

where $L_{C_k, (uu)}$ is the $(m-1) \times (m-1)$ principal submatrix formed by deleting any arbitrary row $u$ and column $u$.

---

## 3. The Forest Process Prior

The **Forest Process** defines a nonparametric prior distribution over graph partitions $\mathcal{C} = \{C_1, \dots, C_K\}$ on similarity graphs:

$$P(\mathcal{C} \mid \alpha, \beta, \theta) = \frac{\alpha^K}{Z(\alpha, \beta, \theta)} \prod_{k=1}^K \Gamma(|C_k| - \beta) \cdot \tau(C_k, W)^{\theta}$$

where:
- $\alpha > 0$ is the concentration hyper-parameter controlling the number of clusters $K$.
- $\beta \in [0, 1)$ is the Pitman-Yor discount parameter governing power-law cluster size tails.
- $\theta \ge 0$ is the graph topological scaling exponent regulating spanning forest connectivity.

---

## 4. Eigen-Equivalence to Normalized Spectral Clustering

### Theorem 2 (Duan & Roy, 2023)
Let $P_{ij} = \mathbb{P}(\text{node } i \text{ and node } j \text{ belong to same cluster} \mid X)$ be the MCMC posterior co-clustering probability matrix. 

Under mild conditions on the graph weights $W$, the leading $K$ eigenvectors of $P$ converge asymptotically to the leading eigenvectors of the normalized graph Laplacian $L_{sym} = D^{-1/2} L D^{-1/2}$.

---

## 5. Theoretical Consistency under Misspecification (Zheng, Duan & Roy, 2024)

### Theorem 3 (Posterior Concentration)
Let data $X_1, \dots, X_n$ be generated from an unknown collection of $K^*$ component distributions $P_1^*, \dots, P_{K^*}^*$. 

Assuming a mild asymptotic cluster separation condition holds with probability tending to 1:

$$\min_{k \neq k'} \text{dist}(C_k^*, C_{k'}^*) \ge \Delta_n > 0$$

The posterior distribution concentrates on the true partition $\mathcal{C}^*$:

$$\mathbb{P}(\mathcal{C} = \mathcal{C}^* \mid X_1, \dots, X_n) \xrightarrow{p} 1 \quad \text{as } n \to \infty$$

### Corollary (Expected Misclassification Rate Upper Bound)
The expected misclassification rate $\mathbb{E}[R_n]$ under BSF is upper-bounded by:

$$\mathbb{E}[R_n] \le C \exp\left( - \frac{\Delta_n^2}{8\sigma^2} \right)$$
