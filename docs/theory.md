# Theory notes

This page separates two things: what the cited papers establish, and what this
package implements. Statements about the papers are paraphrased; consult the papers
for exact assumptions.

## The Bayesian spanning forest model (Duan & Roy, 2024)

Data $y_1, \dots, y_n \in \mathbb{R}^p$ are generated along a spanning forest. Each
cluster $V_k$ is a tree $T_k$ with a root $\rho_k$:

$$
L(y \mid V, F, R) = \prod_{k=1}^{K} r(y_{\rho_k}) \prod_{(i,j) \in T_k} f(y_i \mid y_j),
$$

with a diffuse root density $r$ (multivariate Cauchy in the paper) and a symmetric leaf
density $f$ (Gaussian, with local scales $\sigma_{ij} = \tilde\sigma_i \tilde\sigma_j$).
The edges and roots are latent; the partition $V$ is the parameter of interest.

Adding an auxiliary node $0$ joined to every root turns a rooted forest into one
spanning tree $\mathcal T$ on $n+1$ nodes. With the paper's prior
$\Pi(\mathcal T) \propto \lambda^K$, the posterior is a weighted spanning-tree
distribution:

$$
\Pi(\mathcal T \mid y) \propto \prod_{(i,j) \in \mathcal T} A_{ij},
\qquad A_{ij} = f(y_i \mid y_j), \quad A_{0i} = \lambda\, r(y_i).
$$

Summing over trees with Kirchhoff's matrix-tree theorem gives the marginal posterior
of the partition:

$$
\Pi(V \mid y) \propto \lambda^K \prod_{k=1}^{K}
\Big(\sum_{i \in V_k} r(y_i)\Big)\, \tau(V_k),
\qquad
\tau(V_k) = \sum_{T \text{ spans } V_k} \prod_{(i,j) \in T} A_{ij} = \det L_{V_k}^{(u)},
$$

where $L_{V_k}^{(u)}$ is the Laplacian of $A$ restricted to $V_k$ with any one row and
column removed.

**Results in the papers.**

- *Duan & Roy (2024), Theorem 1.* The leading eigenvectors of the posterior edge-inclusion
  matrix $M_{ij} = \Pr[(i,j) \in \mathcal T \mid y]$ are close to those used by normalized
  spectral clustering, with an explicit perturbation bound. This is the sense in which
  the forest model is a generative counterpart of spectral clustering.
- *Duan & Roy (2024), Theorems 2 and 3.* Posterior ratio consistency for separable
  clusters, under a diminishing kernel scale, a minimum-separation condition and a
  near-flat root density.
- *Zheng, Duan & Roy (arXiv:2409.19129, submitted).* Posterior concentration on the true
  partition when data come from an unknown collection of component distributions under
  a mild asymptotic separation condition, with an upper bound on the expected
  misclassification rate.

## Exact sampling of spanning trees

Wilson's algorithm (1996) draws a spanning tree with probability proportional to the
product of its edge weights, exactly, by running loop-erased random walks. Rooting the
walks at the auxiliary node $0$ therefore yields exact draws from
$\Pi(\mathcal T \mid y)$ for fixed kernel parameters. Duan & Roy use the related
Aldous–Broder covering walk; Tam, Dunson & Duan (2025) develop a fast-forwarded cover
algorithm for graphs with bottlenecks. This package implements Wilson's algorithm.

## Implementation status

The notes above describe the published model. The package's estimators are being
brought in line with it; until that work lands, `BayesianSpanningForest` implements a
simplified fixed-$K$ partition model and should be read as a prototype, not as a
reproduction of Duan & Roy (2024).

## References

- L. L. Duan and A. Roy. Spectral clustering, Bayesian spanning forest, and forest
  process. *Journal of the American Statistical Association*, 119(547):2140–2153, 2024.
  [arXiv:2202.00493](https://arxiv.org/abs/2202.00493)
- Y. Zheng, L. L. Duan and A. Roy. Consistency of graphical model-based clustering:
  robust clustering using Bayesian spanning forest. Preprint,
  [arXiv:2409.19129](https://arxiv.org/abs/2409.19129)
- L. L. Duan and D. B. Dunson. Bayesian spanning tree: estimating the backbone of the
  dependence graph. *Journal of Machine Learning Research*, 24(397):1–44, 2023.
  [arXiv:2106.16120](https://arxiv.org/abs/2106.16120)
- L. L. Duan and D. B. Dunson. Bayesian distance clustering. *Journal of Machine
  Learning Research*, 22(224):1–27, 2021.
  [jmlr.org/papers/v22/20-688](https://jmlr.org/papers/v22/20-688.html)
- E. Tam, D. B. Dunson and L. L. Duan. Exact sampling of spanning trees via
  fast-forwarded random walks. *Biometrika*, 112(2):asaf031, 2025.
  [arXiv:2405.03096](https://arxiv.org/abs/2405.03096)
- D. B. Wilson. Generating random spanning trees more quickly than the cover time.
  *Proceedings of STOC*, 1996.
