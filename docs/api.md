# API

```python
from bgforest import (
    TreeLaw,                           # the tempered tree law of a point cloud
    BayesianSpanningForest,            # the clustering model
    ConstrainedBayesianSpanningForest, # ... conditioned on must-link / cannot-link pairs
    ForestKernel,                      # augmented-graph weights and closed forms
    SpanningTreeBackbone,              # tree backbone of a feature dependence graph
    WilsonSampler,                     # exact weighted spanning trees and forests
)
```

## `TreeLaw`

The law $\Pr(T) \propto \exp(-\lambda \sum_{e \in T} \lVert x_i - x_j \rVert^2)$ over the
spanning trees of a point cloud: the posterior over trees for Gaussian diffusion along an
unknown tree with step variance $1/(2\lambda)$. See [the temperature dial](temperature.md).

```python
from bgforest import TreeLaw
from bgforest.treelaw import lam_from_scale

law = TreeLaw(X, lam=lam_from_scale(TreeLaw(X, 1.0).costs, kappa=10.0))
```

| Method | Returns |
|---|---|
| `edge_probability()` | $(n, n)$ matrix of $\Pr(e \in T) = w_e R_{\mathrm{eff}}(e)$, exact at any temperature |
| `log_partition()` | $\log Z$ (matrix-tree theorem) |
| `log_resistance()` | $(n, n)$ log effective resistances |
| `expected_length()` | $\mathbb E\, S_T$ |
| `free_energy()`, `free_energy_grad()` | soft MST length $-\lambda^{-1}\log Z$ and its gradient $2L_\pi X$ |
| `mst()`, `mst_overlap()` | the $\lambda \to \infty$ limit and the expected fraction of tree edges it shares |
| `trace_error()` | $\lvert\sum_e \pi_e - (n-1)\rvert/(n-1)$, a built-in accuracy gate |
| `sample(n_trees, random_state)` | exact Wilson draws, as parent arrays |

`TreeLaw.from_costs(C, lam)` accepts any symmetric cost matrix, with `inf` for a missing
edge. `bgforest.ultrametric` adds `single_linkage_ultrametric`,
`log_resistance_ultrametric`, `tree_covariance`, `tree_diffusion` and `cantor_dust`.

## `BayesianSpanningForest`

scikit-learn style: `fit(X)`, `fit_predict(X)`, `predict_proba()`.

| Parameter | Default | Meaning |
|---|---|---|
| `n_clusters` | `None` | $K$ for the point estimate; the posterior is over all $K$ and the default uses its mode |
| `scales` | `"adaptive"` | `"adaptive"` (the paper's hierarchy, Gibbs), `"local"` (fixed $k$-NN distance: exact independent draws), or an array |
| `n_neighbors` | `7` | neighbour rank for local scales and for initialising adaptive ones |
| `lam` | `0.5` | prior weight per cluster, $\Pi(\mathcal T) \propto \lambda^K$ |
| `root_scale` | `1.0` | Cauchy root scale (fixed scales) or its start value (adaptive) |
| `n_draws`, `burn_in` | `200`, `200` | retained draws; burn-in is ignored for fixed scales |
| `standardize` | `True` | standardise features first, as the paper assumes |

| Attribute | Shape | Meaning |
|---|---|---|
| `labels_` | $(n,)$ | point estimate |
| `partitions_`, `forests_` | $(D, n)$ | posterior draws: cluster labels and parent pointers ($-1$ marks a root) |
| `n_clusters_posterior_` | dict | Monte Carlo estimate of the posterior over $K$ |
| `co_clustering_matrix_` | $(n, n)$ | Monte Carlo estimate of $\Pr(i, j \text{ share a cluster} \mid y)$ |
| `edge_inclusion_`, `root_probability_` | $(n, n)$, $(n,)$ | edge and root probabilities (closed form for fixed scales) |
| `n_clusters_trace_`, `root_scale_trace_`, `mean_scale_trace_` | $(\text{sweeps},)$ | chain diagnostics (adaptive scales) |

`assignment_uncertainty()` returns, per point, one minus its average co-clustering
probability with the other members of its estimated cluster.

## `ConstrainedBayesianSpanningForest`

Same parameters plus `must_link`, `cannot_link` (lists of index pairs) and
`max_attempts`. Draws that violate a constraint are rejected, so every retained draw is
an exact draw from the constraint-conditioned law for the current fixed kernel.
Adaptive mode remains a correlated Gibbs chain. `acceptance_rate_` records accepted
draws divided by proposal attempts; for a single fixed kernel it estimates the
unconstrained probability of satisfying the constraints. Contradictory constraint sets
raise `ValueError`.

## `ForestKernel`

`ForestKernel.from_data(X, scales, rho=1.0, lam=0.5)` builds the augmented graph.
Methods: `log_marginal(partition)` (unnormalised log posterior of a partition, trees
summed out), `edge_inclusion()` (returns $M$ and the root probabilities),
`expected_n_clusters()`, `log_tree_weight_sum(nodes)`.

`root_count_distribution()` returns an array of length $n+1$ with exact fixed-kernel
$\Pr(K=k)$ at index $k$. `root_assignment()` returns the $(n,n)$ matrix whose $(i,j)$
entry is the probability that node $i$'s component is rooted at $j$. This matrix is not
the co-clustering matrix. Both methods assume a fixed unconstrained kernel; applying
them to an adaptive fit's posterior-mean-scale kernel does not integrate the posterior.

For a general graph law, construct `ForestKernel(log_leaf, log_root)` directly using
symmetric log leaf weights (diagonal `-np.inf`) and finite log root weights. Constant
root weight $q$ means `log_root=np.full(n, np.log(q))`, a separate family from Cauchy
observation-dependent roots. [Target-law details](theory.md#rooted-graph-laws-counts-and-attribution).

## `WilsonSampler`

`sample_forests(kernel, n_draws)` (exact posterior forests for a fixed kernel),
`sample_spanning_tree(W, root=None)`, `sample_spanning_forest(W, roots)`.

## `SpanningTreeBackbone`

`fit(X)` on an $(n, p)$ matrix gives `backbone_edges_` (the minimum-dissimilarity
spanning tree on $1 - |r|$; the Chow–Liu tree for Gaussian data), `edge_probability_`
(exact spanning-tree inclusion probabilities on the kernel $e^{-(1 - |r|)/h}$), and
`sample_trees(k)`.

## `bgforest.exact`

Brute-force enumeration for small graphs (all labelled trees, all set partitions,
the exact forest posterior). It exists to check everything above.
