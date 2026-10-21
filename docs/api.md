# API

```python
from bgforest import (
    BayesianSpanningForest,            # the clustering model
    ConstrainedBayesianSpanningForest, # ... conditioned on must-link / cannot-link pairs
    ForestKernel,                      # augmented-graph weights and closed forms
    SpanningTreeBackbone,              # tree backbone of a feature dependence graph
    WilsonSampler,                     # exact weighted spanning trees and forests
)
```

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
| `n_clusters_posterior_` | dict | posterior over $K$ |
| `co_clustering_matrix_` | $(n, n)$ | $\Pr(i, j \text{ share a cluster} \mid y)$ |
| `edge_inclusion_`, `root_probability_` | $(n, n)$, $(n,)$ | edge and root probabilities (closed form for fixed scales) |
| `n_clusters_trace_`, `root_scale_trace_`, `mean_scale_trace_` | $(\text{sweeps},)$ | chain diagnostics (adaptive scales) |

`assignment_uncertainty()` returns, per point, one minus its average co-clustering
probability with the other members of its estimated cluster.

## `ConstrainedBayesianSpanningForest`

Same parameters plus `must_link`, `cannot_link` (lists of index pairs) and
`max_attempts`. Draws that violate a constraint are rejected, so every retained draw is
an exact draw from the conditional posterior. `acceptance_rate_` estimates the posterior
probability of the constraints. Contradictory constraint sets raise `ValueError`.

## `ForestKernel`

`ForestKernel.from_data(X, scales, rho=1.0, lam=0.5)` builds the augmented graph.
Methods: `log_marginal(partition)` (unnormalised log posterior of a partition, trees
summed out), `edge_inclusion()` (returns $M$ and the root probabilities),
`expected_n_clusters()`, `log_tree_weight_sum(nodes)`.

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
