# bayesian-geometric-forest

**Clusters as trees. Forests drawn exactly. Uncertainty you can see.**

![Posterior draws of a Bayesian spanning forest](figs/forest-draws.svg)

`bgforest` implements the Bayesian spanning forest of Duan & Roy (*JASA* 2024). Every
cluster is a tree grown from a root; the trees are summed out, and what remains is a
posterior over partitions, including the number of clusters. That posterior is a weighted
spanning-tree law on an augmented graph, so a whole forest can be drawn exactly with
Wilson's algorithm and edge probabilities have a closed form. Every identity the code
relies on is checked against brute-force enumeration.

[Open the interactive explorer](explorer/index.html){ .md-button .md-button--primary }
[Source on GitHub](https://github.com/numberingeometry/bayesian-geometric-forest){ .md-button }

## In ten lines

```python
from sklearn.datasets import make_moons
from bgforest import BayesianSpanningForest

X, y = make_moons(n_samples=300, noise=0.07, random_state=1)
model = BayesianSpanningForest(random_state=0).fit(X)   # K is inferred

model.n_clusters_posterior_      # posterior over the number of clusters
model.labels_                    # point estimate
model.co_clustering_matrix_      # Pr(i and j share a cluster | data)
model.assignment_uncertainty()   # per-point uncertainty
```

## One posterior, four views

![Data, one posterior forest, co-clustering and uncertainty](figs/hero.svg)

## How it compares

![Benchmark against k-means, Gaussian mixtures, spectral clustering and the MST cut](figs/benchmark.svg)

The forest matches the best graph methods on shape-defined clusters, keeps its accuracy when
5% outliers are added (the minimum-spanning-tree cut collapses), and loses to a Gaussian
mixture on elliptical, overlapping clusters, where the mixture is the right model.

## Spectral clustering, read probabilistically

![Eigenvectors of the edge-probability matrix against normalized spectral clustering](figs/theorem1.svg)

## Where to next

- [Model and math](theory.md): the model, the two closed forms, the Gibbs sampler, and
  exactly where this package follows or departs from the paper.
- [Verification](verification.md): every exactness claim against brute force.
- [API](api.md): parameters and attributes.
