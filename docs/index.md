---
hide:
  - navigation
  - toc
---

<div class="forest-hero" markdown>

<div class="forest-eyebrow">PROBABILITY · GEOMETRY · STRUCTURE</div>

# One tree is a decision.<br>A forest distribution is a question.

Explore how observations connect, where that connection is uncertain, and what changes
when the model changes. Bayesian Geometric Forest is an open research workbench for
probabilistic trees, clustering, and graph geometry, grounded in existing methods and
reproducible experiments.

[Enter the workbench](explorer/index.html){ .md-button .md-button--primary }
[Read the mathematical contract](theory.md){ .md-button }

<div class="forest-caption">18 precomputed fits · 3 public / reproducible datasets · 6 clustering baselines · 2 exact fixed-law samplers</div>

</div>

<div class="grid cards" markdown>

-   **See the structure**

    Trace individual forest draws, inspect roots, and compare the strongest edge probabilities.
    Link a selected observation to its co-clustering row and the complete pair matrix.

-   **Change the assumptions**

    Select fixed local scales or adaptive Gibbs inference, then vary the component weight.
    Every control selects an actual recorded fit with explicit sampler and uncertainty labels.

-   **Compare the evidence**

    Six methods see identical standardized data. Scores, wall times, seeds, warnings, and
    failures accompany the results. Exact small graphs provide an independent sampler check.

</div>

## The reference model

The implemented clustering model is the **Bayesian spanning forest of Duan & Roy
(JASA 2024)**. Each cluster is a rooted tree, with Gaussian leaf densities and a Cauchy
root density. Conditional on the kernel, joining every root to an auxiliary vertex
produces a weighted spanning-tree law, sampled exactly with Wilson's algorithm.

Fixed-kernel draws are independent. Adaptive-scale Gibbs draws are correlated; an exact
conditional forest step does not establish convergence of the whole chain. Supplied $K$
sets the final cluster readout, not the posterior's number of components.

[Model, identities, and departures from the paper](theory.md) · [Finite-state verification](verification.md)

## Reproduce before extending

![Clustering agreement and measured fit time for six reference methods](figs/benchmark.svg)

Curved groups, outliers, elliptical overlap, and unknown component counts test different
assumptions. The report includes the cases where forest clustering loses. A separate
sampler benchmark compares Wilson and weighted Aldous–Broder against the same law;
faster output alone does not establish sampling correctness.

[Read the protocol and results](benchmarks.md){ .md-button }
[Download clustering records](results/clustering.json){ .md-button }
[Download sampler records](results/sampling.json){ .md-button }

## Start with a distribution

```python
from sklearn.datasets import make_moons
from bgforest import BayesianSpanningForest

X, _ = make_moons(200, noise=0.07, random_state=1)
model = BayesianSpanningForest(scales="local", random_state=0).fit(X)

model.co_clustering_matrix_                 # estimated component-sharing probabilities
model.kernel_.root_count_distribution()    # exact P(K=k) for this fixed kernel
model.kernel_.root_assignment()            # root attribution, a different probability
```

```bash
python -m pip install -e ".[dev,docs]"
python scripts/run_benchmarks.py --profile standard
python examples/07_explorer.py
```

## A platform with a defined scope

The current release implements forest clustering, weighted graph laws, exact identities,
constraints by rejection, and descriptive feature backbones. Latent manifold learning,
new density models, tensor-tree ensembles, and model-reliability applications are
[research questions](research.md), with explicit baselines and gates before a new result
is claimed.

[Source and installation](https://github.com/numberingeometry/bayesian-geometric-forest) ·
[API](api.md) · [Methods and citations](theory.md)
