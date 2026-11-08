# Benchmark protocols

A useful comparison identifies the statistical target, the information each method gets,
and the computational budget. The clustering and sampling experiments answer different
questions. Neither is a claim of universal superiority or a new method.

[Clustering JSON](results/clustering.json) · [Clustering CSV](results/clustering.csv) ·
[Sampling JSON](results/sampling.json) · [Sampling CSV](results/sampling.csv)

## Clustering: shape, noise, and the cost of a decision

The standard profile uses **five seeds**, **300 synthetic observations** per case
(Iris always has 150), **200 retained BSF draws**, and **200 adaptive burn-in sweeps**.
Six fixed methods are evaluated on two moons, moons with 5% uniform outliers, concentric
circles, stretched Gaussian groups, two spirals, and Fisher's four-feature Iris data.
The quick profile uses three seeds, 120 synthetic observations, and 80/80 retained/burn-in
sweeps. The JSON `config.profile` states which report is currently checked in.

| Method | Information supplied | Fixed settings |
|---|---|---|
| k-means | Reference $K$ | 10 starts |
| Gaussian mixture | Reference $K$ | Full covariance, 3 starts |
| Spectral clustering | Reference $K$ | Union 10-NN affinity, `cluster_qr` readout |
| MST cut / single linkage | Reference $K$ | Euclidean single linkage |
| BSF, $K$ readout | Reference $K$ for the final decision | Adaptive scales, $\lambda=0.5$ |
| BSF, $K$ inferred | No reference $K$ | Same adaptive posterior, mode-$K$ readout |

Every method sees the **same standardized feature matrix**. Reference labels are used only
to supply $K$ where declared and to score the final result; no model sees the individual
labels. There is no label-driven hyperparameter search. This fixed-setting comparison
is not a demonstration that each baseline has been optimally tuned.

Supplying `n_clusters=k` to BSF only changes its spectral + k-means point estimate. It does
not restrict the sampled posterior to $K=k$. The supplied-$K$ and inferred-$K$ rows therefore
measure different decisions under the same forest law and must be read separately.

### Metrics and exclusions

- **Adjusted Rand index (ARI)** and **adjusted mutual information (AMI)** compare the
  fitted partition with reference groups. Higher is better; ARI can be negative.
- **Pairwise Brier score** for BSF compares the estimated co-clustering probability with
  reference pair membership. Lower is better. Pairs share observations and this is a
  descriptive transductive score, not a held-out calibration guarantee.
- **Fit seconds** include each model's inference and final readout, excluding common
  preprocessing. Individual preprocessing time is recorded. Single-thread limits reduce
  hidden parallelism differences; timings remain machine and implementation dependent.
- **Estimated $K$**, posterior mean/mode $K$, every warning and failed run are retained.
  The chart shows the mean and standard deviation across all successful requested seeds;
  failures remain in the downloadable records rather than being replaced by another seed.

In the outlier case every observation is clustered, but the 5% uniform outliers have no
reference cluster and are excluded from both ARI and AMI scoring. `n_scored` makes that
exclusion explicit. The source generates all synthetic data from fixed seeds; Iris comes
from `sklearn.datasets.load_iris`, without a network download.

![Clustering agreement and elapsed fit time](figs/benchmark.svg)

<!-- measured-results:start -->
**Recorded profile: standard**; 5 seeds; 300 synthetic observations; 200 retained / 200 burn-in sweeps.

Mean ARI ± population standard deviation across requested seeds:

| Dataset | k-means | GMM | Spectral | MST cut | BSF K readout | BSF K inferred |
|---|---|---|---|---|---|---|
| Two moons | 0.452 ± 0.025 | 0.466 ± 0.022 | 0.777 ± 0.155 | 1.000 ± 0.000 | 0.989 ± 0.010 | 0.989 ± 0.010 |
| Moons + 5% outliers | 0.484 ± 0.017 | 0.506 ± 0.015 | 0.525 ± 0.202 | 0.000 ± 0.000 | 0.997 ± 0.006 | 0.551 ± 0.142 |
| Circles | -0.003 ± 0.000 | -0.003 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Stretched blobs | 0.442 ± 0.038 | 0.856 ± 0.135 | 0.457 ± 0.117 | 0.000 ± 0.000 | 0.528 ± 0.144 | 0.087 ± 0.173 |
| Spirals | 0.053 ± 0.040 | 0.071 ± 0.058 | 0.738 ± 0.366 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.563 ± 0.091 |
| Iris (real) | 0.618 ± 0.004 | 0.904 ± 0.000 | 0.476 ± 0.000 | 0.558 ± 0.000 | 0.604 ± 0.044 | 0.580 ± 0.036 |

180 requested fits; 0 failed fits. All run-level scores and timings are in the linked records.
<!-- measured-results:end -->

The workbench's comparison panel embeds this **standard repeated-run report**. Its six
benchmark cases are separate from the interactive fits above the table, which use
180 synthetic / 150 Iris observations with 120 draws and 120 burn-in sweeps. The data
bundle also preserves one-seed baseline fits on those displayed observations under
`illustrative_benchmarks`; they are not used for the repeated-run table. Geometric views are descriptive:
a k-NN graph, MST, or spectral embedding is not evidence that a latent manifold has been
recovered.

## Sampling: same law, two exact constructions

The graph benchmark uses a symmetric weighted path or one-dimensional Gaussian geometric
graph, with **constant positive root weight $q$**. It compares Wilson's loop-erased walk
with a direct weighted Aldous–Broder first-entry cover walk on the same augmented graph.
This is a rooted graph law, not a fit of the Duan–Roy Gaussian/Cauchy observation model:

$$
\Pr(F,R)=\frac{q^{|R|}\prod_{e\in F}w_e}{\det(L+qI)}.
$$

There is one selected root in each component. After roots are marginalized, the law of an
unrooted edge set includes $\prod_C|C|$; it is not just $q^K\prod_e w_e$.
The standard grid uses $n\in\{5,32,96\}$, $q\in\{0.15,0.5,1.5\}$, five seeds, and
5,000 independent draws per sampler/configuration. The quick grid uses $n\in\{5,24\}$,
three seeds and 1,000 draws.

For $n=5$, enumerating all $6^4=1,296$ augmented labelled trees gives the entire partition
law. Larger graphs use exact edge and root marginals from the grounded Laplacian.
The report records:

- edge and root marginal RMSE against their exact targets;
- sampled versus exact mean $K$;
- partition total variation from the enumerated target when $n=5$;
- elapsed batch time and draws per second, including each sampler's transition
  preprocessing but excluding construction of the shared kernel and exact reference;
- cover-walk step counts, status, and the dense input's byte size (not peak memory).

Finite samples have nonzero RMSE and total variation even for a correct exact sampler.
Both samplers generate independent draws at a fixed kernel; this experiment does not test
mixing of the adaptive-scale Gibbs chain. The reference cover walk has a per-draw step
budget; exceeding it records an error. Wilson's current implementation has no walk budget.
The grid avoids extreme conditioning and is not a comprehensive bottleneck stress test.

## Run and extend

<!-- sampling-results:start -->
**Recorded profile: standard**; 180 sampler batches, 900,000 requested draws, 0 failed batches.

Means of marginal RMSE and median batch throughput across the declared graph / root-weight / seed grid:

| Sampler | n | Edge RMSE | Root RMSE | Median draws / second |
|---|---:|---:|---:|---:|
| Wilson | 5 | 0.0040 | 0.0067 | 25,944 |
| Wilson | 32 | 0.0024 | 0.0052 | 22,036 |
| Wilson | 96 | 0.0014 | 0.0044 | 15,659 |
| Aldous–Broder | 5 | 0.0039 | 0.0076 | 48,001 |
| Aldous–Broder | 32 | 0.0025 | 0.0056 | 4,331 |
| Aldous–Broder | 96 | 0.0014 | 0.0044 | 1,175 |

Maximum empirical partition total variation at n=5: 0.0331. The raw report retains each configuration; averages do not characterize worst-case walk time. Edge RMSE averages over a different number of pairs at each n, so it should not be used to rank accuracy across graph sizes.
<!-- sampling-results:end -->

```bash
python -m pip install -e ".[dev,docs]"
python scripts/run_benchmarks.py --profile standard
python scripts/run_benchmarks.py --profile quick --only sampling --output /tmp/bgf-smoke
python examples/07_explorer.py
```

The runner records versions, platform, requested seeds, sample counts, and UTC generation
time. Set thread limits or use the runner's one-thread context when comparing timings.
CSV files are flat records; JSON also preserves protocol metadata. Running the standard
profile with the default output replaces the published result files and figure. Use
`--output` for a separate result directory: its clustering figure and flat compatibility
CSV are written there as `benchmark.svg` and `benchmark.csv`, alongside the protocol
records. A custom output leaves the canonical figures and documentation unchanged.

## Sources

[Duan & Roy, JASA 2024](https://arxiv.org/abs/2202.00493) supplies the clustering model;
[Wilson, STOC 1996](https://doi.org/10.1145/237814.237880) and
[Aldous, SIAM J. Discrete Math. 1990](https://doi.org/10.1137/0403039) /
[Broder, FOCS 1989](https://doi.org/10.1109/SFCS.1989.63516) supply the sampling constructions.
[Chebotarev & Shamis](https://arxiv.org/abs/math/0602070) and
[Avena & Gaudillière](https://arxiv.org/abs/1310.1723) give matrix-forest and rooted-forest
background. See the [mathematical reference](theory.md) for assumptions and distinctions.
