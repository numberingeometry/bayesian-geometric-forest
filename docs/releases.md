# Release notes

## November 2026: the temperature dial (0.4.0)

A new [workbench page](dial/index.html) turns one dial between a uniformly random tree
and single linkage. Each edge glows with its exact probability under the tree law of a
point cloud, computed live in the browser; points can be dragged, and an Ultrametric
view shows Kirchhoff resistances converging to the single-linkage ultrametric.

`bgforest.TreeLaw` computes the same law in Python through a log-space Kron elimination
that stays exact at temperatures where a dense Laplacian inverse has no correct digits.
`bgforest.ultrametric` adds the single-linkage ultrametric, Brownian-motion tree
covariances and a tree-diffusion data generator. [The temperature dial](temperature.md)
states the mathematics with sources, and seven new rows on the
[verification page](verification.md) check it against enumeration.

## November 2026: Forest Lab

The workbench now connects forest draws, edge probabilities, co-clustering, assignment
disagreement and geometric reference graphs. Eighteen saved fits cover three datasets,
fixed and adaptive scales, and three component weights. Shared views and PNG export
make an experiment easy to revisit or discuss.

The [standard benchmark report](benchmarks.md) records 180 clustering fits across five
seeds and six datasets. A separate 900,000-draw comparison checks Wilson and weighted
Aldous–Broder against the same fixed graph laws. The reports retain measured timings,
failures, protocol metadata and machine-readable results.

The [mathematical reference](theory.md) now explains the rooted-forest law, root
attribution and the exact conditional distribution of the number of trees. Independent
enumeration checks accompany these identities. Equations render from a locally hosted,
pinned MathJax bundle.

This release establishes a reference and experimental hub for existing methods.
