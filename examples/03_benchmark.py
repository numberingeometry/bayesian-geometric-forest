"""Benchmark against standard clustering methods, with the same K and seeds for all.

    python examples/03_benchmark.py          # ~10 minutes; writes figs/benchmark.svg and .csv

Every method sees standardised features. All methods except "BSF, K inferred" are
told the true number of clusters; single linkage with K clusters is the same as
cutting the K - 1 longest edges of the minimum spanning tree, the deterministic
counterpart of a spanning-forest model. Scores are adjusted Rand index against the
generating labels (Iris: the species), mean and spread over five seeds. In "Moons + 5%
outliers" every method clusters all points, and only the moon points are scored.
"""

import csv
import sys
import time

import numpy as np
from _common import FIGS, ROOT, benchmark_datasets, save

sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from bgforest import BayesianSpanningForest  # noqa: E402

SEEDS = range(5)
METHODS = {
    "k-means": lambda k, s: KMeans(k, n_init=10, random_state=s),
    "Gaussian mixture": lambda k, s: GaussianMixture(k, n_init=3, random_state=s),
    "Spectral (k-NN graph)": lambda k, s: SpectralClustering(
        k, affinity="nearest_neighbors", n_neighbors=10, random_state=s, assign_labels="cluster_qr"
    ),
    "MST cut (single linkage)": lambda k, s: AgglomerativeClustering(k, linkage="single"),
    "BSF, K given": lambda k, s: BayesianSpanningForest(n_clusters=k, n_draws=200, burn_in=200, random_state=s),
    "BSF, K inferred": lambda k, s: BayesianSpanningForest(n_draws=200, burn_in=200, random_state=s),
}

rows = []
t0 = time.time()
for seed in SEEDS:
    for name, (X, y) in benchmark_datasets(seed).items():
        Z = StandardScaler().fit_transform(X)
        inlier = y >= 0  # outliers (label -1) are clustered but not scored
        k = len(np.unique(y[inlier]))
        for method, make in METHODS.items():
            labels = make(k, seed).fit_predict(Z)
            rows.append((name, method, seed, adjusted_rand_score(y[inlier], labels[inlier])))
        print(f"seed {seed} {name}: done ({time.time() - t0:.0f}s)", flush=True)

with open(FIGS / "benchmark.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "method", "seed", "ari"])
    w.writerows([(d, m, s, f"{a:.4f}") for d, m, s, a in rows])

datasets = list(dict.fromkeys(r[0] for r in rows))
methods = list(METHODS)
colors = ["#9ca3af", "#6b7280", "#a78bfa", "#f59e0b", "#2563eb", "#1e3a8a"]
fig, ax = plt.subplots(figsize=(11.5, 3.6))
width = 0.8 / len(methods)
for j, method in enumerate(methods):
    for i, d in enumerate(datasets):
        vals = np.array([a for (dd, m, _, a) in rows if dd == d and m == method])
        x = i + (j - (len(methods) - 1) / 2) * width
        ax.bar(x, vals.mean(), width * 0.9, color=colors[j], label=method if i == 0 else None)
        if vals.std() > 1e-6:
            ax.errorbar(x, vals.mean(), yerr=vals.std(), color="#111827", linewidth=0.8, capsize=1.5)
ax.set_xticks(range(len(datasets)), datasets)
ax.set_ylabel("Adjusted Rand index")
ax.set_ylim(0, 1.05)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(ncol=3, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
fig.tight_layout()
save(fig, "benchmark.svg")

print("\nmean ARI (sd) over seeds")
for d in datasets:
    cells = []
    for m in methods:
        v = np.array([a for (dd, mm, _, a) in rows if dd == d and mm == m])
        cells.append(f"{m}: {v.mean():.3f} ({v.std():.3f})")
    print(f"{d}: " + " | ".join(cells))
