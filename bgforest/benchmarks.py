"""Reproducible reference experiments; scores are evidence, not model guarantees.

The clustering protocol is transductive: every method sees all unlabelled points.
The separate sampler protocol compares draws from exactly the same fixed kernel.
"""

import platform
import time
import warnings

import numpy as np
import scipy
import sklearn
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.datasets import load_iris, make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from bgforest.core.forest_model import ForestKernel
from bgforest.exact import canonical, forest_posterior_bruteforce
from bgforest.models.bsf import BayesianSpanningForest
from bgforest.samplers.wilson import (
    WilsonSampler,
    _cumulative_from_log,
    forest_partition,
)


def environment_metadata():
    """Small, portable provenance record; timings depend on this environment."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def explorer_benchmark_report(report):
    """Describe a recorded report without conflating it with interactive fits."""
    config = report["config"]
    return {
        "config": config, "generated_at": report["generated_at"], "rows": report["rows"],
        "protocol": {
            "seeds": f"Separate {config['profile']} benchmark: {len(config['seeds'])} declared seeds, {config['synthetic_n']} synthetic observations (150 Iris), {config['n_draws']} retained BSF draws and {config['burn_in']} burn-in sweeps. Interactive fits above use their own stated data and inference counts.",
            "preprocessing": "Every method sees the same standardized features within each benchmark case. Settings are fixed; no label-driven tuning.",
            "scoring": "ARI and AMI score reference groups. Uniform outliers are fitted but excluded from scoring. Iris uses one fixed dataset with repeated algorithm seeds.",
            "k": "All except BSF K inferred receive reference K for the final decision; BSF posterior draws are not conditioned on that K.",
            "runtime": "Single-thread elapsed fit times include inference and readout, excluding shared preprocessing. Every requested run is retained in the downloadable report.",
        },
    }


def clustering_datasets(seed=0, n_samples=300):
    """Public sklearn datasets and explicitly seeded synthetic stress cases."""
    n = int(n_samples)
    if n < 40:
        raise ValueError("n_samples must be at least 40.")
    rng = np.random.default_rng(100 + seed)
    Xb, yb = make_blobs(
        n, centers=[[0, 0], [3, 0], [1.5, 2.6]],
        cluster_std=[0.5, 0.9, 0.6], random_state=seed,
    )
    n_out = max(1, int(n * 0.05))
    Xm, ym = make_moons(n - n_out, noise=0.08, random_state=seed)
    t = np.sqrt(rng.uniform(0.08, 1.0, n // 2)) * 2.6 * np.pi
    spiral = np.vstack([
        np.c_[t * np.cos(t + k * np.pi), t * np.sin(t + k * np.pi)]
        + rng.normal(0, 0.22, (len(t), 2)) for k in range(2)
    ])
    iris = load_iris()
    return {
        "Two moons": make_moons(n, noise=0.08, random_state=seed),
        "Moons + 5% outliers": (
            np.vstack([Xm, rng.uniform([-1.5, -1.0], [2.5, 1.5], (n_out, 2))]),
            np.r_[ym, np.full(n_out, -1)],
        ),
        "Circles": make_circles(n, noise=0.05, factor=0.5, random_state=seed),
        "Stretched blobs": (Xb @ np.array([[0.6, -0.64], [-0.41, 0.85]]), yb),
        "Spirals": (spiral, np.repeat([0, 1], len(t))),
        "Iris (real)": (iris.data, iris.target),
    }


def clustering_methods(k, seed, n_draws=200, burn_in=200):
    """Identical settings across datasets; no label-based parameter search."""
    return {
        "k-means": KMeans(k, n_init=10, random_state=seed),
        "Gaussian mixture": GaussianMixture(k, n_init=3, random_state=seed),
        "Spectral (10-NN)": SpectralClustering(
            k, affinity="nearest_neighbors", n_neighbors=10,
            random_state=seed, assign_labels="cluster_qr",
        ),
        "MST cut (single linkage)": AgglomerativeClustering(k, linkage="single"),
        "BSF, K readout": BayesianSpanningForest(
            n_clusters=k, n_draws=n_draws, burn_in=burn_in,
            standardize=False, random_state=seed,
        ),
        "BSF, K inferred": BayesianSpanningForest(
            n_draws=n_draws, burn_in=burn_in, standardize=False, random_state=seed,
        ),
    }


def compare_clustering(X, truth, dataset, seed=0, n_draws=200, burn_in=200):
    """Fit six baselines once each; retain failures and warnings in the record.

    Negative reference labels denote excluded points, never supplied to a model.
    K readout changes the final decision, not the BSF posterior over K.
    """
    X, truth = np.asarray(X), np.asarray(truth)
    score_mask = truth >= 0
    k = len(np.unique(truth[score_mask]))
    start = time.perf_counter()
    Z = StandardScaler().fit_transform(X)
    preprocessing_seconds = time.perf_counter() - start
    rows = []
    for name, model in clustering_methods(k, seed, n_draws, burn_in).items():
        row = {
            "dataset": dataset, "method": name, "seed": int(seed),
            "n_samples": len(X), "n_features": X.shape[1],
            "n_scored": int(score_mask.sum()), "reference_k": k,
            "given_k": name != "BSF, K inferred",
            "preprocessing_seconds": preprocessing_seconds,
            "n_draws": n_draws if isinstance(model, BayesianSpanningForest) else None,
            "burn_in": burn_in if isinstance(model, BayesianSpanningForest) else None,
        }
        start = time.perf_counter()
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                labels = model.fit_predict(Z)
            row.update(
                status="ok", ari=float(adjusted_rand_score(truth[score_mask], labels[score_mask])),
                ami=float(adjusted_mutual_info_score(truth[score_mask], labels[score_mask])),
                k_estimate=int(len(np.unique(labels))),
                warnings=list(dict.fromkeys(str(w.message) for w in captured)),
            )
            if isinstance(model, BayesianSpanningForest):
                row["posterior_mean_k"] = float(np.mean(model.n_clusters_draws_))
                row["posterior_mode_k"] = int(max(model.n_clusters_posterior_, key=model.n_clusters_posterior_.get))
                row["pair_brier"] = pair_brier(model.co_clustering_matrix_, truth, score_mask)
        except Exception as exc:  # preserve failed runs instead of selecting successful seeds
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
        row["runtime_seconds"] = time.perf_counter() - start
        rows.append(row)
    return rows


def pair_brier(probability, truth, mask=None):
    """Descriptive pairwise probability score; no independent test-pair claim."""
    truth = np.asarray(truth)
    ids = np.flatnonzero(truth >= 0 if mask is None else mask)
    i, j = np.triu_indices(len(ids), 1)
    target = truth[ids[i]] == truth[ids[j]]
    return float(np.mean((probability[ids[i], ids[j]] - target) ** 2))


def aldous_broder_forests(kernel, n_draws, random_state=0, max_steps=2_000_000):
    """Reference weighted cover-walk sampler on the augmented undirected graph.

    First-entry edges point toward the previously visited vertex. A per-draw step
    budget turns pathological walk times into a recorded failure, not a hang.
    """
    rng = np.random.default_rng(random_state)
    cum = _cumulative_from_log(kernel.augmented_log_weights())
    n = kernel.n
    parents = np.full((n_draws, n), -1, dtype=int)
    total_steps = 0
    for d in range(n_draws):
        seen = np.zeros(n + 1, dtype=bool)
        seen[n] = True
        u, visited, steps = n, 1, 0
        while visited < n + 1:
            v = min(int(np.searchsorted(cum[u], rng.random(), side="right")), n)
            if not seen[v]:
                seen[v] = True
                visited += 1
                if v != n:
                    parents[d, v] = -1 if u == n else u
            u = v
            steps += 1
            if steps > max_steps:
                raise RuntimeError(f"cover walk exceeded {max_steps} steps in draw {d}")
        total_steps += steps
    return parents, total_steps


def sampling_kernel(n, family, q):
    """A graph-law benchmark, distinct from the Gaussian/Cauchy BSF model.

    P(F,R) proportional to q**K product(w_e), one root per component.
    """
    if family == "path":
        W = np.zeros((n, n))
        for i in range(n - 1):
            W[i, i + 1] = W[i + 1, i] = 1.0
    elif family == "geometric":
        x = np.linspace(0, 1, n)[:, None]
        W = np.exp(-cdist(x, x, "sqeuclidean") / (2 * 0.18**2))
        np.fill_diagonal(W, 0)
    else:
        raise ValueError("family must be path or geometric")
    with np.errstate(divide="ignore"):
        return ForestKernel(np.log(W), np.full(n, np.log(q)))


def compare_samplers(n=5, family="geometric", q=0.5, seed=0, n_draws=2000):
    """Compare equal-draw batches under one fixed kernel and exact identities."""
    kernel = sampling_kernel(n, family, q)
    exact_edge, exact_root = kernel.edge_inclusion()
    exact_parts = forest_posterior_bruteforce(kernel)[0] if n <= 5 else None
    rows = []
    for name in ("Wilson", "Aldous–Broder"):
        row = {"method": name, "n": n, "family": family, "q": q,
               "seed": seed, "n_draws": n_draws, "enumerated": n <= 5}
        start = time.perf_counter()
        try:
            if name == "Wilson":
                draws = WilsonSampler(seed).sample_forests(kernel, n_draws)
            else:
                draws, row["walk_steps"] = aldous_broder_forests(kernel, n_draws, seed)
            row["runtime_seconds"] = time.perf_counter() - start
            edge, root = np.zeros((n, n)), np.zeros(n)
            counts = {}
            for parent in draws:
                child = np.flatnonzero(parent >= 0)
                edge[child, parent[child]] += 1
                root += parent < 0
                key = canonical(forest_partition(parent))
                counts[key] = counts.get(key, 0) + 1
            edge = (edge + edge.T) / n_draws
            root /= n_draws
            upper = np.triu_indices(n, 1)
            row.update(
                status="ok", edge_rmse=float(np.sqrt(np.mean((edge[upper] - exact_edge[upper]) ** 2))),
                root_rmse=float(np.sqrt(np.mean((root - exact_root) ** 2))),
                expected_k=float(exact_root.sum()), sampled_mean_k=float(root.sum()),
                draws_per_second=n_draws / row["runtime_seconds"],
                dense_input_bytes=int((n + 1)**2 * 8),
            )
            if exact_parts is not None:
                row["partition_tv"] = float(0.5 * sum(
                    abs(counts.get(p, 0) / n_draws - prob) for p, prob in exact_parts.items()
                ))
        except Exception as exc:
            row.update(status="error", runtime_seconds=time.perf_counter() - start,
                       error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows
