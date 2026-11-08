"""Build the self-contained, precomputed research workbench (no network data).

python examples/07_explorer.py                 # release: 120 draws, 120 burn-in
python examples/07_explorer.py --profile quick # development: 40 draws, 40 burn-in

Controls select actual fits. They never interpolate or pretend to run inference
in the browser. All pair summaries use all draws; animation keeps at most 40.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bgforest-matplotlib")
import numpy as np
from _common import ROOT, showcase
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian, minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris, make_circles
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(ROOT))
from bgforest import BayesianSpanningForest  # noqa: E402
from bgforest.benchmarks import (  # noqa: E402
    compare_clustering,
    environment_metadata,
    explorer_benchmark_report,
    pair_brier,
)
from bgforest.diagnostics import compute_effective_sample_size  # noqa: E402


def graph_geometry(X):
    """Descriptive Euclidean graphs, not extra learned forest models."""
    Z = StandardScaler().fit_transform(X)
    dist = cdist(Z, Z)
    neighbour_dist = dist.copy()
    np.fill_diagonal(neighbour_dist, np.inf)
    order = np.argsort(neighbour_dist, axis=1)[:, :10]
    pairs = sorted({tuple(sorted((i, int(j)))) for i, row in enumerate(order) for j in row})
    W = np.zeros_like(dist)
    for i, j in pairs:
        W[i, j] = W[j, i] = 1.0
    _, vec = np.linalg.eigh(laplacian(W, normed=True))
    # Sparse graph APIs use zero as "no edge"; preserve duplicate-point edges.
    mst_dist = np.maximum(dist, 1e-12)
    np.fill_diagonal(mst_dist, 0.0)
    tree = minimum_spanning_tree(csr_matrix(mst_dist)).tocoo()
    return {
        "knn_edges": [list(pair) for pair in pairs],
        "mst_edges": [[int(i), int(j)] for i, j in zip(tree.row, tree.col)],
        "spectral_xy": np.round(vec[:, 1:3], 5).tolist(),
        "note": "Union 10-nearest-neighbour graph and Euclidean MST on standardized features. Spectral coordinates are the second and third normalized-Laplacian eigenvectors; disconnected graphs have a non-unique zero eigenspace. These are descriptive comparisons, not an inferred latent manifold.",
    }


def pack_run(X, truth, k, seed, mode, lam, n_draws, burn_in):
    start = time.perf_counter()
    model = BayesianSpanningForest(
        n_clusters=k, scales=mode, lam=lam, n_draws=n_draws,
        burn_in=burn_in, random_state=seed,
    ).fit(X)
    elapsed = time.perf_counter() - start
    keep = np.linspace(0, n_draws - 1, min(40, n_draws)).astype(int)
    u = model.assignment_uncertainty()
    upper = np.triu_indices(len(X), 1)
    candidates = sorted(
        [(int(i), int(j), round(float(p), 5)) for i, j, p in
         zip(*upper, model.edge_inclusion_[upper]) if p >= 0.01],
        key=lambda edge: -edge[2],
    )
    score_mask = truth >= 0
    mode_k = int(max(model.n_clusters_posterior_, key=model.n_clusters_posterior_.get))
    fixed = mode == "local"
    sizes = np.bincount(model.labels_)
    focus_candidates = np.flatnonzero((sizes[model.labels_] > 5) & (model.root_probability_ < 0.8))
    focus = int(focus_candidates[np.argmax(u[focus_candidates])]) if len(focus_candidates) else int(np.argmax(u))
    result = {
        "id": f"{mode}-lambda-{lam:g}",
        "label": "Fixed local scales" if fixed else "Adaptive scales (Gibbs)",
        "mode": mode, "lam": lam, "seed": seed,
        "scale_note": "Fixed 7th-neighbour leaf scales; exact independent Wilson draws." if fixed else "Duan–Roy hierarchical scale updates; Wilson is exact conditional on each current kernel. Joint draws are correlated, with no convergence guarantee.",
        "n_draws": n_draws, "co_counts_denominator": n_draws,
        "burn_in": 0 if fixed else burn_in, "retained_animation_draws": len(keep),
        "draws": model.forests_[keep].tolist(),
        "partitions": model.partitions_[keep].tolist(),
        "draw_k": model.n_clusters_draws_[keep].tolist(),
        "labels": model.labels_.tolist(), "readout_k": k,
        "uncertainty": np.round(u, 5).tolist(),
        "co_counts": np.rint(model.co_clustering_matrix_ * n_draws).astype(int).tolist(),
        "root_probability": np.round(model.root_probability_, 6).tolist(),
        "edges": candidates[:900],
        "edge_display_note": f"Strongest {min(900, len(candidates))} edges with probability at least 0.01; display filtering does not change the fitted dense model.",
        "edge_probability_kind": "Exact conditional on fixed scales" if fixed else "Monte Carlo over adaptive-scale Gibbs draws",
        "k_posterior": {str(kk): float(p) for kk, p in model.n_clusters_posterior_.items()},
        "k_draws": model.n_clusters_draws_.tolist(),
        "expected_k": float(model.root_probability_.sum()),
        "k_ess": float(n_draws if fixed else compute_effective_sample_size(model.n_clusters_draws_)),
        "k_note": f"Sampled K mode {mode_k}. Labels use a {k}-cluster readout; the posterior is not conditioned on K={k}. Isolated-point trees count toward K.",
        "default_focus": focus,
        "ari": float(adjusted_rand_score(truth[score_mask], model.labels_[score_mask])),
        "pair_brier": pair_brier(model.co_clustering_matrix_, truth, score_mask),
        "n_scored": int(score_mask.sum()), "runtime_seconds": elapsed,
    }
    if fixed:
        result["k_exact"] = np.round(model.kernel_.root_count_distribution(), 8).tolist()
        result["pair_mcse_note"] = "For a pair with estimated probability p, iid Monte Carlo standard error is approximately sqrt(p(1-p)/n_draws)."
    else:
        result["pair_mcse_note"] = "Draws are correlated. The displayed scalar K ESS is a diagnostic, not a pairwise Monte Carlo error estimate or a convergence certificate."
    print(f"{mode} lambda={lam:g}: {elapsed:.2f}s, K mode={mode_k}, ARI={result['ari']:.3f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "release"], default="release")
    parser.add_argument("--benchmark-report", type=str, default=str(ROOT / "docs" / "results" / "clustering.json"))
    args = parser.parse_args()
    quick = args.profile == "quick"
    n, draws, burn = (100, 40, 40) if quick else (180, 120, 120)
    X1, y1 = showcase(n=n)
    X2, y2 = make_circles(n, noise=0.06, factor=0.5, random_state=2)
    iris = load_iris()
    Z3 = StandardScaler().fit_transform(iris.data)
    specifications = [
        ("moons", "Moons, blob, outliers", "Two moons, a compact blob, a short bridge and isolated points. The bridge and outliers are excluded from the reference score, but all points are fitted.", X1, y1, X1, 3, 11,
         "Seeded make_moons plus the fully specified examples/_common.py showcase generator.", "Original 2-D coordinates; fitting uses standardized features."),
        ("circles", "Circles", "Two noisy concentric rings test connectivity-based clustering against centroid and Gaussian models.", X2, y2, X2, 2, 3,
         "scikit-learn make_circles; seed 2, noise 0.06, factor 0.5.", "Original 2-D coordinates; fitting uses standardized features."),
        ("iris", "Iris", "150 flower measurements in four dimensions. Species labels are reference annotations; connectivity need not match species boundaries.", iris.data, iris.target, PCA(2).fit_transform(Z3), 3, 5,
         "Fisher Iris, via sklearn.datasets.load_iris (CC BY 4.0 / UCI source).", "First two principal components of standardized features, for display only; fitting uses all four features."),
    ]
    datasets, rows = [], []
    with threadpool_limits(limits=1):
        for key, name, about, X, truth, xy, k, seed, source, display in specifications:
            print(name, flush=True)
            runs = [pack_run(X, truth, k, seed, mode, lam, draws, burn)
                    for mode in ("adaptive", "local") for lam in (0.15, 0.5, 1.5)]
            default = next(r for r in runs if r["id"] == "adaptive-lambda-0.5")
            data = {"id": key, "name": name, "about": about, "n": len(X),
                    "dimension": X.shape[1], "source": source, "display_note": display,
                    "xy": np.round(xy, 5).tolist(), "truth": truth.tolist(),
                    "default_run": default["id"], "runs": runs, "geometry": graph_geometry(X)}
            # Compatibility fields allow older renderers to inspect the default run.
            data.update(default)
            data["id"] = key
            datasets.append(data)
            baseline = compare_clustering(X, truth, name, seed, draws, burn)
            for row in baseline:
                row["dataset_id"] = key
            rows.extend(baseline)
    payload = {
        "schema_version": 2,
        "meta": {"generated_at": datetime.now(timezone.utc).isoformat(),
                 "profile": args.profile, "n_draws": draws, "burn_in": burn,
                 "environment": environment_metadata(),
                 "provenance": "python examples/07_explorer.py --profile " + args.profile,
                 "notes": "All controls select precomputed fits; all probability summaries use all draws, while animation retains at most 40. Reference labels only set the optional final K readout and evaluate scores."},
        "datasets": datasets,
        "benchmarks": {"protocol": {"name": "Same-data illustrative comparison", "seeds": "One specified seed per dataset; not a repeated-run leaderboard.",
                       "scoring": "Adjusted Rand index and adjusted mutual information; negative reference labels excluded from score, never from fitting.",
                       "preprocessing": "One StandardScaler fit on all unlabelled observations for every method. Six fixed settings; no label-driven hyperparameter search.",
                       "k": "All except BSF K inferred receive the reference K for the final decision. For BSF this does not condition the sampled posterior.",
                       "runtime": "Elapsed fit time including inference and readout, excluding common preprocessing. Machine-specific single-thread timings."}, "rows": rows},
    }
    report_path = ROOT / args.benchmark_report
    if report_path.exists():
        payload["illustrative_benchmarks"] = payload["benchmarks"]
        payload["benchmarks"] = explorer_benchmark_report(json.loads(report_path.read_text()))
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    payload["meta"]["data_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    template = (ROOT / "examples" / "explorer_template.html").read_text(encoding="utf-8")
    out = ROOT / "docs" / "explorer" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(template.replace("/*DATA*/null", encoded), encoding="utf-8")
    (out.parent / "data.json").write_text(encoded + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
