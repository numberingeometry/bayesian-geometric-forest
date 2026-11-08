"""Declared clustering and fixed-kernel sampler protocols; no data downloads."""

# The source checkout and headless backend must be configured before these imports.
# ruff: noqa: E402

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bgforest-matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from bgforest.benchmarks import (
    clustering_datasets,
    compare_clustering,
    compare_samplers,
    environment_metadata,
)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def plot_clustering(rows, output_path=None):
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    colors = ["#94a3b8", "#64748b", "#a78bfa", "#f59e0b", "#22b8a5", "#2563eb"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    width = 0.8 / len(methods)
    for j, method in enumerate(methods):
        for i, dataset in enumerate(datasets):
            selected = [r for r in rows if r["dataset"] == dataset and r["method"] == method and r["status"] == "ok"]
            if not selected:
                continue
            x = i + (j - (len(methods) - 1) / 2) * width
            ari = [r["ari"] for r in selected]
            axes[0].bar(x, np.mean(ari), width * 0.92, color=colors[j], label=method if i == 0 else None)
            axes[0].errorbar(x, np.mean(ari), yerr=np.std(ari), color="#334155", lw=0.6, capsize=1)
            axes[1].bar(x, np.mean([r["runtime_seconds"] for r in selected]), width * 0.92, color=colors[j])
    axes[0].set_ylabel("Adjusted Rand index")
    axes[0].set_ylim(-0.08, 1.06)
    axes[0].legend(ncol=3, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    axes[1].set_ylabel("Fit seconds (log)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(datasets)), datasets, fontsize=9)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    plt.rcParams["svg.fonttype"] = "none"
    svg = Path(output_path) if output_path is not None else ROOT / "figs" / "benchmark.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def write_clustering_summary(rows, config):
    """Keep the human-readable report tied to the actual checked-in run."""
    page = ROOT / "docs" / "benchmarks.md"
    if not page.exists():
        return
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    short = ["k-means", "GMM", "Spectral", "MST cut", "BSF K readout", "BSF K inferred"]
    lines = [f"**Recorded profile: {config['profile']}**; {len(config['seeds'])} seeds; "
             f"{config['synthetic_n']} synthetic observations; "
             f"{config['n_draws']} retained / {config['burn_in']} burn-in sweeps.",
             "", "Mean ARI ± population standard deviation across requested seeds:", "",
             "| Dataset | " + " | ".join(short) + " |",
             "|---|" + "---|" * len(methods)]
    for dataset in datasets:
        cells = []
        for method in methods:
            values = [r["ari"] for r in rows if r["dataset"] == dataset and r["method"] == method and r["status"] == "ok"]
            cells.append(f"{np.mean(values):.3f} ± {np.std(values):.3f}" if values else "failed")
        lines.append("| " + dataset + " | " + " | ".join(cells) + " |")
    failed = sum(r["status"] != "ok" for r in rows)
    lines += ["", f"{len(rows)} requested fits; {failed} failed fits. All run-level scores and timings are in the linked records."]
    start, end = "<!-- measured-results:start -->", "<!-- measured-results:end -->"
    text = page.read_text()
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        page.write_text(head + start + "\n" + "\n".join(lines) + "\n" + end + tail)


def write_sampling_summary(rows, config):
    page = ROOT / "docs" / "benchmarks.md"
    if not page.exists():
        return
    lines = [f"**Recorded profile: {config['profile']}**; {len(rows)} sampler batches, "
             f"{sum(r['n_draws'] for r in rows):,} requested draws, "
             f"{sum(r['status'] != 'ok' for r in rows)} failed batches.", "",
             "Means of marginal RMSE and median batch throughput across the declared graph / root-weight / seed grid:", "",
             "| Sampler | n | Edge RMSE | Root RMSE | Median draws / second |",
             "|---|---:|---:|---:|---:|"]
    for name in ("Wilson", "Aldous–Broder"):
        for n in config["sampler_sizes"]:
            selected = [r for r in rows if r["method"] == name and r["n"] == n and r["status"] == "ok"]
            if selected:
                lines.append(f"| {name} | {n} | "
                             f"{np.mean([r['edge_rmse'] for r in selected]):.4f} | "
                             f"{np.mean([r['root_rmse'] for r in selected]):.4f} | "
                             f"{np.median([r['draws_per_second'] for r in selected]):,.0f} |")
    tv = [r["partition_tv"] for r in rows if "partition_tv" in r]
    if tv:
        lines += ["", f"Maximum empirical partition total variation at n=5: {max(tv):.4f}. "
                  "The raw report retains each configuration; averages do not characterize worst-case walk time. "
                  "Edge RMSE averages over a different number of pairs at each n, so it should not be used to rank accuracy across graph sizes."]
    start, end = "<!-- sampling-results:start -->", "<!-- sampling-results:end -->"
    text = page.read_text()
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        page.write_text(head + start + "\n" + "\n".join(lines) + "\n" + end + tail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "standard"], default="standard")
    parser.add_argument("--only", choices=["all", "clustering", "sampling"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "results")
    args = parser.parse_args()
    quick = args.profile == "quick"
    config = {"profile": args.profile, "seeds": [0, 1, 2] if quick else list(range(5)),
              "synthetic_n": 120 if quick else 300, "n_draws": 80 if quick else 200,
              "burn_in": 80 if quick else 200, "sampler_draws": 1000 if quick else 5000,
              "sampler_sizes": [5, 24] if quick else [5, 32, 96],
              "sampler_q": [0.15, 0.5, 1.5], "threads": 1}
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(), "config": config,
            "environment": environment_metadata(),
            "protocol": "transductive-standardized-v1; same-kernel-sampling-v1"}
    with threadpool_limits(limits=1):
        if args.only in ("all", "clustering"):
            rows = []
            for seed in config["seeds"]:
                for name, (X, truth) in clustering_datasets(seed, config["synthetic_n"]).items():
                    rows.extend(compare_clustering(X, truth, name, seed, config["n_draws"], config["burn_in"]))
                    print(f"clustering seed={seed}: {name}", flush=True)
            (args.output / "clustering.json").write_text(json.dumps({**meta, "rows": rows}, indent=2) + "\n")
            write_csv(args.output / "clustering.csv", rows)
            if args.output.resolve() == (ROOT / "docs" / "results").resolve():
                write_csv(ROOT / "figs" / "benchmark.csv", rows)
                plot_clustering(rows)
                write_clustering_summary(rows, config)
            else:
                write_csv(args.output / "benchmark.csv", rows)
                plot_clustering(rows, args.output / "benchmark.svg")
        if args.only in ("all", "sampling"):
            rows = []
            for seed in config["seeds"]:
                for n in config["sampler_sizes"]:
                    for family in ("path", "geometric"):
                        for q in config["sampler_q"]:
                            rows.extend(compare_samplers(n, family, q, seed, config["sampler_draws"]))
                            print(f"sampling seed={seed}: n={n} {family} q={q}", flush=True)
            (args.output / "sampling.json").write_text(json.dumps({**meta, "rows": rows}, indent=2) + "\n")
            write_csv(args.output / "sampling.csv", rows)
            if args.output.resolve() == (ROOT / "docs" / "results").resolve():
                write_sampling_summary(rows, config)
    print(f"results: {args.output}")


if __name__ == "__main__":
    main()
