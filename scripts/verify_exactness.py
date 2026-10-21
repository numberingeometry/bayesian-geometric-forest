"""verify_exactness.py -- check every exactness claim of bgforest against brute force.

    python scripts/verify_exactness.py            # print the table, exit 1 on any failure
    python scripts/verify_exactness.py --write    # also refresh docs/verification-results.md

Each check compares a fast computation (matrix-tree theorem, closed-form edge
probabilities, Wilson's algorithm, rejection under constraints) with an independent
answer obtained by listing every labelled tree on the augmented graph (Pruefer codes)
or every set partition. Nothing here uses a tolerance tuned to make it pass: exact
identities are held to 1e-10, and Monte Carlo checks use a chi-square test at
alpha = 0.001 with seeds fixed in advance.

The last block illustrates Theorem 1 of Duan & Roy (JASA 2024) numerically; it is an
illustration of a published theorem, not a test of it.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from scipy.stats import chisquare

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bgforest.core.forest_model import ForestKernel, local_scales  # noqa: E402
from bgforest.exact import (  # noqa: E402
    all_set_partitions,
    canonical,
    forest_posterior_bruteforce,
    log_tree_weight_sum_bruteforce,
)
from bgforest.models.constrained import ConstrainedBayesianSpanningForest  # noqa: E402
from bgforest.samplers.wilson import WilsonSampler, forest_partition  # noqa: E402

ALPHA = 1e-3
EXACT_TOL = 1e-10
rows = []


def record(check, method, result, passed, label=None):
    status = label or ("PASS" if passed else "FAIL")
    rows.append((check, method, result, status))
    print(f"[{status}] {check}: {result}")


def example_kernel(n, seed, lam=0.5):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 0.45, (n // 2, 2)), rng.normal(1.6, 0.45, (n - n // 2, 2))])
    return ForestKernel.from_data(X, local_scales(X, 2), rho=1.0, lam=lam)


def chi2(observed, expected, n):
    keys = [k for k, p in expected.items() if p * n >= 5]
    obs = np.array([observed.get(k, 0) for k in keys], dtype=float)
    exp = np.array([expected[k] * n for k in keys])
    rest_o, rest_e = n - obs.sum(), n - exp.sum()
    if rest_e >= 5:
        obs, exp = np.append(obs, rest_o), np.append(exp, rest_e)
    p = chisquare(obs, exp * obs.sum() / exp.sum()).pvalue
    tv = 0.5 * sum(abs(observed.get(k, 0) / n - v) for k, v in expected.items())
    tv += 0.5 * sum(c / n for k, c in observed.items() if k not in expected)
    return p, tv, len(keys)


def main(write: bool) -> int:
    t0 = time.time()

    # 1. Kirchhoff's matrix-tree theorem.
    rng = np.random.default_rng(1)
    worst = 0.0
    for m in range(2, 8):
        W = rng.uniform(0.05, 3.0, (m, m))
        W = (W + W.T) / 2
        logW = np.log(W)
        np.fill_diagonal(logW, -np.inf)
        k = ForestKernel(logW, np.zeros(m))
        worst = max(worst, abs(k.log_tree_weight_sum(np.arange(m)) - log_tree_weight_sum_bruteforce(logW)))
    record("Matrix-tree theorem", "log tau via one determinant vs sum over all m^(m-2) trees, m = 2..7",
           f"max abs error {worst:.1e} (log scale)", worst < EXACT_TOL)

    # 2-3. Marginal partition posterior and closed-form edge probabilities.
    n = 6
    kernel = example_kernel(n, seed=0)
    exact, M_exact, root_exact = forest_posterior_bruteforce(kernel)
    parts = list(all_set_partitions(n))
    log_m = np.array([kernel.log_marginal(np.array(p)) for p in parts])
    prob = np.exp(log_m - logsumexp(log_m))
    err = max(abs(q - exact.get(p, 0.0)) for p, q in zip(parts, prob))
    record("Partition posterior", f"closed form vs all 7^5 = 16,807 augmented trees, all {len(parts)} partitions of n = 6",
           f"max abs error {err:.1e}", err < EXACT_TOL)
    M, roots = kernel.edge_inclusion()
    err = max(np.abs(M - M_exact).max(), np.abs(roots - root_exact).max())
    record("Edge probabilities", "A_ij R_eff(i, j) vs enumeration, incl. root edges",
           f"max abs error {err:.1e}", err < EXACT_TOL)

    # 4. Wilson's algorithm draws the exact posterior.
    draws = 100_000
    parents = WilsonSampler(2026).sample_forests(kernel, draws)
    observed = {}
    for p in parents:
        key = canonical(forest_partition(p))
        observed[key] = observed.get(key, 0) + 1
    pval, tv, cells = chi2(observed, exact, draws)
    record("Exact forest draws", f"{draws:,} Wilson draws vs enumerated posterior over {cells} partitions",
           f"chi-square p = {pval:.3f}, TV = {tv:.4f}", pval > ALPHA)

    # 5. Constraints by rejection give the exact conditional posterior.
    must, cannot = [(0, 1)], [(2, 5)]
    ok = {k: v for k, v in exact.items() if k[0] == k[1] and k[2] != k[5]}
    pr_c = sum(ok.values())
    expected = {k: v / pr_c for k, v in ok.items()}
    model = ConstrainedBayesianSpanningForest(must_link=must, cannot_link=cannot)
    model._attempts = model._accepted = 0
    draws = 40_000
    parents = model._draw_forests(kernel, WilsonSampler(7), draws)
    observed = {}
    for p in parents:
        key = canonical(forest_partition(p))
        observed[key] = observed.get(key, 0) + 1
    pval, tv, cells = chi2(observed, expected, draws)
    acc = model._accepted / model._attempts
    record("Constrained draws", f"{draws:,} rejection draws vs enumerated conditional posterior ({cells} partitions)",
           f"chi-square p = {pval:.3f}, TV = {tv:.4f}; acceptance {acc:.3f} vs Pr(constraints) {pr_c:.3f}",
           pval > ALPHA and abs(acc - pr_c) < 0.02)

    # 6. Theorem 1 of Duan & Roy (2024), numerically.
    # A global leaf scale (sigma_ij = 1) keeps every A_ij / D_ii small, the regime the
    # theorem's bound covers (its Remark 7); self-tuning local scales do not.
    lines = []
    for n_pts in (60, 120, 240, 480, 960):
        g = np.random.default_rng(n_pts)
        X = np.vstack([g.normal(c, 1.0, (n_pts // 3, 2)) for c in ((0, 0), (2, 2), (4, 4))])
        k3 = ForestKernel.from_data(X, np.ones(len(X)), lam=0.5)
        Wa = np.exp(k3.augmented_log_weights() - k3._shift())
        np.fill_diagonal(Wa, 0.0)
        Ma, ra = k3.edge_inclusion()
        Mfull = np.zeros_like(Wa)
        Mfull[:-1, :-1], Mfull[:-1, -1], Mfull[-1, :-1] = Ma, ra, ra
        d = Wa.sum(axis=1)
        Nsym = np.eye(len(d)) - Wa / np.sqrt(np.outer(d, d))
        U = np.linalg.eigh(Mfull)[1][:, ::-1][:, :3]
        V = np.linalg.eigh(Nsym)[1][:, :3]
        s = np.linalg.svd(U.T @ V, compute_uv=False)
        sin_theta = float(np.sqrt(max(0.0, 3 - np.sum(s**2))))
        lines.append(f"n = {n_pts}: {sin_theta:.3f}")
    record("Duan & Roy Theorem 1 (illustration)",
           "subspace distance (Frobenius sin-theta) between top-3 eigenvectors of M and of -N; 3-component mixture, sigma_ij = 1",
           "; ".join(lines), True, label="shown")

    failed = sum(r[3] == "FAIL" for r in rows)
    checks = sum(r[3] != "shown" for r in rows)
    print(f"\n{checks - failed}/{checks} checks passed in {time.time() - t0:.0f} s")

    if write:
        table = ["| Check | Method | Result | |", "|---|---|---|---|"]
        table += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
        table = "\n".join(table)
        page = ROOT / "docs" / "verification.md"
        page.write_text(PAGE.format(table=table), encoding="utf-8")
        readme = ROOT / "README.md"
        text = readme.read_text(encoding="utf-8")
        start, end = "<!-- verification:start -->", "<!-- verification:end -->"
        if start in text and end in text:
            head, rest = text.split(start, 1)
            _, tail = rest.split(end, 1)
            readme.write_text(f"{head}{start}\n{table}\n{end}{tail}", encoding="utf-8")
        print(f"wrote {page.relative_to(ROOT)} and the README table")
    return 1 if failed else 0


PAGE = """<!-- generated by scripts/verify_exactness.py --write; edit the script, not this page -->

# Verification

Every exactness claim in this package is checked against an independent answer obtained
by brute force: all $N^{{N-2}}$ labelled trees on the augmented graph (Pruefer codes) or
all set partitions. Exact identities are held to $10^{{-10}}$; sampling checks use a
chi-square test at $\\alpha = 0.001$ with seeds fixed in advance. The same checks run in
continuous integration on every push.

```sh
python scripts/verify_exactness.py
```

{table}

**What the checks establish.** The matrix-tree marginal, the closed-form edge and root
probabilities, and the Wilson sampler implement the Bayesian spanning forest posterior
exactly for a given kernel, and rejection under constraints draws the exact conditional
posterior. They do not certify the mixing of the adaptive-scale Gibbs sampler; for that,
inspect `n_clusters_trace_`, `root_scale_trace_` and `mean_scale_trace_`, or run several
seeds.

The last row illustrates Theorem 1 of Duan & Roy (2024): the leading eigenvectors of the
edge-inclusion matrix $M$ stay within a small subspace distance of those used by
normalized spectral clustering. It is an illustration of a published theorem, not a test.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    sys.exit(main(ap.parse_args().write))
