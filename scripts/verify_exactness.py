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
from scipy.special import erf, logsumexp
from scipy.stats import chisquare, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bgforest.core.forest_model import ForestKernel, local_scales  # noqa: E402
from bgforest.exact import (  # noqa: E402
    all_labelled_trees,
    all_set_partitions,
    canonical,
    forest_posterior_bruteforce,
    log_tree_weight_sum_bruteforce,
)
from bgforest.models.constrained import ConstrainedBayesianSpanningForest  # noqa: E402
from bgforest.samplers.wilson import WilsonSampler, forest_partition  # noqa: E402
from bgforest.treelaw import TreeLaw, lam_from_scale, mst_edges, squared_distances  # noqa: E402
from bgforest.ultrametric import single_linkage_ultrametric, tree_covariance  # noqa: E402

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

    count_exact = np.zeros(n + 1)
    for partition, probability in exact.items():
        count_exact[max(partition) + 1] += probability
    err = np.max(np.abs(kernel.root_count_distribution() - count_exact))
    record("Root count distribution", "determinantal Bernoulli convolution vs enumerated partition counts",
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

    tree_law_checks()

    # Theorem 1 of Duan & Roy (2024), numerically.
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


def enumerated_tree_law(C, lam):
    """log Z, edge probabilities, the trees and their probabilities, by listing all trees."""
    n = C.shape[0]
    trees = list(all_labelled_trees(n))
    logw = np.array([-lam * sum(C[i, j] for i, j in t) for t in trees])
    log_z = logsumexp(logw)
    prob = np.exp(logw - log_z)
    P = np.zeros((n, n))
    for t, p in zip(trees, prob):
        for i, j in t:
            P[i, j] += p
            P[j, i] += p
    return log_z, P, trees, prob


def nested_clusters(seed=3):
    """Three clusters of three subclusters: the dial's "clusters within clusters"."""
    g = np.random.default_rng(seed)
    parts = [np.add(c, s) + 0.28 * g.normal(size=(12, 2))
             for c in ((0, 0), (6, 1), (2.5, 5.5)) for s in ((0, 0), (1.6, 0.4), (0.5, 1.5))]
    return np.vstack(parts)


def tree_law_checks():
    """The tempered tree law Pr(T) ~ exp(-lam * sum of squared edge lengths)."""
    # Exact at every temperature.
    worst_pi = worst_z = 0.0
    for n in (4, 5, 6, 7):
        X = np.random.default_rng(10 + n).uniform(size=(n, 2))
        C = squared_distances(X)
        for kappa in (0.0, 0.1, 10.0, 1e3, 1e4):
            lam = lam_from_scale(C, kappa) if kappa else 0.0
            law = TreeLaw(X, lam)
            log_z, P, _, _ = enumerated_tree_law(C, lam)
            worst_pi = max(worst_pi, np.abs(P - law.edge_probability()).max())
            worst_z = max(worst_z, abs(log_z - law.log_partition()) / max(1.0, abs(log_z)))
    record("Tempered tree law", "log-space Kron elimination vs all n^(n-2) trees, n = 4..7, kappa = 0 to 10^4",
           f"edge probabilities {worst_pi:.1e}, log Z relative {worst_z:.1e}", max(worst_pi, worst_z) < EXACT_TOL)

    # Where a dense Laplacian inverse stops working.
    X = nested_clusters()
    C = squared_distances(X)
    n = len(X)
    dense, stable = [], []
    for kappa in (0.1, 1.0, 10.0, 100.0, 1e4):
        lam = lam_from_scale(C, kappa)
        W = np.exp(-lam * (C - C[C > 0].min()))
        np.fill_diagonal(W, 0.0)
        L = np.diag(W.sum(axis=1)) - W
        with np.errstate(all="ignore"):
            G = np.zeros((n, n))
            try:
                G[1:, 1:] = np.linalg.inv(L[1:, 1:])
                g = np.diag(G)
                gate = abs(np.triu(W * (g[:, None] + g[None, :] - 2 * G), 1).sum() - (n - 1)) / (n - 1)
            except np.linalg.LinAlgError:
                gate = np.nan
        dense.append(f"{gate:.0e}" if np.isfinite(gate) else "fails")
        stable.append(TreeLaw(X, lam).trace_error())
    record("Low temperature", "trace gate |sum pi - (n-1)| / (n-1), 108 nested-cluster points, kappa = 0.1, 1, 10, 100, 10^4",
           f"dense inverse {', '.join(dense)}; log-space at most {max(stable):.0e}", max(stable) < EXACT_TOL)

    # Kirchhoff melts into single linkage: 4/n^2 <= R exp(-lam u) <= n - 1 on any connected graph.
    worst = 0.0
    ok = True
    for n, k in ((10, None), (60, None), (60, 5)):
        X = np.random.default_rng(n + (k or 0)).uniform(size=(n, 2))
        C = squared_distances(X)
        if k is not None:  # a k-nearest-neighbour graph, made connected by its MST
            keep = np.zeros_like(C, dtype=bool)
            nn = np.argsort(C, axis=1)[:, 1 : k + 1]
            keep[np.repeat(np.arange(n), k), nn.ravel()] = True
            e = mst_edges(C)
            keep[e[:, 0], e[:, 1]] = True
            keep |= keep.T
            C = np.where(keep, C, np.inf)
        U = single_linkage_ultrametric(np.where(np.isinf(C), 1e300, C))
        off = ~np.eye(n, dtype=bool)
        for kappa in (0.1, 1.0, 10.0, 100.0, 1e3, 1e4):
            lam = lam_from_scale(np.where(np.isinf(C), 1e300, C), kappa)
            excess = (TreeLaw.from_costs(C, lam).log_resistance() - lam * U)[off]
            slack = 1e-12 * lam * U[off].max()
            ok &= bool(excess.min() >= np.log(4 / n**2) - slack and excess.max() <= np.log(n - 1) + slack)
            worst = max(worst, np.abs(excess).max() / (2 * np.log(n)))
    record("Resistance to single linkage", "log R_eff - lam u within [log(4/n^2), log(n-1)] for every pair; complete and k-NN graphs, n = 10, 60",
           f"largest |lam^-1 log R - u| is {worst:.2f} of the bound 2 log(n) / lam", ok and worst <= 1.0)

    # Soft minimum spanning tree.
    worst_f, ok = 0.0, True
    X = np.random.default_rng(31).uniform(size=(6, 2))
    C = squared_distances(X)
    e = mst_edges(C)
    for kappa in (0.1, 1.0, 10.0, 100.0):
        lam = lam_from_scale(C, kappa)
        log_z, _, _, _ = enumerated_tree_law(C, lam)
        F = TreeLaw(X, lam).free_energy()
        worst_f = max(worst_f, abs(F + log_z / lam) / max(1.0, abs(F)))
        gap = C[e[:, 0], e[:, 1]].sum() - F
        ok &= bool(-1e-12 <= gap <= 4 * np.log(6) / lam)
    Xg = np.random.default_rng(32).uniform(size=(20, 2))
    lam = lam_from_scale(squared_distances(Xg), 3.0)
    grad = TreeLaw(Xg, lam).free_energy_grad()
    num = np.zeros_like(Xg)
    h = 1e-6
    for i in range(20):
        for d in range(2):
            Xp, Xm = Xg.copy(), Xg.copy()
            Xp[i, d] += h
            Xm[i, d] -= h
            num[i, d] = (TreeLaw(Xp, lam).free_energy() - TreeLaw(Xm, lam).free_energy()) / (2 * h)
    gerr = np.abs(num - grad).max() / np.abs(grad).max()
    record("Soft MST", "F = -log(Z)/lam vs enumeration and 0 <= S_MST - F <= (n-2) log(n)/lam (n = 6); grad F = 2 L_pi X vs central differences (n = 20)",
           f"F relative {worst_f:.1e}; bound holds; gradient relative {gerr:.1e}", ok and worst_f < EXACT_TOL and gerr < 1e-6)

    # Brownian motion on a tree: the tree law is a posterior over ultrametric covariances.
    n = 6
    X = np.random.default_rng(41).normal(size=(n, 2))
    lam = 0.8
    s2 = 1 / (2 * lam)
    log_z, _, trees, prob = enumerated_tree_law(squared_distances(X), lam)
    loglik, det_err = [], 0.0
    for t in trees:
        parent = reroot_parent(t, n)
        G = tree_covariance(parent)[1:, 1:]
        det_err = max(det_err, abs(np.linalg.det(G) - 1))
        Y = X[1:] - X[0]
        loglik.append(-0.5 * np.trace(Y.T @ np.linalg.solve(s2 * G, Y)) - Y.shape[1] / 2 * np.linalg.slogdet(2 * np.pi * s2 * G)[1])
    post = np.exp(np.array(loglik) - logsumexp(loglik))
    err = np.abs(post - prob).max()
    record("Brownian motion on a tree", "posterior over the 1,296 tree covariances G_T (flat prior, Gaussian steps) vs the tree law, n = 6",
           f"max abs error {err:.1e}; |det G_T - 1| at most {det_err:.1e}", err < EXACT_TOL and det_err < EXACT_TOL)

    # Two independent trees: E|T & T'| = sum pi_e^2 and Var |T & T'| <= sum pi_e^2 (1 - pi_e^2).
    X = np.random.default_rng(51).uniform(size=(5, 2))
    C = squared_distances(X)
    _, P, trees, prob = enumerated_tree_law(C, lam_from_scale(C, 1.0))
    sets = [frozenset((min(a, b), max(a, b)) for a, b in t) for t in trees]
    N = np.array([[len(a & b) for b in sets] for a in sets], dtype=float)
    mean = prob @ N @ prob
    var = prob @ (N**2) @ prob - mean**2
    pe = P[np.triu_indices(5, 1)]
    record("Overlap of two trees", "mean and variance of |T & T'| over all 125^2 pairs, n = 5",
           f"mean error {abs(mean - np.sum(pe**2)):.1e}; variance {var:.3f} <= bound {np.sum(pe**2 * (1 - pe**2)):.3f}",
           abs(mean - np.sum(pe**2)) < EXACT_TOL and var <= np.sum(pe**2 * (1 - pe**2)) + EXACT_TOL)

    # Annealed partition function on the circle: E Z = n^(n-2) kappa^(n-1).
    n, lam, draws = 5, 20.0, 20_000
    g = np.random.default_rng(61)
    Z = np.empty(draws)
    for d in range(draws):
        x = g.uniform(size=n)
        dist = np.abs(x[:, None] - x[None, :])
        dist = np.minimum(dist, 1 - dist)
        Z[d] = np.exp(TreeLaw.from_costs(dist**2, lam).log_partition())
    expected = n ** (n - 2) * (np.sqrt(np.pi / lam) * erf(np.sqrt(lam) / 2)) ** (n - 1)
    z = (Z.mean() - expected) / (Z.std(ddof=1) / np.sqrt(draws))
    pval = 2 * norm.sf(abs(z))
    record("Annealed partition function", f"{draws:,} uniform 5-point sets on the circle, lam = 20, vs n^(n-2) kappa^(n-1)",
           f"mean {Z.mean():.4f} vs {expected:.4f}, z = {z:.2f}, p = {pval:.3f}", pval > ALPHA)


def reroot_parent(edges, n, root=0):
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    parent = np.full(n, -2)
    parent[root] = -1
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if parent[v] == -2:
                parent[v] = u
                stack.append(v)
    return parent


PAGE = """<!-- generated by scripts/verify_exactness.py --write; edit the script, not this page -->

# Verification

The core fixed-kernel identities are checked against an independent answer obtained
by brute force: all $N^{{N-2}}$ labelled trees on the augmented graph (Pruefer codes) or
all set partitions. Exact identities are held to $10^{{-10}}$; sampling checks use a
chi-square test at $\\alpha = 0.001$ with seeds fixed in advance. The same checks run in
continuous integration on every push.

Unit tests additionally compare every root-attribution entry with independently enumerated
augmented trees, and check the weighted Aldous–Broder reference sampler against enumerated
edge and root marginals. Run `python -m pytest tests/test_benchmark_references.py`.

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

The tempered-tree rows check [the temperature dial](temperature.md): the log-space solver
against enumeration at every temperature, the point where a dense Laplacian inverse
loses its digits on clustered data, the resistance-to-single-linkage sandwich on
complete and sparse graphs, the soft-MST bound and gradient, the Brownian-motion
reading of the law, the overlap of two independent trees and the annealed partition
function (a Monte Carlo test, at the same $\\alpha$).

The last row illustrates Theorem 1 of Duan & Roy (2024): the leading eigenvectors of the
edge-inclusion matrix $M$ stay within a small subspace distance of those used by
normalized spectral clustering. It is an illustration of a published theorem, not a test.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    sys.exit(main(ap.parse_args().write))
