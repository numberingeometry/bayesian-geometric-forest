"""Duan & Roy (JASA 2024), Theorem 1, numerically: forest edge probabilities vs spectral clustering.

    python examples/04_theorem1.py

The posterior probability that edge (i, j) is in the forest, M_ij, has closed form
A_ij R_eff(i, j). The theorem says the leading eigenvectors of M are close to those
normalized spectral clustering uses (the bottom eigenvectors of N = I - D^-1/2 A D^-1/2)
when every A_ij / D_ii is small. This reproduces the paper's Figure 2 setting: a
three-component Gaussian mixture in R^2 with means (0,0), (2,2), (4,4) and identity
covariance; here with a global leaf scale sigma_ij = 1.
"""

import sys

import numpy as np
from _common import ROOT, save

sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402

from bgforest.core.forest_model import ForestKernel  # noqa: E402
from bgforest.viz.plotting import PALETTE  # noqa: E402


def eigen_pair(X, K=3):
    kern = ForestKernel.from_data(X, np.ones(len(X)), lam=0.5)
    A = np.exp(kern.augmented_log_weights() - kern._shift())
    np.fill_diagonal(A, 0.0)
    M_data, M_root = kern.edge_inclusion()
    M = np.zeros_like(A)
    M[:-1, :-1], M[:-1, -1], M[-1, :-1] = M_data, M_root, M_root
    d = A.sum(axis=1)
    N = np.eye(len(d)) - A / np.sqrt(np.outer(d, d))
    U = np.linalg.eigh(M)[1][:, ::-1][:, :K]
    V = np.linalg.eigh(N)[1][:, :K]
    # Align V to U with the orthogonal Procrustes rotation the theorem allows.
    P, s, Qt = np.linalg.svd(V.T @ U)
    sin_theta = float(np.sqrt(max(0.0, K - np.sum(s**2))))
    return U, V @ (P @ Qt), sin_theta


n = 300
rng = np.random.default_rng(42)
centers = ((0, 0), (2, 2), (4, 4))
X = np.vstack([rng.normal(c, 1.0, (n // 3, 2)) for c in centers])
U, V, sin_theta = eigen_pair(X)

sizes = (60, 120, 240, 480, 960)
dist = []
for m in sizes:
    g = np.random.default_rng(m)
    dist.append(eigen_pair(np.vstack([g.normal(c, 1.0, (m // 3, 2)) for c in centers]))[2])

fig = plt.figure(figsize=(12.5, 3.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.4)
group = np.repeat([0, 1, 2], n // 3)
for k in range(3):
    ax = fig.add_subplot(gs[0, k])
    lo, hi = min(U[:n, k].min(), V[:n, k].min()), max(U[:n, k].max(), V[:n, k].max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#d1d5db", linewidth=1, zorder=0)
    ax.scatter(V[:n, k], U[:n, k], s=8, c=[PALETTE[g] for g in group], linewidths=0)
    ax.set_title(f"Eigenvector {k + 1}")
    ax.set_xlabel("normalized spectral clustering")
    if k == 0:
        ax.set_ylabel("forest edge probabilities M")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal")
    ax.locator_params(nbins=4)
ax = fig.add_subplot(gs[0, 3])
ax.plot(sizes, dist, "o-", color=PALETTE[0], linewidth=1.2, markersize=4)
ax.set_xscale("log")
ax.set_xticks(sizes, [str(s) for s in sizes])
ax.set_ylim(0, 0.2)
ax.set_title("Subspace distance")
ax.set_xlabel("n")
ax.set_ylabel(r"$\|\sin\Theta\|_F$")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "theorem1.svg")
print(f"n = {n}: ||sin Theta||_F = {sin_theta:.3f}; sweep: " + ", ".join(f"{s}: {d:.3f}" for s, d in zip(sizes, dist)))
