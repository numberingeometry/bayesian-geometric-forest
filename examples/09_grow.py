"""Grow a cloud along a random tree, then ask which tree grew it.

    python examples/09_grow.py

Writes docs/assets/grow.gif (800 x 500) and docs/assets/grow-still.webp. The cloud is
grown by Gaussian diffusion along a uniform random labelled tree, one generation per
beat. Then the true tree fades and the exact posterior over trees takes over, first at
the temperature that grew the cloud, then cooling to the minimum spanning tree.
"""

import sys

import numpy as np
from _common import ROOT

sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

from bgforest import TreeLaw  # noqa: E402
from bgforest.treelaw import lam_from_scale, squared_distances  # noqa: E402
from bgforest.ultrametric import tree_diffusion  # noqa: E402

PAPER, INK, GOLD, MUTED, LINE = "#fcfaf5", "#004b49", "#be8c14", "#5a726f", "#dce4d9"
ASSETS = ROOT / "docs" / "assets"
N, SIGMA, SEED = 80, 1.0, 14


def ease(p):
    p = np.clip(p, 0.0, 1.0)
    return 1 - (1 - p) ** 3


class Canvas:
    def __init__(self, X):
        self.lo = X.min(axis=0) - 0.6
        self.hi = X.max(axis=0) + 0.6

    def new(self):
        fig = plt.figure(figsize=(8.0, 5.0), dpi=100, facecolor=PAPER)
        ax = fig.add_axes([0.04, 0.14, 0.92, 0.84])
        ax.set_facecolor(PAPER)
        ax.set_xlim(self.lo[0], self.hi[0])
        ax.set_ylim(self.lo[1], self.hi[1])
        ax.set_aspect("equal")
        ax.axis("off")
        return fig, ax

    @staticmethod
    def caption(fig, text):
        fig.text(0.5, 0.115, text, ha="center", va="center", color=MUTED, fontsize=10)

    @staticmethod
    def dial(fig, t):
        d = fig.add_axes([0.25, 0.04, 0.5, 0.03])
        d.axis("off")
        d.set_xlim(-0.04, 1.04)
        d.set_ylim(-1, 1)
        d.plot([0, 1], [0, 0], color=LINE, lw=5, solid_capstyle="round")
        d.plot([0, t], [0, 0], color="#9fc2ad", lw=5, solid_capstyle="round")
        d.scatter([t], [0], s=90, c=PAPER, edgecolors=INK, linewidths=1.6, zorder=3)
        fig.text(0.24, 0.055, "hot", ha="right", va="center", color=MUTED, fontsize=10)
        fig.text(0.76, 0.055, "cold", ha="left", va="center", color=MUTED, fontsize=10)

    @staticmethod
    def image(fig):
        fig.canvas.draw()
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)
        return img


def grow_frame(cv, X, parent, depth, t):
    """t is measured in generations: node v slides out from its parent during [depth, depth + 1]."""
    cur = np.full_like(X, np.nan)
    for lev in range(depth.max() + 1):
        for v in np.flatnonzero(depth == lev):
            if t < lev:
                continue
            if parent[v] < 0:
                cur[v] = X[v]
            else:
                a = cur[parent[v]]
                cur[v] = a + ease(t - lev) * (X[v] - a)
    fig, ax = cv.new()
    shown = ~np.isnan(cur[:, 0])
    kids = [v for v in np.flatnonzero(shown) if parent[v] >= 0]
    ax.add_collection(LineCollection([[cur[parent[v]], cur[v]] for v in kids], colors=GOLD, linewidths=1.6, capstyle="round"))
    ax.scatter(cur[shown, 0], cur[shown, 1], s=16, c=INK, linewidths=0, zorder=3)
    root = int(np.flatnonzero(parent < 0)[0])
    ax.scatter([X[root, 0]], [X[root, 1]], s=140, facecolors="none", edgecolors=GOLD, linewidths=1.8, zorder=4)
    cv.caption(fig, "grow · each step adds an independent Gaussian displacement")
    return cv.image(fig)


def law_frame(cv, X, parent, log_kappa, lo, hi, truth_alpha, caption):
    law = TreeLaw(X, lam_from_scale(squared_distances(X), 10**log_kappa))
    P = law.edge_probability()
    i, j = np.triu_indices(len(X), 1)
    p = P[i, j]
    keep = p > 0.004
    order = np.argsort(p[keep])
    pk = p[keep][order]
    segs = np.stack([X[i[keep]], X[j[keep]]], axis=1)[order]
    ink = np.array([0, 75, 73]) / 255
    fig, ax = cv.new()
    if truth_alpha > 0:
        kids = np.flatnonzero(parent >= 0)
        ax.add_collection(LineCollection([[X[parent[v]], X[v]] for v in kids], colors=GOLD, alpha=truth_alpha, linewidths=1.6))
    ax.add_collection(LineCollection(segs, colors=np.c_[np.tile(ink, (len(pk), 1)), np.minimum(1, 0.08 + 0.92 * pk**0.8)],
                                     linewidths=0.5 + 1.9 * pk, capstyle="round"))
    ax.scatter(X[:, 0], X[:, 1], s=16, c=INK, linewidths=0, zorder=3)
    cv.caption(fig, caption)
    cv.dial(fig, (log_kappa - lo) / (hi - lo))
    return cv.image(fig)


def main():
    X, parent = tree_diffusion(N, sigma=SIGMA, random_state=SEED)
    depth = np.zeros(N, dtype=int)
    for v in range(N):
        u = v
        while parent[u] >= 0:
            u = parent[u]
            depth[v] += 1
    C = squared_distances(X)
    log_true = float(np.log10(lam_from_scale(C, 1.0) ** -1 / (2 * SIGMA**2)))  # kappa of lam = 1/(2 sigma^2)
    lo, hi = min(-1.0, log_true - 0.2), 2.6
    cv = Canvas(X)

    frames, durations = [], []
    G = depth.max() + 1
    for t in np.linspace(0, G, 2 * G + 1):
        frames.append(grow_frame(cv, X, parent, depth, t))
        durations.append(110)
    durations[-1] = 1300
    infer = "infer · every tree that could have grown it, by its posterior probability"
    for a in (0.75, 0.45, 0.2):
        frames.append(law_frame(cv, X, parent, log_true, lo, hi, a, infer))
        durations.append(160)
    frames.append(law_frame(cv, X, parent, log_true, lo, hi, 0.0, infer))
    durations.append(1200)
    for v in np.linspace(log_true, hi, 22)[1:]:
        frames.append(law_frame(cv, X, parent, v, lo, hi, 0.0, "cool · the posterior settles into the minimum spanning tree"))
        durations.append(110)
    durations[-1] = 1800

    law = TreeLaw(X, 1 / (2 * SIGMA**2))
    P = law.edge_probability()
    kids = np.flatnonzero(parent >= 0)
    print(f"{N} points, {G} generations, true kappa = {10**log_true:.3f}")
    print(f"at the true temperature: {100 * P[kids, parent[kids]].sum() / (N - 1):.0f}% of the true tree's edges expected, "
          f"{100 * law.mst_overlap():.0f}% of the MST's")

    sample = Image.new("RGB", (frames[0].width, 2 * frames[0].height))
    sample.paste(frames[2 * G], (0, 0))
    sample.paste(frames[-1], (0, frames[0].height))
    palette = sample.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    gif[0].save(ASSETS / "grow.gif", save_all=True, append_images=gif[1:], duration=durations, loop=0, optimize=True)
    frames[2 * G].save(ASSETS / "grow-still.webp", quality=88)
    for name in ("grow.gif", "grow-still.webp"):
        print(f"wrote docs/assets/{name} ({(ASSETS / name).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
