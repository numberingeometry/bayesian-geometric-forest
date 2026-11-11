"""The temperature dial: README animation, its still frame, and the distortion table.

    python examples/08_dial.py

Writes docs/assets/dial.gif (800 x 500, loops hot -> cold -> hot) and
docs/assets/dial-still.webp, and prints the Cantor-dust / uniform-square table of
docs/temperature.md. Every frame shows exact edge probabilities from bgforest.TreeLaw.
"""

import sys

import numpy as np
from _common import ROOT

sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402

from bgforest import TreeLaw  # noqa: E402
from bgforest.treelaw import lam_from_scale, squared_distances  # noqa: E402
from bgforest.ultrametric import cantor_dust, single_linkage_ultrametric  # noqa: E402

PAPER, INK, GOLD, MUTED, LINE = "#fcfaf5", "#004b49", "#be8c14", "#5a726f", "#dce4d9"
ASSETS = ROOT / "docs" / "assets"


def distortion_table():
    def worst_ratio(P):
        D = squareform(pdist(P))
        U = single_linkage_ultrametric(D)
        off = ~np.eye(len(P), dtype=bool)
        return float((D[off] / U[off]).max())

    rng = np.random.default_rng(0)
    print("| | 16 points | 64 points | 256 points | 1024 points |")
    print("|---|---|---|---|---|")
    print("| Cantor dust, largest l/u | " + " | ".join(f"{worst_ratio(cantor_dust(k)):.2f}" for k in (2, 3, 4, 5)) + " |")
    print("| Uniform square, largest l/u | " + " | ".join(f"{worst_ratio(rng.uniform(size=(m, 2))):.2f}" for m in (16, 64, 256, 1024)) + " |")


def nested_clusters(seed=3):
    g = np.random.default_rng(seed)
    parts = [np.add(c, s) + 0.28 * g.normal(size=(12, 2))
             for c in ((0, 0), (6, 1), (2.5, 5.5)) for s in ((0, 0), (1.6, 0.4), (0.5, 1.5))]
    return np.vstack(parts)


def frame(X, kappa, lo, hi):
    C = squared_distances(X)
    law = TreeLaw(X, lam_from_scale(C, kappa))
    P = law.edge_probability()
    i, j = np.triu_indices(len(X), 1)
    p = P[i, j]
    keep = p > 0.004
    order = np.argsort(p[keep])
    segs = np.stack([X[i[keep]], X[j[keep]]], axis=1)[order]
    pk = p[keep][order]
    ink = np.array([0, 75, 73]) / 255
    colors = np.c_[np.tile(ink, (len(pk), 1)), np.minimum(1, 0.08 + 0.92 * pk**0.8)]

    fig = plt.figure(figsize=(8.0, 5.0), dpi=100, facecolor=PAPER)
    ax = fig.add_axes([0.06, 0.15, 0.88, 0.8])
    ax.set_facecolor(PAPER)
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=0.5 + 1.9 * pk, capstyle="round"))
    ax.scatter(X[:, 0], X[:, 1], s=16, c=INK, linewidths=0, zorder=3)
    pad = 0.4
    ax.set_xlim(X[:, 0].min() - pad, X[:, 0].max() + pad)
    ax.set_ylim(X[:, 1].min() - pad, X[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")

    dial = fig.add_axes([0.2, 0.065, 0.6, 0.03])
    dial.axis("off")
    dial.set_xlim(-0.04, 1.04)
    dial.set_ylim(-1, 1)
    t = (np.log10(kappa) - lo) / (hi - lo)
    dial.plot([0, 1], [0, 0], color=LINE, lw=5, solid_capstyle="round")
    dial.plot([0, t], [0, 0], color="#9fc2ad", lw=5, solid_capstyle="round")
    dial.scatter([t], [0], s=90, c=PAPER, edgecolors=INK, linewidths=1.6, zorder=3)
    fig.text(0.19, 0.08, "hot · any tree", ha="right", va="center", color=MUTED, fontsize=10)
    fig.text(0.81, 0.08, "cold · single linkage", ha="left", va="center", color=MUTED, fontsize=10)
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img, law.mst_overlap()


def main():
    distortion_table()
    X = nested_clusters()
    lo, hi = -1.0, 2.6
    up = list(np.linspace(lo, hi, 30))
    path = up + [hi] * 8 + up[::-1] + [lo] * 4
    frames = []
    for v in path:
        img, ov = frame(X, 10**v, lo, hi)
        frames.append(img)
    still, ov = frame(X, 10**1.2, lo, hi)
    print(f"still frame: {100 * ov:.0f}% of tree edges shared with the MST")
    palette = frames[len(up) // 2].quantize(colors=48, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    gif[0].save(ASSETS / "dial.gif", save_all=True, append_images=gif[1:], duration=100, loop=0, optimize=True)
    still.save(ASSETS / "dial-still.webp", quality=88)
    for name in ("dial.gif", "dial-still.webp"):
        print(f"wrote docs/assets/{name} ({(ASSETS / name).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
