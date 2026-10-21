"""A dependence backbone for the 13 chemical measurements of the UCI wine data.

    python examples/06_backbone.py

Edge widths are exact spanning-tree inclusion probabilities on a correlation kernel
(weight times effective resistance); dark edges form the backbone, which for Gaussian
data is the Chow-Liu tree.
"""

import sys

import numpy as np
from _common import ROOT, save

sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.datasets import load_wine  # noqa: E402

from bgforest import SpanningTreeBackbone  # noqa: E402

wine = load_wine()
names = [n.replace("_", " ").replace("od280/od315 of diluted wines", "OD280/OD315") for n in wine.feature_names]
model = SpanningTreeBackbone(bandwidth=0.08, random_state=0).fit(wine.data)
P, p = model.edge_probability_, len(names)
angle = np.pi / 2 - 2 * np.pi * np.arange(p) / p
pos = np.c_[np.cos(angle), np.sin(angle)]
backbone = set(model.backbone_edges_)

fig, ax = plt.subplots(figsize=(6.4, 6.0))
for i in range(p):
    for j in range(i + 1, p):
        if P[i, j] < 0.08 and (i, j) not in backbone:
            continue
        on = (i, j) in backbone
        ax.plot(*pos[[i, j]].T, color="#1e3a8a" if on else "#93c5fd", linewidth=0.6 + 5.5 * P[i, j],
                alpha=0.95 if on else 0.55, solid_capstyle="round", zorder=2 if on else 1)
ax.scatter(*pos.T, s=70, color="white", edgecolors="#1f2937", linewidths=1.2, zorder=3)
for (x, y), name in zip(pos, names):
    ha = "left" if x > 0.15 else ("right" if x < -0.15 else "center")
    va = "bottom" if y > 0.9 else ("top" if y < -0.9 else "center")
    ax.text(1.09 * x + 0.03 * np.sign(x) * (ha != "center"), 1.09 * y, name, ha=ha, va=va, fontsize=8.5, color="#111827")
ax.set_xlim(-1.9, 1.9)
ax.set_ylim(-1.35, 1.35)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Wine: spanning-tree edge probabilities (backbone in dark)")
save(fig, "backbone.svg")
print("backbone:", [(names[i], names[j]) for i, j in model.backbone_edges_])
