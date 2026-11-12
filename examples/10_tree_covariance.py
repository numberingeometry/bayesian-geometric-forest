"""Ultrametrics hiding in the forest: a tree's Brownian covariance and its Haar transform.

    python examples/10_tree_covariance.py

Writes figs/tree-covariance.svg: a cloud grown by diffusion along a complete binary tree,
the covariance G_T of that diffusion (an ultrametric matrix of nested blocks), and
W' G_T W in the Haar basis of the tree's clades, which vanishes off nested pairs.
"""

import sys

import numpy as np
from _common import ROOT, save

sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from bgforest.ultrametric import clade_haar_basis, tree_covariance, tree_diffusion  # noqa: E402

INK, GOLD, PAPER = "#004b49", "#be8c14", "#fcfaf5"
DEPTH = 5

n = 2 ** (DEPTH + 1) - 1
parent = np.array([-1] + [(v - 1) // 2 for v in range(1, n)])
X, _ = tree_diffusion(n, sigma=1.0, parent=parent, random_state=4)
G = tree_covariance(parent)
W, owner, clades = clade_haar_basis(parent)
B = W.T @ G @ W

# Depth-first order of the nodes, so every clade is a contiguous block.
order, stack = [], [0]
while stack:
    v = stack.pop()
    order.append(v)
    stack.extend([c for c in (2 * v + 2, 2 * v + 1) if c < n])
nested = np.array([[a < 0 or b < 0 or clades[a] <= clades[b] or clades[b] <= clades[a] for b in owner] for a in owner])
print(f"{n} nodes; |W'W - I| = {np.abs(W.T @ W - np.eye(n)).max():.1e}; largest entry off nested pairs "
      f"{np.abs(B[~nested]).max():.1e}; nested pairs {nested.mean():.1%} of the matrix")

cmap = LinearSegmentedColormap.from_list("forest", [PAPER, "#cfe0cf", "#6aa592", INK])
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), gridspec_kw={"width_ratios": [1.15, 1, 1]})
ax = axes[0]
kids = np.flatnonzero(parent >= 0)
ax.add_collection(LineCollection(np.stack([X[kids], X[parent[kids]]], 1), colors=GOLD, linewidths=1.3))
ax.scatter(X[:, 0], X[:, 1], s=10, c=INK, zorder=3)
ax.scatter(*X[0], s=90, facecolors="none", edgecolors=GOLD, linewidths=1.6, zorder=4)
ax.autoscale()
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Diffusion along a binary tree (root ringed)")
axes[1].imshow(G[np.ix_(order, order)], cmap=cmap, interpolation="nearest")
axes[1].set_title("Covariance $G_T$: nested blocks")
axes[2].imshow(np.log10(np.abs(B) + 1e-6), cmap=cmap, vmin=-3, interpolation="nearest")
axes[2].set_title("Haar transform $W^\\top G_T W$: nested pairs only")
for a in axes[1:]:
    a.set_xticks([])
    a.set_yticks([])
save(fig, "tree-covariance.svg")
