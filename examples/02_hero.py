"""The README figures: four views of one posterior, and an animated loop of posterior draws.

    python examples/02_hero.py

Writes figs/hero.svg and figs/forest-draws.svg (an animated SVG: each frame is one
posterior forest from the chain, cycled with CSS, so it stays vector and needs no script).
"""

import json
import sys

import numpy as np
from _common import FIGS, ROOT, save, showcase

sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt  # noqa: E402

from bgforest import BayesianSpanningForest  # noqa: E402
from bgforest.viz import plot_coclustering, plot_uncertainty  # noqa: E402
from bgforest.viz.plotting import PALETTE  # noqa: E402

X, _ = showcase()
model = BayesianSpanningForest(n_clusters=3, n_draws=300, burn_in=300, random_state=11).fit(X)
Z = model.X_
labels = model.labels_
uncert = model.assignment_uncertainty()
k_post = model.n_clusters_posterior_
print("K posterior:", {k: round(v, 3) for k, v in sorted(k_post.items())})
print("mean uncertainty:", round(float(uncert.mean()), 3), " points above 0.2:", int((uncert > 0.2).sum()))


def draw_colors(parent):
    """Colour each tree of a draw by the point-estimate label most of its members carry."""
    from bgforest.samplers.wilson import forest_partition

    part = forest_partition(parent)
    out = np.empty(len(part), dtype=object)
    for c in np.unique(part):
        members = part == c
        if members.sum() < 6:
            out[members] = "#9ca3af"
        else:
            out[members] = PALETTE[int(np.bincount(labels[members]).argmax())]
    return out


# ---------------------------------------------------------------- static panels
fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1.25, 1.0, 1.25]})
axes[0].scatter(Z[:, 0], Z[:, 1], s=10, c="#374151", linewidths=0)
axes[0].set_title("Data")
axes[0].set_aspect("equal")
axes[0].axis("off")

draw = model.forests_[-1]
colors = draw_colors(draw)
child = np.flatnonzero(draw >= 0)
from matplotlib.collections import LineCollection  # noqa: E402

axes[1].add_collection(LineCollection(np.stack([Z[child], Z[draw[child]]], 1), colors=list(colors[child]), linewidths=1.0))
axes[1].scatter(Z[:, 0], Z[:, 1], s=9, c=list(colors), linewidths=0, zorder=3)
roots = draw < 0
axes[1].scatter(Z[roots, 0], Z[roots, 1], s=46, facecolors="white", edgecolors=list(colors[roots]), linewidths=1.4, zorder=4)
axes[1].set_title(f"One posterior draw ({int(roots.sum())} trees)")
axes[1].set_aspect("equal")
axes[1].axis("off")

plot_coclustering(model.co_clustering_matrix_, labels, ax=axes[2])
axes[2].set_title("Pr(i, j in the same cluster)")

plot_uncertainty(Z, uncert, ax=axes[3], point_size=12, vmax=0.5)
axes[3].set_title("Assignment uncertainty")
sm = plt.cm.ScalarMappable(cmap="magma_r", norm=plt.Normalize(0, 0.5))
cb = fig.colorbar(sm, ax=axes[3], fraction=0.035, pad=0.02)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=8)
fig.tight_layout()
save(fig, "hero.svg")

# ------------------------------------------------------------- animated draws
W, H, PAD = 720, 380, 14
lo, hi = Z.min(axis=0), Z.max(axis=0)
scale = min((W - 2 * PAD) / (hi[0] - lo[0]), (H - 2 * PAD) / (hi[1] - lo[1]))
off = np.array([(W - scale * (hi[0] - lo[0])) / 2, (H - scale * (hi[1] - lo[1])) / 2])


def xy(i):
    p = (Z[i] - lo) * scale + off
    return f"{p[0]:.1f}", f"{H - p[1]:.1f}"


frames = model.forests_[-240::20]  # 12 draws spread over the chain
n_frames = len(frames)
dwell = 0.9
period = n_frames * dwell
parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
    f'aria-label="Twelve posterior draws of a Bayesian spanning forest, cycling">',
    "<style>",
    f".f{{opacity:0;animation:cyc {period:.1f}s steps(1,end) infinite}}",
    f"@keyframes cyc{{0%{{opacity:1}}{100 / n_frames:.4f}%{{opacity:0}}100%{{opacity:0}}}}",
    "line{stroke-width:1.3;stroke-linecap:round}",
    "</style>",
    f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
]
for f, draw in enumerate(frames):
    colors = draw_colors(draw)
    parts.append(f'<g class="f" style="animation-delay:{f * dwell - period:.1f}s">')
    for c in sorted(set(colors)):
        segs = []
        for i in np.flatnonzero((draw >= 0) & (colors == c)):
            (x1, y1), (x2, y2) = xy(i), xy(draw[i])
            segs.append(f"M{x1} {y1}L{x2} {y2}")
        if segs:
            parts.append(f'<path d="{"".join(segs)}" stroke="{c}" fill="none" stroke-width="1.3"/>')
    for i in np.flatnonzero(draw < 0):
        x, y = xy(i)
        parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#fff" stroke="{colors[i]}" stroke-width="1.6"/>')
    parts.append(
        f'<text x="{W - 12}" y="{H - 12}" text-anchor="end" font-family="Inter,Helvetica,Arial,sans-serif" '
        f'font-size="13" fill="#6b7280">posterior draw {f + 1} of {n_frames} &#183; '
        f'{int((draw < 0).sum())} trees</text>'
    )
    parts.append("</g>")
# points on top, shaded by assignment uncertainty
for i in range(len(Z)):
    x, y = xy(i)
    u = min(uncert[i] / 0.5, 1.0)
    fill = PALETTE[labels[i]] if u < 0.4 else "#111827"
    parts.append(f'<circle cx="{x}" cy="{y}" r="{2.2 + 1.6 * u:.1f}" fill="{fill}"/>')
parts.append("</svg>")
out = FIGS / "forest-draws.svg"
out.write_text("".join(parts), encoding="utf-8")
print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")

summary = {
    "n": int(len(Z)),
    "n_clusters_posterior": {int(k): round(v, 4) for k, v in k_post.items()},
    "mean_assignment_uncertainty": round(float(uncert.mean()), 4),
}
(FIGS / "hero.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
