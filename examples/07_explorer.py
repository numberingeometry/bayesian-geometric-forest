"""Build the interactive explorer: docs/explorer/index.html, one self-contained page.

    python examples/07_explorer.py      # a few minutes (Iris is the slow one)

The page embeds, per dataset: 2-D display coordinates, the retained posterior draws
(parent pointers and partitions), the point estimate and per-point uncertainty, and
the posterior over K. Co-clustering for a clicked point is computed in the browser
from the stored partitions.
"""

import json
import sys

import numpy as np
from _common import ROOT, showcase
from sklearn.datasets import load_iris, make_circles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(ROOT))
from bgforest import BayesianSpanningForest  # noqa: E402

N_KEEP = 60


def pack(name, about, X_fit, X_show, k, seed):
    m = BayesianSpanningForest(n_clusters=k, n_draws=300, burn_in=300, random_state=seed).fit(X_fit)
    keep = np.linspace(0, len(m.forests_) - 1, N_KEEP).astype(int)
    u = m.assignment_uncertainty()
    post = {int(kk): round(v, 4) for kk, v in m.n_clusters_posterior_.items()}
    mode = max(post, key=post.get)
    print(f"{name}: K posterior mode {mode}, mean uncertainty {u.mean():.3f}")
    return {
        "name": name,
        "about": about,
        "n": int(len(X_fit)),
        "xy": np.round(X_show, 3).tolist(),
        "labels": m.labels_.tolist(),
        "uncertainty": np.round(u, 3).tolist(),
        "draws": m.forests_[keep].tolist(),
        "partitions": m.partitions_[keep].tolist(),
        "k_posterior": post,
        "k_note": f"Most probable K is {mode}; isolated points count as their own trees.",
        "default_focus": int(np.argmax(u)),
    }


X1, _ = showcase()
X2, _ = make_circles(300, noise=0.06, factor=0.5, random_state=2)
iris = load_iris()
Z3 = StandardScaler().fit_transform(iris.data)

datasets = [
    pack("Moons, blob, outliers", "Two interleaved moons, a compact blob joined to one moon by a thin bridge, "
         "and a few isolated points. Watch the bridge: some draws join the blob to the moon, most do not.",
         X1, X1, 3, 11),
    pack("Circles", "Two noisy concentric rings: no centroid method can separate them, a tree can.",
         X2, X2, 2, 3),
    pack("Iris", "Fisher's iris measurements (4 features, shown on two principal components). The "
         "posterior separates setosa with near certainty but joins versicolor and virginica: they form "
         "one connected cloud, and a forest clusters by connectivity, not by shape. A Gaussian mixture "
         "does better on these elliptical classes (see the benchmark).",
         iris.data, PCA(2, random_state=0).fit_transform(Z3), 2, 5),
]

template = (ROOT / "examples" / "explorer_template.html").read_text(encoding="utf-8")
html = template.replace("/*DATA*/null", json.dumps({"datasets": datasets}, separators=(",", ":")))
out = ROOT / "docs" / "explorer" / "index.html"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8")
print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size / 1024:.0f} KB)")
