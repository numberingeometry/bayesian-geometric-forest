"""The README quick start, runnable as is.

    python examples/01_quickstart.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import make_moons  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from bgforest import BayesianSpanningForest  # noqa: E402

X, y = make_moons(n_samples=300, noise=0.07, random_state=1)

model = BayesianSpanningForest(random_state=0).fit(X)  # K is inferred

print("posterior over K:", {k: round(p, 2) for k, p in sorted(model.n_clusters_posterior_.items())})
print("ARI of the point estimate:", round(adjusted_rand_score(y, model.labels_), 3))
print("posterior mean of K:", round(model.root_probability_.sum(), 2))
u = model.assignment_uncertainty()
print("most uncertain points:", u.argsort()[-3:][::-1].tolist(), "with", u.max().round(2))
