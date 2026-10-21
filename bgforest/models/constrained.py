"""
Bayesian spanning forest with must-link and cannot-link constraints
===================================================================

Constraints condition the posterior: Pi(V | y, C) is Pi(V | y) restricted to the
partitions that satisfy C. Because each forest draw is exact given the kernel,
rejection keeps it exact: draw forests until one satisfies C. The acceptance rate
estimates Pr(C | y, kernel), which is itself informative: a constraint the data
strongly disagree with is accepted rarely, and the fit says so instead of forcing it.

This conditioning is an extension in this package, not part of Duan & Roy (2024).
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from bgforest.core.forest_model import ForestKernel
from bgforest.models.bsf import BayesianSpanningForest
from bgforest.samplers.wilson import RandomLike, WilsonSampler, forest_partition

Pair = Tuple[int, int]


class ConstrainedBayesianSpanningForest(BayesianSpanningForest):
    """:class:`BayesianSpanningForest` conditioned on pairwise constraints.

    Parameters
    ----------
    must_link, cannot_link : sequences of (i, j) index pairs
        Pairs that must share, or must not share, a cluster.
    max_attempts : int, default=2000
        Forest draws allowed per accepted draw before the fit gives up and reports
        the acceptance rate.

    All other parameters are those of :class:`BayesianSpanningForest`.

    Attributes
    ----------
    acceptance_rate_ : float
        Accepted / attempted forest draws: a Monte Carlo estimate of the posterior
        probability that the constraints hold (averaged over kernels in adaptive mode).
    """

    def __init__(
        self,
        must_link: Sequence[Pair] = (),
        cannot_link: Sequence[Pair] = (),
        max_attempts: int = 2000,
        n_clusters: Optional[int] = None,
        scales: Union[str, np.ndarray] = "adaptive",
        n_neighbors: int = 7,
        lam: float = 0.5,
        root_scale: float = 1.0,
        n_draws: int = 200,
        burn_in: int = 200,
        standardize: bool = True,
        random_state: RandomLike = None,
    ):
        super().__init__(
            n_clusters=n_clusters,
            scales=scales,
            n_neighbors=n_neighbors,
            lam=lam,
            root_scale=root_scale,
            n_draws=n_draws,
            burn_in=burn_in,
            standardize=standardize,
            random_state=random_state,
        )
        self.must_link = must_link
        self.cannot_link = cannot_link
        self.max_attempts = max_attempts

    def _is_constrained(self) -> bool:
        return bool(len(self.must_link) or len(self.cannot_link))

    def check_constraints_satisfied(self, labels: np.ndarray) -> bool:
        labels = np.asarray(labels)
        return all(labels[i] == labels[j] for i, j in self.must_link) and all(
            labels[i] != labels[j] for i, j in self.cannot_link
        )

    def _validate_constraints(self, n: int) -> None:
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for name, pairs in (("must_link", self.must_link), ("cannot_link", self.cannot_link)):
            for pair in pairs:
                if len(pair) != 2:
                    raise ValueError(f"each {name} entry must be a pair of indices.")
                i, j = pair
                if not (isinstance(i, (int, np.integer)) and isinstance(j, (int, np.integer))):
                    raise TypeError(f"{name} indices must be integers.")
                if not (0 <= i < n and 0 <= j < n) or i == j:
                    raise IndexError(f"{name} pair {pair} is out of range or repeats a point.")
        for i, j in self.must_link:
            parent[find(i)] = find(j)
        for i, j in self.cannot_link:
            if find(i) == find(j):
                raise ValueError(
                    f"cannot-link pair ({i}, {j}) is joined by the must-link constraints."
                )

    def fit(self, X: np.ndarray, y=None) -> "ConstrainedBayesianSpanningForest":
        X = np.asarray(X)
        self._validate_constraints(X.shape[0])
        self._attempts = 0
        self._accepted = 0
        super().fit(X)
        self.acceptance_rate_ = self._accepted / max(1, self._attempts)
        if self._is_constrained() and not self.check_constraints_satisfied(self.labels_):
            # Every draw satisfies the constraints; fall back to the draw closest to the
            # co-clustering matrix (Dahl's least-squares estimate) rather than return an
            # infeasible point estimate.
            P = self.co_clustering_matrix_
            losses = [
                np.sum(((lab[:, None] == lab[None, :]) - P) ** 2) for lab in self.partitions_
            ]
            self.labels_ = self.partitions_[int(np.argmin(losses))].copy()
            self.n_clusters_ = int(self.labels_.max() + 1)
        return self

    def _draw_forests(self, kernel: ForestKernel, sampler: WilsonSampler, k: int) -> np.ndarray:
        if not self._is_constrained():
            return sampler.sample_forests(kernel, k)
        accepted = []
        budget = self.max_attempts * k
        tried = 0
        while len(accepted) < k:
            if tried >= budget:
                rate = self._accepted / max(1, self._attempts + tried)
                raise RuntimeError(
                    f"constraints accepted in {len(accepted)} of {tried} forest draws "
                    f"(overall rate {rate:.2e}); the data put very little posterior mass on them."
                )
            batch = sampler.sample_forests(kernel, max(4, 2 * (k - len(accepted))))
            for parent in batch:
                tried += 1
                if self.check_constraints_satisfied(forest_partition(parent)):
                    accepted.append(parent)
                    if len(accepted) == k:
                        break
        self._attempts += tried
        self._accepted += k
        return np.array(accepted)
