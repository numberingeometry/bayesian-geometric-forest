"""Constraint-preserving Bayesian spanning forest clustering."""

from collections import Counter
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array

from bgforest.core.graph import build_knn_similarity, build_rbf_similarity, connect_knn_components
from bgforest.mcmc.sampler import BSFMCMCSampler
from bgforest.models.bsf import BayesianSpanningForest
from bgforest.models.forest_process import ForestProcess


class ConstrainedBayesianSpanningForest(BayesianSpanningForest):
    """BSF estimator that enforces must-link and cannot-link constraints at every draw."""

    def __init__(
        self,
        must_link: Optional[List[Tuple[int, int]]] = None,
        cannot_link: Optional[List[Tuple[int, int]]] = None,
        n_clusters: Optional[int] = 2,
        graph_type: str = "knn",
        n_neighbors: int = 10,
        n_iter: int = 500,
        burn_in: int = 100,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        **kwargs,
    ):
        super().__init__(
            n_clusters=n_clusters,
            graph_type=graph_type,
            n_neighbors=n_neighbors,
            n_iter=n_iter,
            burn_in=burn_in,
            random_state=random_state,
            **kwargs,
        )
        self.must_link = [] if must_link is None else must_link
        self.cannot_link = [] if cannot_link is None else cannot_link

    def _validate_constraints(self, n: int) -> None:
        for name, pairs in (("must_link", self.must_link), ("cannot_link", self.cannot_link)):
            for pair in pairs:
                if len(pair) != 2:
                    raise ValueError(f"Each {name} entry must be a pair of indices.")
                u, v = pair
                if not isinstance(u, (int, np.integer)) or not isinstance(v, (int, np.integer)):
                    raise TypeError(f"{name} indices must be integers.")
                if not (0 <= u < n and 0 <= v < n):
                    raise IndexError(f"{name} contains an index outside [0, {n - 1}].")

    def check_constraints_satisfied(self, partition: np.ndarray) -> bool:
        partition = np.asarray(partition)
        return all(partition[u] == partition[v] for u, v in self.must_link) and all(
            partition[u] != partition[v] for u, v in self.cannot_link
        )

    def _feasible_partition(self, base: np.ndarray, k: int) -> np.ndarray:
        """Colour must-link components while respecting cannot-link edges."""
        n = len(base)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        for u, v in self.must_link:
            union(u, v)
        roots = {find(i) for i in range(n)}
        root_to_component = {root: idx for idx, root in enumerate(sorted(roots))}
        component = np.array([root_to_component[find(i)] for i in range(n)])
        n_components = len(roots)
        if n_components < k:
            raise ValueError("Must-link constraints leave fewer components than n_clusters.")

        adjacency = [set() for _ in range(n_components)]
        for u, v in self.cannot_link:
            a, b = component[u], component[v]
            if a == b:
                raise ValueError("A cannot-link pair is contained in a must-link component.")
            adjacency[a].add(b)
            adjacency[b].add(a)

        preferences = []
        for c in range(n_components):
            votes = np.bincount(base[component == c], minlength=k)
            preferences.append(list(np.argsort(votes)[::-1]))

        colors = -np.ones(n_components, dtype=np.int64)
        uncoloured = set(range(n_components))
        while uncoloured:
            # DSATUR: highest colour saturation, then degree.
            current = max(
                uncoloured,
                key=lambda c: (
                    len({colors[v] for v in adjacency[c] if colors[v] >= 0}),
                    len(adjacency[c]),
                ),
            )
            used = {colors[v] for v in adjacency[current] if colors[v] >= 0}
            colour = next((value for value in preferences[current] if value not in used), None)
            if colour is None:
                raise ValueError("The cannot-link graph is not colourable with n_clusters.")
            colors[current] = colour
            uncoloured.remove(current)

        # Retain exactly K non-empty labels for the fixed-K Gibbs kernel.
        missing = [label for label in range(k) if label not in colors]
        for label in missing:
            colour_counts = np.bincount(colors, minlength=k)
            candidates = [c for c in range(n_components) if colour_counts[colors[c]] > 1]
            if not candidates:
                raise ValueError("Could not retain every cluster while satisfying the constraints.")
            colors[candidates[0]] = label
        result = colors[component]
        if len(np.unique(result)) != k or not self.check_constraints_satisfied(result):
            raise ValueError("Could not construct a feasible non-empty constrained partition.")
        return result

    @staticmethod
    def _connect_initial_clusters(
        W: np.ndarray, X: np.ndarray, partition: np.ndarray
    ) -> np.ndarray:
        """Add minimal local bridges so the feasible initial state has support."""
        repaired = W.copy()
        distances = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
        positive = repaired[repaired > 0]
        bridge_weight = float(np.min(positive)) if positive.size else 1.0
        for label in np.unique(partition):
            nodes = np.where(partition == label)[0]
            while len(nodes) > 1:
                n_components, labels = connected_components(
                    csr_matrix(repaired[np.ix_(nodes, nodes)] > 0), directed=False
                )
                if n_components == 1:
                    break
                best = None
                for left in range(n_components):
                    left_nodes = nodes[labels == left]
                    for right in range(left + 1, n_components):
                        right_nodes = nodes[labels == right]
                        local = distances[np.ix_(left_nodes, right_nodes)]
                        idx = np.unravel_index(np.argmin(local), local.shape)
                        candidate = (local[idx], left_nodes[idx[0]], right_nodes[idx[1]])
                        if best is None or candidate[0] < best[0]:
                            best = candidate
                _, u, v = best
                repaired[u, v] = repaired[v, u] = bridge_weight
        return repaired

    @staticmethod
    def _posterior_mode(samples: List[np.ndarray]) -> np.ndarray:
        def canonical(sample: np.ndarray) -> Tuple[int, ...]:
            mapping, result = {}, []
            for value in sample:
                value = int(value)
                if value not in mapping:
                    mapping[value] = len(mapping)
                result.append(mapping[value])
            return tuple(result)

        return np.asarray(
            Counter(canonical(sample) for sample in samples).most_common(1)[0][0], dtype=np.int64
        )

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        initial_partition: Optional[np.ndarray] = None,
    ):
        X_raw = check_array(X, ensure_min_samples=1, ensure_min_features=1, dtype=np.float64)
        n = X_raw.shape[0]
        self._validate_constraints(n)
        if self.n_clusters is None:
            raise ValueError("ConstrainedBayesianSpanningForest requires an explicit n_clusters.")
        k = int(self.n_clusters)
        if not 1 <= k <= n:
            raise ValueError("n_clusters must be between 1 and n_samples.")

        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_proc = self.scaler_.fit_transform(X_raw)
        else:
            X_proc = X_raw

        if self.graph_type == "rbf":
            W = build_rbf_similarity(
                X_proc,
                gamma=self.gamma,
                sigma=self.sigma,
                adaptive_bandwidth=self.adaptive_bandwidth,
            )
        elif self.graph_type == "knn":
            W = build_knn_similarity(X_proc, n_neighbors=self.n_neighbors, symmetric=True)
            if self.ensure_connected:
                W = connect_knn_components(W, X_proc)
        else:
            raise ValueError("graph_type must be either 'rbf' or 'knn'.")
        if initial_partition is None:
            from sklearn.cluster import SpectralClustering

            base = SpectralClustering(
                n_clusters=k, affinity="precomputed", n_init=10, random_state=self.random_state
            ).fit_predict(W)
        else:
            base = np.asarray(initial_partition, dtype=np.int64)
            if base.ndim != 1 or len(base) != n:
                raise ValueError("initial_partition must have one label per observation.")
        initial = self._feasible_partition(base, k)
        W = self._connect_initial_clusters(W, X_proc, initial)
        self.W_ = W

        sampler = BSFMCMCSampler(
            forest_process=ForestProcess(alpha=self.alpha, beta=self.beta, theta=self.theta),
            n_iter=self.n_iter,
            burn_in=self.burn_in,
            thinning=self.thinning,
            sigma_likelihood=self.sigma_likelihood,
            random_state=self.random_state,
        )
        self.mcmc_sampler_ = sampler
        samples = sampler.sample(
            X_proc,
            W,
            initial,
            target_n_clusters=k,
            constraints_check_fn=self.check_constraints_satisfied,
        )
        self.log_posterior_trace_ = sampler.traces_
        self.acceptance_rate_ = sampler.acceptance_rate_
        self.co_clustering_matrix_ = sampler.compute_co_clustering_matrix()
        self.posterior_co_clustering_ = self.co_clustering_matrix_
        self.labels_ = self._posterior_mode(samples)
        self.n_clusters_inferred_ = k
        self.spectral_eigenvalues_ = np.array([])
        self.spectral_eigenvectors_ = np.empty((n, 0))
        return self
