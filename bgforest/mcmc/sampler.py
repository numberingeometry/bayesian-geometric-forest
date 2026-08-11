"""Posterior sampler for fixed-cardinality Bayesian spanning forests.

The sampler uses single-site Gibbs updates.  Each update enumerates all valid
existing-cluster assignments for one observation and draws from their exact
conditional posterior probabilities.  Unlike the earlier heuristic relocate /
split / merge routine, this kernel needs neither an annealing schedule nor an
omitted Metropolis-Hastings proposal correction.
"""

from typing import Callable, List, Optional, Union

import numpy as np
from scipy.special import logsumexp

from bgforest.core.graph import validate_similarity_matrix
from bgforest.models.forest_process import ForestProcess

ConstraintCheck = Callable[[np.ndarray], bool]


class BSFMCMCSampler:
    """Gibbs sampler for partitions with a fixed positive number of clusters."""

    def __init__(
        self,
        forest_process: Optional[ForestProcess] = None,
        n_iter: int = 1000,
        burn_in: int = 300,
        thinning: int = 1,
        sigma_likelihood: Optional[float] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
    ):
        if n_iter < 1:
            raise ValueError("n_iter must be at least 1.")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")
        if thinning < 1:
            raise ValueError("thinning must be at least 1.")
        if sigma_likelihood is not None and sigma_likelihood <= 0:
            raise ValueError("sigma_likelihood must be strictly positive.")

        self.forest_process = forest_process if forest_process is not None else ForestProcess()
        self.n_iter = int(n_iter)
        self.burn_in = int(burn_in)
        self.thinning = int(thinning)
        self.sigma_likelihood = sigma_likelihood
        self.rng = (
            random_state
            if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )

        self.traces_: List[float] = []
        self.samples_: List[np.ndarray] = []
        self.acceptance_rate_: float = 0.0
        self.n_clusters_: Optional[int] = None

    def compute_log_likelihood(self, X: np.ndarray, W: np.ndarray, partition: np.ndarray) -> float:
        """Compute the fixed-variance within-cluster Gaussian log likelihood."""
        del W  # retained for backward-compatible method signature
        X = np.asarray(X, dtype=np.float64)
        partition = np.asarray(partition, dtype=np.int64)
        sigma_sq = self.sigma_likelihood**2
        log_lik = 0.0
        for label in np.unique(partition):
            X_k = X[partition == label]
            if len(X_k) > 1:
                centered = X_k - np.mean(X_k, axis=0)
                log_lik -= float(np.sum(centered * centered)) / (2.0 * sigma_sq)
        return float(log_lik)

    def compute_log_posterior(
        self,
        X: np.ndarray,
        W: np.ndarray,
        partition: np.ndarray,
        constraints_check_fn: Optional[ConstraintCheck] = None,
    ) -> float:
        """Compute the unnormalised posterior and reject infeasible states."""
        partition = np.asarray(partition, dtype=np.int64)
        if constraints_check_fn is not None and not constraints_check_fn(partition):
            return -np.inf
        log_prior = self.forest_process.log_prior(W, partition)
        if np.isneginf(log_prior):
            return -np.inf
        return float(log_prior + self.compute_log_likelihood(X, W, partition))

    @staticmethod
    def _canonicalize(partition: np.ndarray) -> np.ndarray:
        """Relabel a partition by first occurrence without changing memberships."""
        mapping = {}
        result = np.empty(len(partition), dtype=np.int64)
        next_label = 0
        for i, value in enumerate(partition):
            value = int(value)
            if value not in mapping:
                mapping[value] = next_label
                next_label += 1
            result[i] = mapping[value]
        return result

    def _initial_partition(self, W: np.ndarray, target_n_clusters: int) -> np.ndarray:
        n = W.shape[0]
        if target_n_clusters == 1:
            return np.zeros(n, dtype=np.int64)
        from sklearn.cluster import SpectralClustering

        try:
            return SpectralClustering(
                n_clusters=target_n_clusters,
                affinity="precomputed",
                n_init=10,
                random_state=self.rng.randint(0, 2**31 - 1),
            ).fit_predict(W)
        except Exception:
            # Guaranteed non-empty fallback with no global RNG dependence.
            return self.rng.permutation(np.arange(n)) % target_n_clusters

    def sample(
        self,
        X: np.ndarray,
        W: np.ndarray,
        initial_partition: Optional[np.ndarray] = None,
        target_n_clusters: Optional[int] = None,
        constraints_check_fn: Optional[ConstraintCheck] = None,
    ) -> List[np.ndarray]:
        """Draw posterior samples using valid single-site Gibbs transitions.

        ``target_n_clusters`` is intentionally required in practice: the
        transition kernel samples a well-defined posterior conditional on K.
        Model selection for K belongs outside this fixed-K kernel.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a non-empty 2D array containing finite values.")
        W = validate_similarity_matrix(W)
        if W.shape[0] != X.shape[0]:
            raise ValueError("X and W must have the same number of observations.")
        n = X.shape[0]

        if self.sigma_likelihood is None:
            self.sigma_likelihood = float(max(1e-4, 2.5 * np.sqrt(np.var(X))))

        if initial_partition is None:
            k = int(target_n_clusters) if target_n_clusters is not None else min(4, n)
            current = self._initial_partition(W, k)
        else:
            current = self._canonicalize(np.asarray(initial_partition, dtype=np.int64))
            if current.ndim != 1 or len(current) != n:
                raise ValueError(
                    "initial_partition must be a 1D array with one label per observation."
                )
            k = int(target_n_clusters) if target_n_clusters is not None else len(np.unique(current))

        if not 1 <= k <= n:
            raise ValueError("target_n_clusters must be between 1 and n_samples.")
        if len(np.unique(current)) != k:
            raise ValueError(
                "initial_partition must contain exactly target_n_clusters non-empty clusters."
            )
        if constraints_check_fn is not None and not constraints_check_fn(current):
            raise ValueError("initial_partition violates the supplied constraints.")

        self.n_clusters_ = k
        effective_burn_in = min(self.burn_in, self.n_iter - 1)
        self.traces_ = []
        self.samples_ = []
        moved = 0
        update_attempts = 0
        current_log_post = self.compute_log_posterior(X, W, current, constraints_check_fn)
        if np.isneginf(current_log_post):
            raise ValueError("The initial partition has zero posterior probability.")

        for iteration in range(self.n_iter):
            # Random scan avoids systematic order effects while preserving Gibbs invariance.
            node = int(self.rng.randint(n))
            old_label = int(current[node])
            counts = np.bincount(current, minlength=k)
            candidate_labels = np.arange(k)
            if counts[old_label] == 1:
                candidate_labels = candidate_labels[candidate_labels == old_label]

            candidates, log_weights = [], []
            for label in candidate_labels:
                candidate = current.copy()
                candidate[node] = label
                score = self.compute_log_posterior(X, W, candidate, constraints_check_fn)
                if np.isfinite(score):
                    candidates.append(candidate)
                    log_weights.append(score)

            if not candidates:
                raise RuntimeError("No feasible Gibbs assignment exists for the current state.")
            probabilities = np.exp(np.asarray(log_weights) - logsumexp(log_weights))
            selected = int(self.rng.choice(len(candidates), p=probabilities))
            proposal = candidates[selected]
            update_attempts += 1
            if proposal[node] != current[node]:
                moved += 1
            current = proposal
            current_log_post = float(log_weights[selected])
            self.traces_.append(current_log_post)

            if (
                iteration >= effective_burn_in
                and (iteration - effective_burn_in) % self.thinning == 0
            ):
                self.samples_.append(self._canonicalize(current))

        self.acceptance_rate_ = moved / float(max(1, update_attempts))
        return self.samples_

    def compute_co_clustering_matrix(self) -> np.ndarray:
        """Return posterior co-clustering probabilities from retained samples."""
        if not self.samples_:
            raise RuntimeError("No MCMC samples available. Run sample() first.")
        n = len(self.samples_[0])
        P = np.zeros((n, n), dtype=np.float64)
        for sample in self.samples_:
            P += (sample[:, None] == sample[None, :]).astype(np.float64)
        return P / float(len(self.samples_))
