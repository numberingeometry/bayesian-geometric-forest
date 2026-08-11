"""
Bayesian Spanning Forest MCMC Sampler Engine
============================================
Implements Metropolis-Hastings Markov Chain Monte Carlo sampling routines 
for posterior inference over random spanning forests (Duan & Roy, 2022/2023).
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from bgforest.core.graph import compute_laplacian, extract_cluster_submatrix
from bgforest.core.matrix_tree import compute_forest_log_spanning_trees
from bgforest.models.forest_process import ForestProcess


class BSFMCMCSampler:
    """
    Metropolis-Hastings MCMC sampler for Bayesian Spanning Forest posteriors.
    """

    def __init__(
        self,
        forest_process: Optional[ForestProcess] = None,
        n_iter: int = 1000,
        burn_in: int = 300,
        thinning: int = 1,
        sigma_likelihood: Optional[float] = None,
        random_state: Optional[Union[int, np.random.RandomState]] = None
    ):
        self.forest_process = forest_process if forest_process is not None else ForestProcess()
        self.n_iter = int(n_iter)
        self.burn_in = int(burn_in)
        self.thinning = int(thinning)
        self.sigma_likelihood = sigma_likelihood

        if isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)

        self.traces_: List[float] = []
        self.samples_: List[np.ndarray] = []
        self.acceptance_rate_: float = 0.0

    def compute_log_likelihood(
        self,
        X: np.ndarray,
        W: np.ndarray,
        partition: np.ndarray
    ) -> float:
        """
        Compute marginal log-likelihood P(X | C) given partition C.
        """
        n, d = X.shape
        partition = np.asarray(partition, dtype=np.int64)
        unique_clusters = np.unique(partition)

        sigma_sq = (self.sigma_likelihood ** 2) if self.sigma_likelihood is not None else max(1e-4, np.var(X))

        log_lik = 0.0
        for k in unique_clusters:
            idx = np.where(partition == k)[0]
            m = len(idx)
            if m <= 1:
                continue

            X_k = X[idx]
            mean_k = np.mean(X_k, axis=0)
            sse_k = np.sum((X_k - mean_k) ** 2)

            log_lik += - sse_k / (2.0 * sigma_sq)

        return float(log_lik)

    def compute_log_posterior(
        self,
        X: np.ndarray,
        W: np.ndarray,
        partition: np.ndarray
    ) -> float:
        """Compute unnormalized log-posterior: log P(X | C) + log P(C)."""
        log_prior = self.forest_process.log_prior(W, partition)
        if np.isneginf(log_prior):
            return -np.inf

        log_lik = self.compute_log_likelihood(X, W, partition)
        return log_prior + log_lik

    def sample(
        self,
        X: np.ndarray,
        W: np.ndarray,
        initial_partition: Optional[np.ndarray] = None,
        target_n_clusters: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Execute MCMC chain to sample partitions from posterior.
        """
        n = X.shape[0]

        if self.sigma_likelihood is None:
            self.sigma_likelihood = float(np.sqrt(np.var(X)))

        if self.burn_in >= self.n_iter:
            self.burn_in = max(0, self.n_iter // 2)

        if initial_partition is not None:
            current_partition = np.copy(initial_partition)
        else:
            k_init = target_n_clusters if target_n_clusters is not None else 4
            from sklearn.cluster import SpectralClustering
            try:
                sc = SpectralClustering(
                    n_clusters=k_init, affinity="precomputed",
                    n_init=1, random_state=self.rng.randint(0, 10000)
                )
                current_partition = sc.fit_predict(W)
            except Exception:
                current_partition = self.rng.randint(0, k_init, size=n)

        current_log_post = self.compute_log_posterior(X, W, current_partition)

        self.traces_ = []
        self.samples_ = []
        n_accepted = 0
        n_proposed_moves = 0

        for it in range(self.n_iter):
            if target_n_clusters is not None:
                move_type = "relocate"
            else:
                move_type = self.rng.choice(["relocate", "split", "merge"], p=[0.6, 0.2, 0.2])

            proposed_partition = self._propose_move(current_partition, W, X, move_type, target_n_clusters=target_n_clusters)

            if not np.array_equal(proposed_partition, current_partition):
                n_proposed_moves += 1
                proposed_log_post = self.compute_log_posterior(X, W, proposed_partition)

                if not np.isneginf(proposed_log_post):
                    log_accept_ratio = proposed_log_post - current_log_post
                    half_burn = max(1, self.burn_in // 2)
                    temp = 1.0 + 3.0 * np.exp(-it / float(half_burn))

                    if np.log(self.rng.uniform(0.0, 1.0)) < (log_accept_ratio / temp):
                        current_partition = proposed_partition
                        current_log_post = proposed_log_post
                        n_accepted += 1

            self.traces_.append(current_log_post)

            if it >= self.burn_in and (it - self.burn_in) % self.thinning == 0:
                self.samples_.append(np.copy(current_partition))

        self.acceptance_rate_ = n_accepted / float(max(1, n_proposed_moves))
        return self.samples_

    def _propose_move(
        self,
        partition: np.ndarray,
        W: np.ndarray,
        X: np.ndarray,
        move_type: str,
        target_n_clusters: Optional[int] = None
    ) -> np.ndarray:
        """Helper to generate MCMC candidate state proposals."""
        n = len(partition)
        prop = np.copy(partition)

        if move_type == "relocate":
            # Find boundary nodes that have neighbors with different cluster labels
            boundary_candidates = []
            for i in range(n):
                graph_nbrs = np.where(W[i] > 0)[0]
                dists = np.sum((X - X[i]) ** 2, axis=1)
                spatial_nbrs = np.argsort(dists)[1:15]
                nbrs = np.unique(np.concatenate([graph_nbrs, spatial_nbrs]))
                if any(partition[j] != partition[i] for j in nbrs):
                    boundary_candidates.append(i)

            if len(boundary_candidates) > 0:
                node = self.rng.choice(boundary_candidates)
            else:
                node = self.rng.randint(0, n)

            graph_nbrs = np.where(W[node] > 0)[0]
            dists = np.sum((X - X[node]) ** 2, axis=1)
            spatial_nbrs = np.argsort(dists)[1:15]
            candidate_nbrs = np.unique(np.concatenate([graph_nbrs, spatial_nbrs]))

            # Filter candidate neighbors to those with DIFFERENT cluster labels
            diff_nbrs = [j for j in candidate_nbrs if prop[j] != prop[node]]
            if len(diff_nbrs) > 0:
                target_node = self.rng.choice(diff_nbrs)
                old_c = prop[node]
                new_c = prop[target_node]
                
                if target_n_clusters is not None:
                    if np.sum(prop == old_c) > 1:
                        prop[node] = new_c
                else:
                    prop[node] = new_c
                    _, prop = np.unique(prop, return_inverse=True)

        elif move_type == "split":
            unique_clusters, counts = np.unique(partition, return_counts=True)
            splittable = unique_clusters[counts >= 3]
            if len(splittable) > 0:
                c_split = self.rng.choice(splittable)
                nodes_c = np.where(partition == c_split)[0]
                m = len(nodes_c)

                sub_W = W[np.ix_(nodes_c, nodes_c)]
                sub_L = compute_laplacian(sub_W, normed=True)
                
                try:
                    evals, evecs = np.linalg.eigh(sub_L)
                    fiedler = evecs[:, 1]
                    split_mask = fiedler > 0
                except Exception:
                    split_mask = self.rng.rand(m) > 0.5

                if 0 < np.sum(split_mask) < m:
                    new_c_id = np.max(partition) + 1
                    prop[nodes_c[split_mask]] = new_c_id
                    _, prop = np.unique(prop, return_inverse=True)

        elif move_type == "merge":
            unique_clusters = np.unique(partition)
            if len(unique_clusters) >= 2:
                c1, c2 = self.rng.choice(unique_clusters, size=2, replace=False)
                nodes_c1 = np.where(partition == c1)[0]
                nodes_c2 = np.where(partition == c2)[0]
                sub_W = W[np.ix_(nodes_c1, nodes_c2)]
                if np.sum(sub_W) > 0:
                    prop[nodes_c2] = c1
                    _, prop = np.unique(prop, return_inverse=True)

        return prop

    def compute_co_clustering_matrix(self) -> np.ndarray:
        """
        Compute posterior co-clustering matrix P_ij = P(nodes i and j in same cluster).
        """
        if not self.samples_:
            raise RuntimeError("No MCMC samples available. Run `sample()` first.")

        n_samples = len(self.samples_)
        n_nodes = len(self.samples_[0])
        P = np.zeros((n_nodes, n_nodes), dtype=np.float64)

        for sample in self.samples_:
            eq_matrix = (sample[:, None] == sample[None, :]).astype(np.float64)
            P += eq_matrix

        P /= float(n_samples)
        return P
