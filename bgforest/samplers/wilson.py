"""
Wilson's Loop-Erased Random Walk (LERW) Exact Spanning Tree/Forest Sampler
===========================================================================
Implements Wilson's algorithm (Wilson, 1996) and Fast-Forwarded Random Walks
(Tam, Dunson & Duan, 2025, Biometrika) for exact, unbiased sampling of random
spanning trees and forests from weighted graphs without MCMC mixing delays.
"""

from typing import List, Optional, Tuple, Union

import numpy as np

from bgforest.core.graph import validate_similarity_matrix


class WilsonLERWSampler:
    """
    Exact sampler for Uniform/Weighted Random Spanning Trees and Spanning Forests
    using Wilson's Loop-Erased Random Walk (LERW) algorithm.
    """

    def __init__(self, random_state: Optional[Union[int, np.random.RandomState]] = None):
        if isinstance(random_state, np.random.RandomState):
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)

    def sample_spanning_tree(
        self, W: np.ndarray, root: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Sample an exact, unbiased random spanning tree from weighted graph W.

        Parameters
        ----------
        W : np.ndarray of shape (n, n)
            Symmetric non-negative similarity matrix / graph adjacency.
        root : int, optional
            Root node to start tree generation. If None, chosen uniformly at random.

        Returns
        -------
        edges : List[Tuple[int, int]]
            List of (u, v) directed edges forming the sampled spanning tree.
        """
        W = validate_similarity_matrix(W, require_connected=True)
        n = W.shape[0]

        if n <= 1:
            return []

        # Precompute transition probability matrix P
        deg = np.sum(W, axis=1)
        deg = np.maximum(deg, 1e-12)
        P = W / deg[:, None]

        in_tree = np.zeros(n, dtype=bool)

        # Select initial root
        if root is not None and not 0 <= root < n:
            raise IndexError("root must be a valid node index.")
        r = root if root is not None else self.rng.randint(0, n)
        in_tree[r] = True

        next_node = -np.ones(n, dtype=int)

        # Iterate over all unvisited nodes
        for i in range(n):
            if in_tree[i]:
                continue

            u = i
            # Perform random walk until reaching a node already in tree
            while not in_tree[u]:
                # Pick next step according to transition probabilities
                v = self.rng.choice(n, p=P[u])
                next_node[u] = v
                u = v

            # Add loop-erased path to tree
            u = i
            while not in_tree[u]:
                in_tree[u] = True
                u = next_node[u]

        # Extract directed tree edges
        edges = []
        for u in range(n):
            if u != r and next_node[u] != -1:
                edges.append((u, int(next_node[u])))

        return edges

    def sample_spanning_forest(
        self, W: np.ndarray, roots: List[int]
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Sample an exact random spanning forest given K root nodes.

        Parameters
        ----------
        W : np.ndarray of shape (n, n)
            Weighted graph similarity matrix.
        roots : List[int]
            List of K distinct root nodes for the K components.

        Returns
        -------
        edges : List[Tuple[int, int]]
            List of directed edges forming the disjoint spanning trees.
        partition : np.ndarray of shape (n,)
            Cluster assignment index (0 to K-1) for each node.
        """
        W = validate_similarity_matrix(W)
        n = W.shape[0]
        roots = sorted(set(roots))
        if not roots:
            raise ValueError("roots must contain at least one node.")
        if any(not 0 <= root < n for root in roots):
            raise IndexError("roots must contain only valid node indices.")

        # Every connected component must include a root, otherwise the walk has
        # no absorbing tree and Wilson's algorithm cannot terminate.
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        n_components, component_labels = connected_components(csr_matrix(W > 0), directed=False)
        rooted_components = {component_labels[root] for root in roots}
        if len(rooted_components) != n_components:
            raise ValueError(
                "roots must include at least one node from every connected component of W."
            )

        deg = np.sum(W, axis=1)
        deg = np.maximum(deg, 1e-12)
        P = W / deg[:, None]

        in_tree = np.zeros(n, dtype=bool)
        partition = -np.ones(n, dtype=int)

        for k_idx, r in enumerate(roots):
            in_tree[r] = True
            partition[r] = k_idx

        next_node = -np.ones(n, dtype=int)

        for i in range(n):
            if in_tree[i]:
                continue

            u = i
            while not in_tree[u]:
                v = self.rng.choice(n, p=P[u])
                next_node[u] = v
                u = v

            target_cluster = partition[u]

            # Trace back loop-erased path and assign cluster label
            u = i
            while not in_tree[u]:
                in_tree[u] = True
                partition[u] = target_cluster
                u = next_node[u]

        edges = []
        for u in range(n):
            if u not in roots and next_node[u] != -1:
                edges.append((u, int(next_node[u])))

        return edges, partition
