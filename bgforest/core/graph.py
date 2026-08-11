"""
Graph Construction and Laplacian Matrix Operations
===================================================
Provides graph adjacency building, kernel computations, and Laplacian matrix
algebra required for Bayesian Spanning Forest graph partitioning.
"""

from typing import Optional, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform


def compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix for input data matrix X.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data matrix.

    Returns
    -------
    dist_matrix : np.ndarray of shape (n_samples, n_samples)
        Symmetric matrix of pairwise Euclidean distances.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-dimensional array.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    return squareform(pdist(X, metric="euclidean"))


def build_rbf_similarity(
    X: np.ndarray,
    gamma: Optional[float] = None,
    sigma: Optional[float] = None,
    adaptive_bandwidth: bool = False,
    k_adaptive: int = 7,
) -> np.ndarray:
    """
    Construct Gaussian Radial Basis Function (RBF) similarity matrix.

    W_ij = exp( - ||x_i - x_j||^2 / (2 * sigma_i * sigma_j) )

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input dataset.
    gamma : float, optional
        Kernel coefficient for RBF kernel: gamma = 1 / (2 * sigma^2).
    sigma : float, optional
        Standard deviation bandwidth for Gaussian kernel.
    adaptive_bandwidth : bool, default=False
        If True, computes local self-tuning bandwidth sigma_i based on distance to k-th neighbor.
    k_adaptive : int, default=7
        k parameter for local adaptive bandwidth estimation (Zelnik-Manor & Perona, 2004).

    Returns
    -------
    W : np.ndarray of shape (n_samples, n_samples)
        Symmetric similarity matrix with zeros on diagonal.
    """
    if gamma is not None and gamma <= 0:
        raise ValueError("gamma must be strictly positive when provided.")
    if sigma is not None and sigma <= 0:
        raise ValueError("sigma must be strictly positive when provided.")
    if k_adaptive < 1:
        raise ValueError("k_adaptive must be at least 1.")

    D = compute_pairwise_distances(X)
    n = D.shape[0]

    if adaptive_bandwidth:
        # Distance to k-th nearest neighbor for each sample
        sorted_D = np.sort(D, axis=1)
        k_idx = min(k_adaptive, n - 1)
        sigmas = sorted_D[:, k_idx]
        sigmas = np.maximum(sigmas, 1e-8)  # prevent division by zero

        # Outer product of local bandwidths: sigma_i * sigma_j
        sigma_matrix = np.outer(sigmas, sigmas)
        W = np.exp(-(D**2) / (2.0 * sigma_matrix))
    else:
        if gamma is not None:
            bandwidth_sq = 1.0 / (2.0 * gamma)
        elif sigma is not None:
            bandwidth_sq = sigma**2
        else:
            # Default heuristic: median pairwise distance squared
            median_dist = np.median(D)
            bandwidth_sq = (median_dist**2) if median_dist > 0 else 1.0

        W = np.exp(-(D**2) / (2.0 * bandwidth_sq))

    np.fill_diagonal(W, 0.0)
    return W


def build_knn_similarity(
    X: np.ndarray,
    n_neighbors: int = 10,
    metric: str = "euclidean",
    mode: str = "distance",
    symmetric: bool = True,
) -> np.ndarray:
    """
    Construct k-Nearest Neighbors similarity graph.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input dataset.
    n_neighbors : int, default=10
        Number of nearest neighbors per node.
    metric : str, default='euclidean'
        Distance metric.
    mode : str, default='distance'
        'distance' converts distances to Gaussian weights, 'connectivity' uses binary 0/1.
    symmetric : bool, default=True
        If True, makes graph symmetric (W + W^T) / 2 or max(W, W^T).

    Returns
    -------
    W : np.ndarray of shape (n_samples, n_samples)
        k-NN adjacency matrix.
    """
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must be a non-empty 2-dimensional array.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")
    if mode not in {"distance", "connectivity"}:
        raise ValueError("mode must be either 'distance' or 'connectivity'.")

    n = X.shape[0]
    if n == 1:
        return np.zeros((1, 1), dtype=np.float64)
    k = min(n_neighbors + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
    distances, indices = nn.kneighbors(X)

    W = np.zeros((n, n), dtype=np.float64)

    if mode == "distance":
        # Estimate sigma from median neighbor distance
        sigma = np.median(distances[:, 1:])
        sigma = max(sigma, 1e-6)
        weights = np.exp(-(distances**2) / (2.0 * sigma**2))
    else:
        weights = np.ones_like(distances)

    for i in range(n):
        for idx_pos in range(1, k):  # Skip self
            j = indices[i, idx_pos]
            w = weights[i, idx_pos]
            W[i, j] = w

    if symmetric:
        W = np.maximum(W, W.T)

    np.fill_diagonal(W, 0.0)
    return W


def compute_laplacian(
    W: np.ndarray, normed: bool = False, return_degree: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute graph Laplacian matrix L from weighted adjacency W.

    Unnormalized Laplacian: L = D - W
    Normalized Laplacian: L_normed = I - D^{-1/2} W D^{-1/2}

    Parameters
    ----------
    W : np.ndarray of shape (n_samples, n_samples)
        Symmetric adjacency matrix.
    normed : bool, default=False
        If True, computes symmetric normalized Laplacian L_sym.
    return_degree : bool, default=False
        If True, returns degree matrix/vector as well.

    Returns
    -------
    L : np.ndarray of shape (n_samples, n_samples)
        Graph Laplacian.
    d : np.ndarray of shape (n_samples,), optional
        Node degrees.
    """
    W = validate_similarity_matrix(W)
    d = np.sum(W, axis=1)

    if normed:
        d_inv_sqrt = np.power(np.maximum(d, 1e-12), -0.5)
        d_inv_sqrt[d == 0] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        L = np.diag(d) - W

    if return_degree:
        return L, d
    return L


def extract_cluster_submatrix(matrix: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Extract principal submatrix corresponding to nodes in a specific cluster.

    Parameters
    ----------
    matrix : np.ndarray of shape (n, n)
        Parent matrix (e.g. Adjacency or Laplacian).
    indices : np.ndarray of shape (n_cluster,)
        Array of node indices in the cluster.

    Returns
    -------
    submatrix : np.ndarray of shape (n_cluster, n_cluster)
        Extracted submatrix.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    if np.any(indices < 0) or np.any(indices >= matrix.shape[0]):
        raise IndexError("indices contain an out-of-range node index.")
    return matrix[np.ix_(indices, indices)]


def validate_similarity_matrix(W: np.ndarray, *, require_connected: bool = False) -> np.ndarray:
    """Validate and return a symmetric, non-negative weighted adjacency matrix."""
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square two-dimensional matrix.")
    if W.shape[0] == 0:
        raise ValueError("W must contain at least one node.")
    if not np.all(np.isfinite(W)):
        raise ValueError("W must contain only finite values.")
    if np.any(W < 0):
        raise ValueError("W must be non-negative.")
    if not np.allclose(W, W.T, rtol=1e-10, atol=1e-12):
        raise ValueError("W must be symmetric.")
    if not np.allclose(np.diag(W), 0.0, atol=1e-12):
        raise ValueError("W must have a zero diagonal.")
    if require_connected and W.shape[0] > 1:
        n_components, _ = connected_components(csr_matrix(W > 0), directed=False)
        if n_components != 1:
            raise ValueError("W must be connected for spanning-tree sampling.")
    return W


def connect_knn_components(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Connect a sparse similarity graph with its closest cross-component edges.

    The function preserves all existing weights and only adds the minimum number
    of edges needed for a connected graph.  It is intended for k-NN graphs used
    by algorithms that require a connected support graph.
    """
    W = validate_similarity_matrix(W)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != W.shape[0]:
        raise ValueError("X must be a 2D array with one row per node in W.")
    if W.shape[0] <= 1:
        return W.copy()

    repaired = W.copy()
    n_components, labels = connected_components(csr_matrix(repaired > 0), directed=False)
    if n_components == 1:
        return repaired

    distances = compute_pairwise_distances(X)
    positive_weights = repaired[repaired > 0]
    bridge_weight = float(np.min(positive_weights)) if positive_weights.size else 1.0

    # Repeatedly join the two components with the closest pair of observations.
    while n_components > 1:
        best = None
        for left in range(n_components):
            left_nodes = np.where(labels == left)[0]
            for right in range(left + 1, n_components):
                right_nodes = np.where(labels == right)[0]
                local = distances[np.ix_(left_nodes, right_nodes)]
                idx = np.unravel_index(np.argmin(local), local.shape)
                candidate = (local[idx], left_nodes[idx[0]], right_nodes[idx[1]])
                if best is None or candidate[0] < best[0]:
                    best = candidate
        _, i, j = best
        repaired[i, j] = bridge_weight
        repaired[j, i] = bridge_weight
        n_components, labels = connected_components(csr_matrix(repaired > 0), directed=False)
    return repaired
