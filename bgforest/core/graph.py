"""
Graph Construction and Laplacian Matrix Operations
===================================================
Provides graph adjacency building, kernel computations, and Laplacian matrix
algebra required for Bayesian Spanning Forest graph partitioning.
"""

from typing import Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.csgraph import connected_components


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
    return squareform(pdist(X, metric='euclidean'))


def build_rbf_similarity(
    X: np.ndarray,
    gamma: Optional[float] = None,
    sigma: Optional[float] = None,
    adaptive_bandwidth: bool = False,
    k_adaptive: int = 7
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
        W = np.exp(-(D ** 2) / (2.0 * sigma_matrix))
    else:
        if gamma is not None:
            bandwidth_sq = 1.0 / (2.0 * gamma)
        elif sigma is not None:
            bandwidth_sq = sigma ** 2
        else:
            # Default heuristic: median pairwise distance squared
            median_dist = np.median(D)
            bandwidth_sq = (median_dist ** 2) if median_dist > 0 else 1.0

        W = np.exp(-(D ** 2) / (2.0 * bandwidth_sq))

    np.fill_diagonal(W, 0.0)
    return W


def build_knn_similarity(
    X: np.ndarray,
    n_neighbors: int = 10,
    metric: str = "euclidean",
    mode: str = "distance",
    symmetric: bool = True
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

    n = X.shape[0]
    k = min(n_neighbors + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
    distances, indices = nn.kneighbors(X)

    W = np.zeros((n, n), dtype=np.float64)

    if mode == "distance":
        # Estimate sigma from median neighbor distance
        sigma = np.median(distances[:, 1:])
        sigma = max(sigma, 1e-6)
        weights = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
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
    W: np.ndarray,
    normed: bool = False,
    return_degree: bool = False
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


def extract_cluster_submatrix(
    matrix: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
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
    return matrix[np.ix_(indices, indices)]
