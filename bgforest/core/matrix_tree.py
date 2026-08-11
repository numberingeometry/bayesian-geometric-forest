"""
Matrix Tree Theorem Log-Determinant Solvers
============================================
Implements numerical procedures for computing spanning tree weight counts 
tau(C_k, W) and partition functions using Cholesky and LU decomposition 
log-determinants based on Kirchhoff's Matrix Tree Theorem.

Includes fast BLAS/LAPACK acceleration for large graphs (n > 5000).
"""

from typing import Union, Tuple, Optional
import numpy as np
from scipy.linalg import cholesky, lu, LinAlgError
from bgforest.core.graph import compute_laplacian, extract_cluster_submatrix


def fast_cholesky_logdet(L_reduced: np.ndarray) -> float:
    """
    Fast BLAS/LAPACK Cholesky log-determinant solver:
    ln det(A) = 2 * sum(ln(diag(L))) where A = L * L^T
    """
    chol_L = cholesky(L_reduced, lower=True, check_finite=False)
    return float(2.0 * np.sum(np.log(np.diag(chol_L))))


def log_spanning_trees_count(
    sub_laplacian: np.ndarray,
    remove_index: int = 0,
    jitter: float = 1e-12
) -> float:
    """
    Compute log of total weight sum of spanning trees for a graph component 
    using Kirchhoff's Matrix Tree Theorem:
    
    ln tau(C_k, W) = ln det( L_{C_k, (uu)} )

    Parameters
    ----------
    sub_laplacian : np.ndarray of shape (m, m)
        Unnormalized Laplacian matrix of the cluster component.
    remove_index : int, default=0
        Index of the row/column to remove (0 <= remove_index < m).
    jitter : float, default=1e-12
        Small diagonal regularization added if matrix is near-singular.

    Returns
    -------
    log_tau : float
        Log weight sum of all spanning trees in the cluster.
        Returns -inf if graph component is disconnected or singular.
    """
    m = sub_laplacian.shape[0]

    # Single-node cluster has exactly 1 spanning tree of weight 1 (log(1) = 0)
    if m <= 1:
        return 0.0

    keep_indices = [i for i in range(m) if i != remove_index]
    L_reduced = sub_laplacian[np.ix_(keep_indices, keep_indices)]

    # 1. Attempt Fast Cholesky decomposition (fastest and most stable for SPD)
    try:
        return fast_cholesky_logdet(L_reduced)
    except (LinAlgError, ValueError):
        pass

    # 2. Fallback to slogdet
    try:
        sign, logdet = np.linalg.slogdet(L_reduced)
        if sign > 0:
            return float(logdet)
        elif sign == 0:
            return -np.inf
    except LinAlgError:
        pass

    # 3. Add small diagonal jitter for near-singular boundary cases
    try:
        L_reg = L_reduced + np.eye(m - 1) * jitter
        sign, logdet = np.linalg.slogdet(L_reg)
        if sign > 0:
            return float(logdet)
    except LinAlgError:
        pass

    return -np.inf


def compute_cluster_log_spanning_trees(
    W: np.ndarray,
    cluster_indices: np.ndarray
) -> float:
    """
    Compute log weight sum of spanning trees for a node cluster from global adjacency W.
    """
    indices = np.asarray(cluster_indices, dtype=np.int64)
    m = len(indices)
    if m <= 1:
        return 0.0

    W_sub = extract_cluster_submatrix(W, indices)
    L_sub = compute_laplacian(W_sub, normed=False)
    return log_spanning_trees_count(L_sub)


def compute_forest_log_spanning_trees(
    W: np.ndarray,
    partition: np.ndarray
) -> float:
    """
    Compute total log weight sum of spanning forest for a full data partition:
    
    ln tau(C, W) = sum_{k=1}^K ln tau(C_k, W)
    """
    partition = np.asarray(partition, dtype=np.int64)
    unique_clusters = np.unique(partition)

    total_log_tau = 0.0
    for k in unique_clusters:
        indices = np.where(partition == k)[0]
        log_tau_k = compute_cluster_log_spanning_trees(W, indices)
        if np.isneginf(log_tau_k):
            return -np.inf
        total_log_tau += log_tau_k

    return float(total_log_tau)
