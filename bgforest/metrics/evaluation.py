"""
Clustering Evaluation and Uncertainty Metrics
=============================================
Provides quantitative metrics including Adjusted Rand Index (ARI), 
Normalized Mutual Information (NMI), point-wise Bayesian assignment entropy, 
and empirical bounds on misclassification rate (Zheng, Duan & Roy, 2024).
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compute_clustering_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard external clustering benchmark metrics.

    Parameters
    ----------
    labels_true : np.ndarray
        Ground truth cluster labels.
    labels_pred : np.ndarray
        Predicted cluster assignment labels.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary containing ARI and NMI metrics.
    """
    labels_true = np.asarray(labels_true, dtype=np.int64)
    labels_pred = np.asarray(labels_pred, dtype=np.int64)

    ari = float(adjusted_rand_score(labels_true, labels_pred))
    nmi = float(normalized_mutual_info_score(labels_true, labels_pred))

    return {
        "ARI": ari,
        "NMI": nmi
    }


def compute_uncertainty_entropy(proba: np.ndarray) -> np.ndarray:
    """
    Compute point-wise Shannon assignment entropy quantifying Bayesian uncertainty.

    H(p_i) = - sum_{k=1}^K p_{ik} * ln(p_{ik} + eps)

    Parameters
    ----------
    proba : np.ndarray of shape (n_samples, n_clusters)
        Soft assignment probabilities from `predict_proba()`.

    Returns
    -------
    entropy : np.ndarray of shape (n_samples,)
        Point-wise uncertainty entropy values (0 = deterministic, high = ambiguous boundary).
    """
    proba = np.asarray(proba, dtype=np.float64)
    eps = 1e-12
    proba_clamped = np.maximum(proba, eps)
    entropy = -np.sum(proba_clamped * np.log(proba_clamped), axis=1)
    return entropy


def compute_misclassification_bound(
    X: np.ndarray,
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    sigma: float = 1.0
) -> Tuple[float, float]:
    """
    Compute empirical misclassification rate and theoretical upper bound (Zheng, Duan & Roy, 2024).

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Data matrix.
    labels_true : np.ndarray
        Ground truth cluster labels.
    labels_pred : np.ndarray
        Model predicted cluster labels.
    sigma : float, default=1.0
        Kernel bandwidth parameter.

    Returns
    -------
    empirical_error : float
        Observed misclassification rate under optimal label permutation.
    bound_estimate : float
        Theoretical asymptotic upper bound estimate exp(- Delta^2 / (8 * sigma^2)).
    """
    from scipy.optimize import linear_sum_assignment

    labels_true = np.asarray(labels_true, dtype=np.int64)
    labels_pred = np.asarray(labels_pred, dtype=np.int64)

    # Compute optimal permutation error via Hungarian algorithm
    unique_t = np.unique(labels_true)
    unique_p = np.unique(labels_pred)
    n_t, n_p = len(unique_t), len(unique_p)
    cost_matrix = np.zeros((n_t, n_p), dtype=np.float64)

    for i, t in enumerate(unique_t):
        for j, p in enumerate(unique_p):
            cost_matrix[i, j] = -np.sum((labels_true == t) & (labels_pred == p))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    correct_count = -cost_matrix[row_ind, col_ind].sum()
    empirical_error = 1.0 - (correct_count / float(len(labels_true)))

    # Estimate min separation distance Delta between cluster centroids
    centroids = []
    for t in unique_t:
        idx = np.where(labels_true == t)[0]
        centroids.append(np.mean(X[idx], axis=0))
    
    centroids = np.array(centroids)
    if len(centroids) >= 2:
        from scipy.spatial.distance import pdist
        min_dist = np.min(pdist(centroids))
    else:
        min_dist = 1.0

    bound_estimate = float(np.exp(-(min_dist ** 2) / (8.0 * (sigma ** 2))))

    return float(empirical_error), bound_estimate
