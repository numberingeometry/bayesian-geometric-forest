"""
Clustering Evaluation and Uncertainty Metrics
=============================================
Provides quantitative metrics including Adjusted Rand Index (ARI),
Normalized Mutual Information (NMI), point-wise Bayesian assignment entropy,
and the empirical misclassification rate under the best label matching.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def compute_clustering_metrics(
    labels_true: np.ndarray, labels_pred: np.ndarray
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

    return {"ARI": ari, "NMI": nmi}


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


def compute_misclassification_rate(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Fraction of points misassigned under the best one-to-one matching of labels.

    The matching is found with the Hungarian algorithm, so the rate is invariant to
    label permutation. Extra predicted clusters count as errors.
    """
    from scipy.optimize import linear_sum_assignment

    labels_true = np.asarray(labels_true, dtype=np.int64)
    labels_pred = np.asarray(labels_pred, dtype=np.int64)
    if labels_true.shape != labels_pred.shape or labels_true.ndim != 1:
        raise ValueError("labels_true and labels_pred must be 1D arrays of equal length.")

    unique_t = np.unique(labels_true)
    unique_p = np.unique(labels_pred)
    overlap = np.zeros((len(unique_t), len(unique_p)), dtype=np.float64)
    for i, t in enumerate(unique_t):
        for j, p in enumerate(unique_p):
            overlap[i, j] = np.sum((labels_true == t) & (labels_pred == p))

    row_ind, col_ind = linear_sum_assignment(-overlap)
    correct = overlap[row_ind, col_ind].sum()
    return float(1.0 - correct / float(len(labels_true)))
