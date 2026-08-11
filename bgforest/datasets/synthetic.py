"""
Synthetic Geometric Dataset Generators
=======================================
Generates complex manifold datasets (Two Moons, Concentric Circles, Spirals, 
Anisotropic Blobs, Heavy-tailed Misspecified Mixtures) for benchmarking 
graph-based spectral and Bayesian Spanning Forest algorithms.
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs


def make_two_moons(
    n_samples: int = 200,
    noise: float = 0.08,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate non-linear Interleaved Two Moons dataset."""
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def make_concentric_circles(
    n_samples: int = 200,
    factor: float = 0.5,
    noise: float = 0.05,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Concentric Circles dataset."""
    return make_circles(
        n_samples=n_samples, factor=factor, noise=noise, random_state=random_state
    )


def make_spirals(
    n_samples: int = 300,
    n_arms: int = 2,
    noise: float = 0.1,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Multi-branch Spiral dataset."""
    rng = np.random.RandomState(random_state)
    n_per_arm = n_samples // n_arms
    
    X_list, y_list = [], []
    for arm in range(n_arms):
        r = np.linspace(0.2, 1.0, n_per_arm)
        theta = np.linspace(arm * 2 * np.pi / n_arms, (arm + 1) * 2 * np.pi, n_per_arm)
        dx = noise * rng.randn(n_per_arm)
        dy = noise * rng.randn(n_per_arm)
        
        x = r * np.cos(theta) + dx
        y = r * np.sin(theta) + dy
        
        X_list.append(np.column_stack([x, y]))
        y_list.append(np.full(n_per_arm, arm, dtype=np.int64))

    return np.vstack(X_list), np.concatenate(y_list)


def make_anisotropic_blobs(
    n_samples: int = 300,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Anisotropic Elongated Gaussian Blobs."""
    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=random_state)
    # Apply linear transformation for anisotropic stretch
    transformation = np.array([[0.6, -0.6], [-0.4, 0.8]])
    X = np.dot(X, transformation)
    return X, y


def make_misspecified_mixtures(
    n_samples: int = 300,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Heavy-tailed Non-Gaussian Misspecified Clusters.

    Models data generated from non-Gaussian t-distributions with asymmetric skewness,
    demonstrating the consistency of Bayesian Spanning Forests under model misspecification
    (Zheng, Duan & Roy, 2024).
    """
    rng = np.random.RandomState(random_state)
    n_per_cluster = n_samples // 3

    # Cluster 1: Heavy-tailed Student-t cluster
    df = 2.0
    c1 = rng.standard_t(df, size=(n_per_cluster, 2)) + np.array([-4.0, 0.0])

    # Cluster 2: Curved parabolic banana shape
    t = np.linspace(-1.5, 1.5, n_per_cluster)
    c2_x = t + 0.1 * rng.randn(n_per_cluster)
    c2_y = t**2 + 0.1 * rng.randn(n_per_cluster) + 4.0
    c2 = np.column_stack([c2_x, c2_y])

    # Cluster 3: Dense compact cluster
    c3 = 0.5 * rng.randn(n_per_cluster, 2) + np.array([4.0, -2.0])

    X = np.vstack([c1, c2, c3])
    y = np.concatenate([
        np.zeros(n_per_cluster, dtype=np.int64),
        np.ones(n_per_cluster, dtype=np.int64),
        np.full(n_per_cluster, 2, dtype=np.int64)
    ])
    return X, y
