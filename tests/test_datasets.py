"""
Unit Tests for Synthetic and Single-Cell Dataset Generators
"""

import numpy as np
import pytest
from bgforest.datasets.synthetic import (
    make_two_moons,
    make_concentric_circles,
    make_spirals,
    make_anisotropic_blobs,
    make_misspecified_mixtures,
)
from bgforest.datasets.single_cell import simulate_scrna_data, preprocess_scrna_data


def test_synthetic_dataset_generators():
    X_m, y_m = make_two_moons(n_samples=50, random_state=42)
    assert X_m.shape == (50, 2)
    assert len(np.unique(y_m)) == 2

    X_c, y_c = make_concentric_circles(n_samples=50, random_state=42)
    assert X_c.shape == (50, 2)
    assert len(np.unique(y_c)) == 2

    X_s, y_s = make_spirals(n_samples=60, n_arms=3, random_state=42)
    assert X_s.shape == (60, 2)
    assert len(np.unique(y_s)) == 3

    X_a, y_a = make_anisotropic_blobs(n_samples=60, random_state=42)
    assert X_a.shape == (60, 2)
    assert len(np.unique(y_a)) == 3

    X_miss, y_miss = make_misspecified_mixtures(n_samples=60, random_state=42)
    assert X_miss.shape == (60, 2)
    assert len(np.unique(y_miss)) == 3


def test_scrna_simulation_and_preprocessing():
    counts, labels, names = simulate_scrna_data(
        n_cells=60, n_genes=100, n_cell_types=3, random_state=42
    )
    assert counts.shape == (60, 100)
    assert len(labels) == 60
    assert len(names) == 3

    pca_emb, hvg, pca_model = preprocess_scrna_data(
        counts, n_hvg=50, n_pcs=10, random_state=42
    )
    assert pca_emb.shape == (60, 10)
    assert hvg.shape == (60, 50)
