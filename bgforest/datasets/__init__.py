"""Public synthetic and single-cell dataset helpers."""

from bgforest.datasets.single_cell import (
    load_real_scrna_benchmark,
    preprocess_scrna_data,
    simulate_scrna_data,
)
from bgforest.datasets.synthetic import (
    make_anisotropic_blobs,
    make_concentric_circles,
    make_misspecified_mixtures,
    make_spirals,
    make_two_moons,
)

__all__ = [
    "load_real_scrna_benchmark",
    "preprocess_scrna_data",
    "simulate_scrna_data",
    "make_anisotropic_blobs",
    "make_concentric_circles",
    "make_misspecified_mixtures",
    "make_spirals",
    "make_two_moons",
]
