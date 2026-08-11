"""
Bayesian Geometric Forest (`bgforest`)
======================================
A Python library implementing Bayesian Spanning Forest and Forest Process models
for robust graphical model-based spectral clustering.

Based on:
- Duan & Roy (2022/2023): Spectral Clustering, Bayesian Spanning Forest, and Forest Process (JASA)
- Zheng, Duan & Roy (2024): Consistency of Graphical Model-based Clustering: Robust Clustering using Bayesian Spanning Forest
"""

__version__ = "0.1.0"
__author__ = "Bobby Zhang"

from bgforest.models.bsf import BayesianSpanningForest
from bgforest.models.forest_process import ForestProcess

__all__ = [
    "BayesianSpanningForest",
    "ForestProcess",
]
