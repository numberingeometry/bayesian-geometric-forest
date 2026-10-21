"""
bgforest: Bayesian spanning forest clustering with exact tree sampling
=======================================================================

Clusters are trees. The posterior over trees is a weighted spanning-tree law on an
augmented graph (Duan & Roy, JASA 2024), so forests can be drawn exactly with
Wilson's algorithm and edge probabilities are available in closed form.
"""

from bgforest.core.forest_model import ForestKernel
from bgforest.models.backbone import SpanningTreeBackbone
from bgforest.models.bsf import BayesianSpanningForest
from bgforest.models.constrained import ConstrainedBayesianSpanningForest
from bgforest.samplers.wilson import WilsonLERWSampler, WilsonSampler

__version__ = "0.3.0"
__all__ = [
    "BayesianSpanningForest",
    "ConstrainedBayesianSpanningForest",
    "ForestKernel",
    "SpanningTreeBackbone",
    "WilsonSampler",
    "WilsonLERWSampler",
]
