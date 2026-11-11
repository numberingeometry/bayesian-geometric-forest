"""
bgforest: random trees on data, exact at every temperature
==========================================================

``TreeLaw`` is the law of a spanning tree when a point cloud was grown by Gaussian
diffusion along it: one temperature dial between the uniform random tree and the
minimum spanning tree (single linkage). ``BayesianSpanningForest`` adds roots, so
clusters are trees (Duan & Roy, JASA 2024). Both are weighted spanning-tree laws, so
trees can be drawn exactly with Wilson's algorithm and edge probabilities are
available in closed form.
"""

from bgforest.core.forest_model import ForestKernel
from bgforest.models.backbone import SpanningTreeBackbone
from bgforest.models.bsf import BayesianSpanningForest
from bgforest.models.constrained import ConstrainedBayesianSpanningForest
from bgforest.samplers.wilson import WilsonLERWSampler, WilsonSampler
from bgforest.treelaw import TreeLaw

__version__ = "0.4.0"
__all__ = [
    "BayesianSpanningForest",
    "ConstrainedBayesianSpanningForest",
    "ForestKernel",
    "SpanningTreeBackbone",
    "TreeLaw",
    "WilsonSampler",
    "WilsonLERWSampler",
]
