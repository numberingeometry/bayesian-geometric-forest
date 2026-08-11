"""
Bayesian Geometric Forest (`bgforest`)
======================================
A high-performance Python package for Bayesian Spanning Forests, Forest Processes, 
Bayesian Spanning Trees, Bayesian Distance Clustering, and Exact Tree Samplers.
"""

from bgforest.models.bsf import BayesianSpanningForest
from bgforest.models.bst import BayesianSpanningTree
from bgforest.models.distance_clustering import BayesianDistanceClustering
from bgforest.models.semi_supervised import ConstrainedBayesianSpanningForest
from bgforest.models.forest_process import ForestProcess
from bgforest.mcmc.sampler import BSFMCMCSampler
from bgforest.samplers.wilson import WilsonLERWSampler

__version__ = "0.2.0"
__all__ = [
    "BayesianSpanningForest",
    "BayesianSpanningTree",
    "BayesianDistanceClustering",
    "ConstrainedBayesianSpanningForest",
    "ForestProcess",
    "BSFMCMCSampler",
    "WilsonLERWSampler",
]
