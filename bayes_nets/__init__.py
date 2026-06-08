"""
bayes_nets - A Python library for learning, sampling, and visualizing
Bayesian networks, focused on discrete variables.

Designed for use within Estimation of Distribution Algorithms (EDAs),
but fully usable as a standalone library.
"""

from bayes_nets.bayesian_network import BayesianNetwork
from bayes_nets.scoring import BICScoringMethod, AICScoringMethod, K2ScoringMethod
from bayes_nets.structure_learning import (
    K2StructureLearner,
    GreedyHillClimbLearner,
    StableHillClimbLearner,
    TabuHillClimbLearner,
)
from bayes_nets.parameter_learning import MLEParameterLearner
from bayes_nets.sampling import ProbabilisticLogicSampler
from bayes_nets.factorization import (
    CliqueFactorization,
    moralize,
    triangulate,
    junction_tree,
)
from bayes_nets.inference import MaxProductInference

__all__ = [
    "BayesianNetwork",
    "BICScoringMethod",
    "AICScoringMethod",
    "K2ScoringMethod",
    "K2StructureLearner",
    "GreedyHillClimbLearner",
    "StableHillClimbLearner",
    "TabuHillClimbLearner",
    "MLEParameterLearner",
    "ProbabilisticLogicSampler",
    "CliqueFactorization",
    "moralize",
    "triangulate",
    "junction_tree",
    "MaxProductInference",
]

__version__ = "0.1.0"
