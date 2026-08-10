"""Test-Time Training: the recurrent hidden state is itself a learner.

Mechanism under study: the hidden state is the weight matrix of an inner model, updated
by gradient descent on a self-supervised reconstruction loss *during the forward pass*.
See ``README.md`` in this package for the equation-to-code mapping and the deviations
from the primary source.
"""

from modern_nn_lab.tracks.ttt.config import TTTConfig, TTTExperimentConfig
from modern_nn_lab.tracks.ttt.layer import InnerModel, LearnerState, TTTLayer, UpdateRule
from modern_nn_lab.tracks.ttt.model import TTT

__all__ = [
    "TTT",
    "InnerModel",
    "LearnerState",
    "TTTConfig",
    "TTTExperimentConfig",
    "TTTLayer",
    "UpdateRule",
]
