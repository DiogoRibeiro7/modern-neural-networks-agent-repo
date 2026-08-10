"""Prior-Fitted Networks: prediction as a forward pass over a labelled context.

This package holds **two deliverables that must not be conflated**, as the track prompt
requires:

- :mod:`~modern_nn_lab.tracks.pfn.model` — a PFN built here from first principles and
  fitted to a controllable synthetic prior;
- :mod:`~modern_nn_lab.tracks.pfn.reference` — an adapter for the *official* pre-trained
  TabPFN checkpoint, which is a different kind of object entirely.

See ``README.md`` in this package for why they are kept apart and what may be said when
comparing them.
"""

from modern_nn_lab.tracks.pfn.config import PFNConfig, PFNExperimentConfig
from modern_nn_lab.tracks.pfn.model import PriorFittedNetwork
from modern_nn_lab.tracks.pfn.prior import (
    PRIORS,
    ImbalancedPrior,
    LinearPrior,
    MLPPrior,
    TaskBatch,
    TaskPrior,
    XORPrior,
    build_prior,
)
from modern_nn_lab.tracks.pfn.reference import (
    build_tabpfn_classifier,
    pretraining_advantage_note,
    tabpfn_available,
)

__all__ = [
    "PRIORS",
    "ImbalancedPrior",
    "LinearPrior",
    "MLPPrior",
    "PFNConfig",
    "PFNExperimentConfig",
    "PriorFittedNetwork",
    "TaskBatch",
    "TaskPrior",
    "XORPrior",
    "build_prior",
    "build_tabpfn_classifier",
    "pretraining_advantage_note",
    "tabpfn_available",
]
