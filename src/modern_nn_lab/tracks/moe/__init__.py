"""Sparse mixture of experts: conditional computation with transparent routing.

The claim this track examines is that a model can hold many parameters while spending only
a few of them per token. That makes **activated parameters** and **total parameters**
different numbers, and a comparison quoting the wrong one is not a comparison — so both are
recorded for every run, alongside an analytic FLOP estimate and a measured throughput.

The prompt's acceptance criterion is that expert utilization and routing entropy must be
reported, because accuracy alone is insufficient. A sparse layer can reach a good metric
while routing every token to one expert, or while silently dropping a third of them for
lack of capacity. :class:`~modern_nn_lab.tracks.moe.router.RoutingInfo` therefore travels
out of every forward pass beside the output, and its statistics reach the experiment
record rather than a log line.
"""

from modern_nn_lab.tracks.moe.config import LayerKind, MoEConfig, MoEExperimentConfig
from modern_nn_lab.tracks.moe.data import (
    MixtureTask,
    generate,
    specialization_matrix,
    specialization_purity,
)
from modern_nn_lab.tracks.moe.layer import (
    DenseFFN,
    DenseMoELayer,
    SparseMoELayer,
    build_expert,
    expert_flops,
)
from modern_nn_lab.tracks.moe.model import MixtureModel, build_layer
from modern_nn_lab.tracks.moe.router import RoutingInfo, TopKRouter

__all__ = [
    "DenseFFN",
    "DenseMoELayer",
    "LayerKind",
    "MixtureModel",
    "MixtureTask",
    "MoEConfig",
    "MoEExperimentConfig",
    "RoutingInfo",
    "SparseMoELayer",
    "TopKRouter",
    "build_expert",
    "build_layer",
    "expert_flops",
    "generate",
    "specialization_matrix",
    "specialization_purity",
]
