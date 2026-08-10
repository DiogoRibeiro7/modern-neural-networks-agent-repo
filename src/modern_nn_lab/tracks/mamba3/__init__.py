"""Mamba-3: selective state-space models with expressive discretization and rotation.

Mechanisms under study, each implemented from the primary source and each removable by a
flag: exponential-trapezoidal discretization, complex-valued dynamics expressed as
data-dependent rotary embeddings, and a MIMO state update. See ``README.md`` in this
package for the equation-to-code mapping and the deviations from the source.
"""

from modern_nn_lab.tracks.mamba3.config import Mamba3Config, Mamba3ExperimentConfig
from modern_nn_lab.tracks.mamba3.model import Mamba3, parity_reference
from modern_nn_lab.tracks.mamba3.ssm import SelectiveSSM, SSMState, rotate_pairs

__all__ = [
    "Mamba3",
    "Mamba3Config",
    "Mamba3ExperimentConfig",
    "SSMState",
    "SelectiveSSM",
    "parity_reference",
    "rotate_pairs",
]
