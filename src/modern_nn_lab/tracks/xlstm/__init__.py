"""xLSTM: extended LSTM with exponential gating and matrix memory.

Mechanism under study: replacing the sigmoid input gate with an exponential one, and
carrying an explicit normalizer state so the unbounded gate stays usable. See
``README.md`` in this package for the mathematical specification, the equation-to-code
mapping, and the deviations from the primary source.
"""

from modern_nn_lab.models.sequence import (
    CausalTransformer,
    RecurrentBaseline,
    TokenSequenceModel,
    count_parameters,
    match_width_to_budget,
)
from modern_nn_lab.tracks.xlstm.cells import (
    GateKind,
    MLSTMCell,
    MLSTMState,
    SLSTMCell,
    SLSTMState,
)
from modern_nn_lab.tracks.xlstm.config import XLSTMConfig, XLSTMExperimentConfig
from modern_nn_lab.tracks.xlstm.model import XLSTM

__all__ = [
    "XLSTM",
    "CausalTransformer",
    "GateKind",
    "MLSTMCell",
    "MLSTMState",
    "RecurrentBaseline",
    "SLSTMCell",
    "SLSTMState",
    "TokenSequenceModel",
    "XLSTMConfig",
    "XLSTMExperimentConfig",
    "count_parameters",
    "match_width_to_budget",
]
