"""Model scaffolding shared by more than one architecture track."""

from modern_nn_lab.models.sequence import (
    CausalTransformer,
    RecurrentBaseline,
    TokenSequenceModel,
    count_parameters,
    match_width_to_budget,
)

__all__ = [
    "CausalTransformer",
    "RecurrentBaseline",
    "TokenSequenceModel",
    "count_parameters",
    "match_width_to_budget",
]
