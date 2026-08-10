"""Titans: neural long-term memory that learns to memorize at test time.

Mechanism under study: a memory whose weights are updated online by a surprise signal
with momentum and adaptive forgetting. See ``README.md`` in this package for the
equation-to-code mapping, the choice of architecture variant, and the deviations from the
primary source.
"""

from modern_nn_lab.tracks.titans.config import TitansConfig, TitansExperimentConfig
from modern_nn_lab.tracks.titans.memory import MemoryState, MemoryTrace, NeuralMemory
from modern_nn_lab.tracks.titans.model import SlidingWindowAttention, TitansMAG

__all__ = [
    "MemoryState",
    "MemoryTrace",
    "NeuralMemory",
    "SlidingWindowAttention",
    "TitansConfig",
    "TitansExperimentConfig",
    "TitansMAG",
]
