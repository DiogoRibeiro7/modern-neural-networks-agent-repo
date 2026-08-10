"""Nested Learning: optimizers and architectures as nested optimization levels.

**Research prototype.** This track implements the level decomposition the primary source
derives for gradient descent and its momentum variant, plus the self-referential learning
rule. It deliberately does **not** implement the Continuum Memory System or the Hope
architecture; see ``docs/nested_learning_audit.md`` for why, and ``README.md`` in this
package for the equation-to-code mapping.
"""

from modern_nn_lab.tracks.hope.learner import (
    LearnerConfig,
    NestedLearner,
    SelfReferentialLearner,
    ValueFn,
)
from modern_nn_lab.tracks.hope.levels import (
    DataMemory,
    DataMemoryConfig,
    GradientMemory,
    Level,
    LevelState,
    NestedTrace,
    local_surprise_signal,
    weight_gradient,
)

__all__ = [
    "DataMemory",
    "DataMemoryConfig",
    "GradientMemory",
    "LearnerConfig",
    "Level",
    "LevelState",
    "NestedLearner",
    "NestedTrace",
    "SelfReferentialLearner",
    "ValueFn",
    "local_surprise_signal",
    "weight_gradient",
]
