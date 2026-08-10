"""Synthetic tasks shared across tracks.

A task lives here once a second track needs it. Until then it belongs to the track that
introduced it, per the "no premature framework" rule in ``docs/architecture.md``.
"""

from modern_nn_lab.experiments.tasks.sequence import (
    IGNORE_INDEX,
    SequenceSplit,
    make_copy_task,
    make_rebinding_task,
    make_selective_recall_task,
    make_state_tracking_task,
    masked_accuracy,
    masked_cross_entropy,
)

__all__ = [
    "IGNORE_INDEX",
    "SequenceSplit",
    "make_copy_task",
    "make_rebinding_task",
    "make_selective_recall_task",
    "make_state_tracking_task",
    "masked_accuracy",
    "masked_cross_entropy",
]
