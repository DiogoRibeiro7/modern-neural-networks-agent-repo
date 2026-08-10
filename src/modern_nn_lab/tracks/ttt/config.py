"""Typed configuration for the Test-Time Training track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modern_nn_lab.tracks.ttt.layer import InnerModel, UpdateRule


@dataclass(frozen=True, slots=True)
class TTTConfig:
    """Configuration of a compact TTT model.

    Attributes:
        d_model: Model width.
        n_blocks: Number of residual blocks.
        d_inner: Width of the reconstruction views. ``None`` uses ``d_model // 2``.
        inner_model: ``"linear"`` for TTT-Linear, ``"mlp"`` for TTT-MLP.
        update_rule: ``"online"`` or ``"batch"``; the latter is provably linear attention.
        learner_updates: Enable the inner loop. ``False`` is the required ablation.
        layernorm_residual: Use the source's ``f(x) = x + LN(f_res(x))``.
    """

    d_model: int = 32
    n_blocks: int = 1
    d_inner: int | None = None
    inner_model: InnerModel = "linear"
    update_rule: UpdateRule = "online"
    learner_updates: bool = True
    layernorm_residual: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is outside its permitted range.
        """

        if self.d_model <= 0 or self.n_blocks <= 0:
            raise ValueError("d_model and n_blocks must be positive")
        if self.d_inner is not None and self.d_inner <= 0:
            raise ValueError("d_inner must be positive when set")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "d_model": self.d_model,
            "n_blocks": self.n_blocks,
            "d_inner": self.d_inner,
            "inner_model": self.inner_model,
            "update_rule": self.update_rule,
            "learner_updates": self.learner_updates,
            "layernorm_residual": self.layernorm_residual,
        }


@dataclass(frozen=True, slots=True)
class TTTExperimentConfig:
    """Settings for the TTT track's experiment suite.

    Matched to Tracks 02 and 03 wherever the tasks overlap, so records line up.

    Attributes:
        seeds: Seeds for every architecture and variant.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        learning_rate_grid: Rates searched per architecture on validation.
        n_sequences: Sequences generated per task before splitting.
        d_model: Reference width; baselines match its parameter count.
        n_blocks: Blocks in the TTT model and layers in the recurrent baselines.
        rebinding_pairs: Key-value pairs in the distribution-shift task.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 50
    batch_size: int = 64
    learning_rate_grid: tuple[float, ...] = (1e-3, 3e-3)
    n_sequences: int = 500
    d_model: int = 32
    n_blocks: int = 1
    rebinding_pairs: int = 3
