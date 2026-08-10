"""Typed configuration for the Mamba-3 track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Mamba3Config:
    """Configuration of a compact Mamba-3 model.

    Each of the three flags corresponds to one contribution of the primary source, and
    turning it off recovers the prior method exactly:

    - ``trapezoidal=False`` gives exponential-Euler, the Mamba-1/2 discretization;
    - ``rotary=False`` gives a real non-negative transition, i.e. a Mamba-2-style SSM;
    - ``rank=1`` gives the SISO recurrence.

    Attributes:
        d_model: Model width, divisible by ``heads``.
        n_blocks: Number of residual blocks.
        heads: SSM heads.
        state_size: State size ``N`` per head; even when ``rotary`` is set.
        rank: MIMO rank ``R``.
        trapezoidal: Enable the exponential-trapezoidal discretization.
        rotary: Enable data-dependent rotations (complex-valued dynamics).
    """

    d_model: int = 32
    n_blocks: int = 1
    heads: int = 2
    state_size: int = 8
    rank: int = 1
    trapezoidal: bool = True
    rotary: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is outside its permitted range.
        """

        if self.d_model <= 0 or self.n_blocks <= 0 or self.heads <= 0:
            raise ValueError("d_model, n_blocks, and heads must be positive")
        if self.d_model % self.heads != 0:
            raise ValueError("d_model must be divisible by heads")
        if self.state_size <= 0 or self.rank <= 0:
            raise ValueError("state_size and rank must be positive")
        if self.rotary and self.state_size % 2 != 0:
            raise ValueError("state_size must be even when rotary is enabled")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "d_model": self.d_model,
            "n_blocks": self.n_blocks,
            "heads": self.heads,
            "state_size": self.state_size,
            "rank": self.rank,
            "trapezoidal": self.trapezoidal,
            "rotary": self.rotary,
        }


@dataclass(frozen=True, slots=True)
class Mamba3ExperimentConfig:
    """Settings for the Mamba-3 track's experiment suite.

    Deliberately identical to the xLSTM track's settings — same tasks, same sequence
    counts, same epochs, same seeds — so the two tracks' records are directly comparable
    without rerunning anything.

    Attributes:
        seeds: Seeds for every architecture and variant.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        learning_rate_grid: Rates searched per architecture on the validation split.
        n_sequences: Sequences generated per task before splitting.
        d_model: Reference width; baselines match its parameter count.
        n_blocks: Blocks in the SSM and layers in the recurrent baselines.
        state_size: State size per head.
        rank: MIMO rank of the default configuration.
        throughput_lengths: Sequence lengths used for the cost-scaling measurement.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 50
    batch_size: int = 64
    learning_rate_grid: tuple[float, ...] = (1e-3, 3e-3)
    n_sequences: int = 500
    d_model: int = 32
    n_blocks: int = 1
    heads: int = 2
    state_size: int = 8
    rank: int = 2
    throughput_lengths: tuple[int, ...] = (16, 32, 64, 128)
