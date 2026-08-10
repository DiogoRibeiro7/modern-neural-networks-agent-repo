"""Typed configuration for the xLSTM track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modern_nn_lab.tracks.xlstm.cells import GateKind
from modern_nn_lab.tracks.xlstm.model import BlockKind


@dataclass(frozen=True, slots=True)
class XLSTMConfig:
    """Configuration of a compact xLSTM.

    Attributes:
        d_model: Model width ``D``.
        n_blocks: Number of residual blocks.
        block_kinds: Cell type per block. ``None`` means all ``mlstm``.
        heads: Memory heads used by ``mlstm`` blocks.
        input_gate: Input-gate kind. ``"sigmoid"`` is the gating ablation.
        forget_gate: Forget-gate kind.
    """

    d_model: int = 64
    n_blocks: int = 1
    block_kinds: tuple[BlockKind, ...] | None = None
    heads: int = 2
    input_gate: GateKind = "exponential"
    forget_gate: GateKind = "sigmoid"

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is outside its permitted range.
        """

        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        if self.heads <= 0:
            raise ValueError("heads must be positive")
        if self.d_model % self.heads != 0:
            raise ValueError("d_model must be divisible by heads")
        if self.block_kinds is not None and len(self.block_kinds) != self.n_blocks:
            raise ValueError("block_kinds must have exactly n_blocks entries")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "d_model": self.d_model,
            "n_blocks": self.n_blocks,
            "block_kinds": list(self.block_kinds) if self.block_kinds else None,
            "heads": self.heads,
            "input_gate": self.input_gate,
            "forget_gate": self.forget_gate,
        }


@dataclass(frozen=True, slots=True)
class XLSTMExperimentConfig:
    """Settings for the xLSTM track's experiment suite.

    Attributes:
        seeds: Seeds for every architecture and variant.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        learning_rate_grid: Learning rates searched per architecture on the validation
            split, so no architecture is penalized by a rate that suits another.
        n_sequences: Sequences generated per task, before splitting.
        d_model: Reference width for the xLSTM; baselines match its parameter count.
        n_blocks: Blocks in the xLSTM and layers in the recurrent baselines.
        context_lengths: Sequence lengths used for the context-scaling study.
        scaling_seeds: Reduced seed budget for the scaling study.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 50
    batch_size: int = 64
    learning_rate_grid: tuple[float, ...] = (1e-3, 3e-3)
    n_sequences: int = 500
    d_model: int = 32
    n_blocks: int = 1
    context_lengths: tuple[int, ...] = (8, 16)
    scaling_seeds: tuple[int, ...] = (0, 1, 2)
