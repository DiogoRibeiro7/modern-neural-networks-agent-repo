"""Typed configuration for the Titans track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TitansConfig:
    """Configuration of a compact Titans Memory-as-Gate model.

    Attributes:
        d_model: Model width.
        n_heads: Attention heads in the short-term branch.
        window: Sliding-window size of the short-term branch.
        d_memory: Key/value/query width of the neural memory.
        memory_depth: Memory MLP layers; the source argues ``>= 2`` beats a linear memory.
        persistent_tokens: Learnable, data-independent prefix tokens (source eq. 19).
        use_short_term: Keep the attention branch. ``False`` is long-term-only.
        use_long_term: Keep the neural memory. ``False`` is short-term-only.
        memory_updates: Allow writes. ``False`` is the frozen-memory ablation.
        learning_rate_scale: Multiplies ``theta_t``; the update-rate ablation.
        use_momentum: Keep the past-surprise term of equation 14.
        use_forgetting: Keep the ``(1 - alpha_t)`` decay of equation 13.
    """

    d_model: int = 32
    n_heads: int = 2
    window: int = 6
    d_memory: int | None = None
    memory_depth: int = 2
    persistent_tokens: int = 2
    use_short_term: bool = True
    use_long_term: bool = True
    memory_updates: bool = True
    learning_rate_scale: float = 1.0
    use_momentum: bool = True
    use_forgetting: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is outside its permitted range.
        """

        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self.memory_depth < 1:
            raise ValueError("memory_depth must be at least 1")
        if self.persistent_tokens < 0:
            raise ValueError("persistent_tokens must be non-negative")
        if self.learning_rate_scale < 0:
            raise ValueError("learning_rate_scale must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "window": self.window,
            "d_memory": self.d_memory,
            "memory_depth": self.memory_depth,
            "persistent_tokens": self.persistent_tokens,
            "use_short_term": self.use_short_term,
            "use_long_term": self.use_long_term,
            "memory_updates": self.memory_updates,
            "learning_rate_scale": self.learning_rate_scale,
            "use_momentum": self.use_momentum,
            "use_forgetting": self.use_forgetting,
        }


@dataclass(frozen=True, slots=True)
class TitansExperimentConfig:
    """Settings for the Titans track's experiment suite.

    Attributes:
        seeds: Seeds for every architecture and variant.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        learning_rate_grid: Rates searched per architecture on validation.
        n_sequences: Sequences generated per task before splitting.
        d_model: Reference width; baselines match its parameter count.
        window: Short-term window; needles are placed beyond it on purpose.
        needle_distances: Distances between the stored fact and the query.
        diagnostic_sequences: Sequences used for the memory-trace artefact.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 40
    batch_size: int = 64
    learning_rate_grid: tuple[float, ...] = (1e-3, 3e-3)
    n_sequences: int = 500
    d_model: int = 32
    window: int = 6
    needle_distances: tuple[int, ...] = (4, 12)
    diagnostic_sequences: int = 64
