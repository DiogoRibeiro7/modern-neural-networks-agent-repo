"""Typed configuration for the relational track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RelationalConfig:
    """Architecture of the relational encoder.

    Attributes:
        d_model: Width of every row embedding.
        n_rounds: Message-passing rounds. Two is the minimum that can reach a linked row's
            attributes, so a one-round model is a deliberate ablation, not a smaller model.
        use_types: Give each table its own row projection.
        use_links: Pass messages along foreign keys. Off gives the homogeneous baseline.
        use_time: Encode elapsed time and gate messages on it.
    """

    d_model: int = 48
    n_rounds: int = 2
    use_types: bool = True
    use_links: bool = True
    use_time: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any size is outside its permitted range.
        """

        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_rounds <= 0:
            raise ValueError("n_rounds must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "d_model": self.d_model,
            "n_rounds": self.n_rounds,
            "use_types": self.use_types,
            "use_links": self.use_links,
            "use_time": self.use_time,
        }


@dataclass(frozen=True, slots=True)
class RelationalExperimentConfig:
    """Settings for the relational track's experiment suite.

    Attributes:
        seeds: Seeds used for every model.
        n_entities: Prediction points per regime, one per entity.
        n_products: Rows in the linked static table.
        orders_per_entity: Mean event count per entity.
        signals_per_entity: Mean distractor count per entity.
        horizon: Length of the time axis.
        recent_window: Window that counts in the temporal regime.
        max_events: Events kept per neighbourhood.
        max_distractors: Distractor rows kept per neighbourhood.
        epochs: Training epochs.
        batch_size: Training batch size.
        learning_rate_grid: Rates searched per architecture on validation.
        data_seed: Seed for database generation; held fixed so every model and every
            training seed sees the identical database.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_entities: int = 600
    n_products: int = 40
    orders_per_entity: float = 8.0
    signals_per_entity: float = 3.0
    horizon: float = 100.0
    recent_window: float = 20.0
    max_events: int = 8
    max_distractors: int = 6
    epochs: int = 60
    batch_size: int = 64
    learning_rate_grid: tuple[float, ...] = (1e-3, 3e-3)
    data_seed: int = 1729
