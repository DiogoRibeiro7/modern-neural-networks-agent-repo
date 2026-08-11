"""Typed configuration for the flow-matching track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class FlowConfig:
    """Architecture of the velocity-field network.

    Attributes:
        dim: Dimensionality of the space being transported.
        d_hidden: Hidden width.
        n_layers: Hidden layers.
        n_frequencies: Fourier features used to embed time.
    """

    dim: int = 2
    d_hidden: int = 128
    n_layers: int = 3
    n_frequencies: int = 6

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is outside its permitted range.
        """

        if min(self.dim, self.d_hidden, self.n_layers, self.n_frequencies) <= 0:
            raise ValueError("dim, d_hidden, n_layers and n_frequencies must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "dim": self.dim,
            "d_hidden": self.d_hidden,
            "n_layers": self.n_layers,
            "n_frequencies": self.n_frequencies,
        }


@dataclass(frozen=True, slots=True)
class FlowExperimentConfig:
    """Settings for the flow-matching experiment suite.

    Attributes:
        seeds: Seeds used for every model.
        n_train: Target samples drawn per training step.
        steps: Training steps.
        batch_size: Pairs per step.
        learning_rate: Adam learning rate.
        n_eval: Samples generated for evaluation.
        solver_steps: Step counts swept when sampling, to expose discretization error.
        eval_method: Solver used for the headline sample-quality numbers.
        paths: Probability paths compared.
        gaussian_mean: Mean of the analytic Gaussian target.
        gaussian_scale: Standard deviation of the analytic Gaussian target.
        data_seed: Seed for the evaluation draws, held fixed across models.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_train: int = 4096
    steps: int = 3000
    batch_size: int = 256
    learning_rate: float = 3e-3
    n_eval: int = 2048
    solver_steps: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128)
    eval_method: Literal["euler", "midpoint"] = "midpoint"
    paths: tuple[str, ...] = ("linear", "trigonometric")
    gaussian_mean: float = 2.0
    gaussian_scale: float = 0.5
    data_seed: int = 1729
