"""Typed configuration for the Prior-Fitted Network track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PFNConfig:
    """Architecture of a compact Prior-Fitted Network.

    Attributes:
        n_features: Input width of the tasks this model is fitted to.
        n_classes: Number of label classes.
        d_model: Model width.
        n_layers: Transformer encoder layers.
        n_heads: Attention heads.
        feedforward: Feed-forward width as a multiple of ``d_model``.
    """

    n_features: int = 4
    n_classes: int = 2
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    feedforward: int = 2

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is outside its permitted range.
        """

        if self.n_features <= 0 or self.n_classes < 2:
            raise ValueError("n_features must be positive and n_classes at least 2")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("d_model, n_layers, and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.feedforward <= 0:
            raise ValueError("feedforward must be positive")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "feedforward": self.feedforward,
        }


@dataclass(frozen=True, slots=True)
class PFNExperimentConfig:
    """Settings for the PFN track's experiment suite.

    Attributes:
        seeds: Seeds used for every model.
        prior_fitting_steps: Gradient steps taken over sampled tasks. This is the *only*
            training in the track; nothing is fitted per evaluation dataset.
        tasks_per_step: Tasks sampled per gradient step.
        learning_rate: Prior-fitting learning rate.
        n_features: Input width of every task.
        train_context: Context size seen during prior fitting.
        n_query: Query points per task.
        eval_tasks: Tasks sampled for each evaluation point.
        context_sweep: Context sizes used for the small-n study.
        label_noise: Label-flip probability used for the noise study.
        feature_counts: Input widths used for the feature-count study. Each width needs
            its own prior fitting, because the model's input projection is width-specific.
        missing_rates: Fractions of feature entries blanked for the missingness study.
        positive_rate: Positive-class fraction used for the class-imbalance study.
        cost_tasks: Tasks timed for the inference-cost measurement.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    prior_fitting_steps: int = 600
    tasks_per_step: int = 16
    learning_rate: float = 3e-4
    n_features: int = 4
    train_context: int = 20
    n_query: int = 20
    eval_tasks: int = 100
    context_sweep: tuple[int, ...] = (5, 10, 20, 40)
    label_noise: float = 0.1
    feature_counts: tuple[int, ...] = (4, 16)
    missing_rates: tuple[float, ...] = (0.3,)
    positive_rate: float = 0.2
    cost_tasks: int = 50
