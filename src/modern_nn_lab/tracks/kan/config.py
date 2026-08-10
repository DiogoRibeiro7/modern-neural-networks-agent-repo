"""Typed configuration for the KAN track."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class KANConfig:
    """Configuration of a Kolmogorov-Arnold network.

    Attributes:
        layer_widths: Widths from input to output, for example ``(2, 5, 1)``. Must have
            at least two entries.
        grid_size: Number of spline grid intervals ``G``.
        spline_order: Spline degree ``k``.
        grid_range: Initial knot domain.
        base_scale: Gain of the residual ``SiLU`` branch.
        spline_noise_scale: Noise scale used to initialize spline coefficients.
        learnable_spline: When false, spline coefficients stay frozen — the *fixed edge
            function* ablation.
        use_base_branch: When false, the residual branch is removed.
        regularization_weight: Coefficient of the L1 surrogate added to the loss.
    """

    layer_widths: tuple[int, ...]
    grid_size: int = 5
    spline_order: int = 3
    grid_range: tuple[float, float] = (-1.0, 1.0)
    base_scale: float = 1.0
    spline_noise_scale: float = 0.1
    learnable_spline: bool = True
    use_base_branch: bool = True
    regularization_weight: float = 0.0

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is outside its permitted range.
        """

        if len(self.layer_widths) < 2:
            raise ValueError("layer_widths needs at least an input and an output width")
        if any(width <= 0 for width in self.layer_widths):
            raise ValueError("every layer width must be positive")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if self.spline_order < 1:
            raise ValueError("spline_order must be at least 1")
        if self.grid_range[0] >= self.grid_range[1]:
            raise ValueError("grid_range must satisfy low < high")
        if self.regularization_weight < 0:
            raise ValueError("regularization_weight must be non-negative")

    @property
    def parameters_per_edge(self) -> int:
        """Parameters stored on one edge: ``G + k`` spline coefficients plus the base weight."""

        return self.grid_size + self.spline_order + (1 if self.use_base_branch else 0)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "layer_widths": list(self.layer_widths),
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": list(self.grid_range),
            "base_scale": self.base_scale,
            "spline_noise_scale": self.spline_noise_scale,
            "learnable_spline": self.learnable_spline,
            "use_base_branch": self.use_base_branch,
            "regularization_weight": self.regularization_weight,
        }


@dataclass(frozen=True, slots=True)
class KANExperimentConfig:
    """Settings for the KAN track's experiment suite.

    Attributes:
        seeds: Seeds used for every architecture and variant.
        sweep_seeds: Seeds used for the hyperparameter sensitivity study, which is a
            secondary result and does not need the full seed budget.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        learning_rate: Fallback learning rate when no grid search is run.
        learning_rate_grid: Learning rates searched *per architecture and variant* on the
            validation split. Every architecture gets the same grid and the same number
            of trials, which is what makes the comparison fair: a single shared learning
            rate silently favours whichever architecture that rate happens to suit.
        n_samples: Samples drawn for each synthetic function task.
        noise_std: Target noise added to synthetic tasks.
        grid_sweep: Grid sizes explored in the sensitivity study.
        order_sweep: Spline orders explored in the sensitivity study.
        regularization_sweep: Regularization weights explored in the sensitivity study.
        edge_samples: Number of points at which learned edge functions are recorded.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    sweep_seeds: tuple[int, ...] = (0, 1, 2)
    epochs: int = 400
    batch_size: int = 128
    learning_rate: float = 1e-2
    learning_rate_grid: tuple[float, ...] = (5e-3, 2e-2, 8e-2)
    n_samples: int = 1500
    noise_std: float = 0.01
    grid_sweep: tuple[int, ...] = (3, 5, 10, 20)
    order_sweep: tuple[int, ...] = (1, 2, 3)
    regularization_sweep: tuple[float, ...] = (0.0, 1e-3, 1e-2)
    edge_samples: int = 101
    extra: dict[str, Any] = field(default_factory=dict)
