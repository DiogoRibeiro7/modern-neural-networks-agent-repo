"""Typed configuration for the JEPA track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

AntiCollapse = Literal["ema", "variance", "none"]
"""Which anti-collapse mechanism a JEPA uses. ``none`` is the ablation that collapses."""


@dataclass(frozen=True, slots=True)
class JEPAConfig:
    """Architecture of a joint-embedding predictive model.

    Attributes:
        d_patch: Width of one observed patch.
        d_hidden: Hidden width of the encoders.
        d_representation: Width of the representation being probed.
        n_encoder_layers: Hidden layers in each encoder.
        d_predictor: Hidden width of the predictor.
        n_predictor_layers: Predictor hidden layers; ``0`` makes it the identity.
        anti_collapse: Which mechanism prevents the constant solution.
        ema_decay: Target-encoder decay, used when ``anti_collapse`` is ``"ema"``.
        variance_weight: Weight on the variance hinge, used when ``"variance"``.
        variance_floor: Standard deviation the hinge pushes each dimension above.
        temperature: InfoNCE temperature, for the contrastive baseline.
    """

    d_patch: int = 12
    d_hidden: int = 64
    d_representation: int = 16
    n_encoder_layers: int = 2
    d_predictor: int = 64
    n_predictor_layers: int = 2
    anti_collapse: AntiCollapse = "ema"
    ema_decay: float = 0.99
    variance_weight: float = 1.0
    variance_floor: float = 1.0
    temperature: float = 0.1

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is outside its permitted range.
        """

        if min(self.d_patch, self.d_hidden, self.d_representation, self.d_predictor) <= 0:
            raise ValueError("widths must be positive")
        if self.n_encoder_layers <= 0:
            raise ValueError("n_encoder_layers must be positive")
        if self.n_predictor_layers < 0:
            raise ValueError("n_predictor_layers must be non-negative")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if self.variance_weight < 0 or self.variance_floor < 0:
            raise ValueError("variance weight and floor must be non-negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

    @property
    def effective_variance_weight(self) -> float:
        """Weight actually applied to the variance hinge, given the chosen mechanism."""

        return self.variance_weight if self.anti_collapse == "variance" else 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "d_patch": self.d_patch,
            "d_hidden": self.d_hidden,
            "d_representation": self.d_representation,
            "n_encoder_layers": self.n_encoder_layers,
            "d_predictor": self.d_predictor,
            "n_predictor_layers": self.n_predictor_layers,
            "anti_collapse": self.anti_collapse,
            "ema_decay": self.ema_decay,
            "variance_weight": self.effective_variance_weight,
            "variance_floor": self.variance_floor,
        }


@dataclass(frozen=True, slots=True)
class JEPAExperimentConfig:
    """Settings for the JEPA experiment suite.

    Attributes:
        seeds: Seeds used for every model.
        n_samples: Samples generated, split into train and test.
        n_patches: Patches per sample.
        d_patch: Width of one patch.
        d_content: Content factors, shared within a sample.
        d_nuisance: Nuisance factors, independent per patch.
        noise: Observation noise.
        n_targets: Patches masked during training.
        mask_sweep: Values of ``n_targets`` compared in the masking analysis.
        predictor_sweep: Predictor depths compared in the capacity ablation.
        steps: Training steps.
        batch_size: Samples per step.
        learning_rate: Adam learning rate.
        data_seed: Seed for the dataset, held fixed across models.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_samples: int = 4000
    n_patches: int = 8
    d_patch: int = 12
    d_content: int = 4
    d_nuisance: int = 3
    noise: float = 0.05
    n_targets: int = 4
    mask_sweep: tuple[int, ...] = (1, 2, 4, 6, 7)
    predictor_sweep: tuple[int, ...] = (0, 1, 2, 4)
    steps: int = 1500
    batch_size: int = 128
    learning_rate: float = 3e-3
    data_seed: int = 1729
