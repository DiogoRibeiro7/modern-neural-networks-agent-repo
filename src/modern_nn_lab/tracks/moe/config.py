"""Typed configuration for the sparse mixture-of-experts track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

LayerKind = Literal["dense-ffn", "dense-moe", "sparse-moe"]
"""Which mixture layer a model uses."""


@dataclass(frozen=True, slots=True)
class MoEConfig:
    """Architecture of one model in the comparison.

    Attributes:
        layer: Which mixture layer to build.
        d_in: Input width of a token.
        d_out: Output width of a token.
        d_model: Model width.
        d_hidden: Hidden width of a single expert.
        n_experts: Number of experts.
        top_k: Experts each token visits, for the sparse layer.
        capacity_factor: Capacity multiplier, for the sparse layer.
        renormalize: Rescale kept gates to sum to one.
        aux_loss_weight: Weight on the load-balancing loss.
        dense_hidden: Hidden width of the dense feed-forward baseline. Set wider than a
            single expert so the baseline is not simply a smaller model; see below.
    """

    layer: LayerKind = "sparse-moe"
    d_in: int = 8
    d_out: int = 1
    d_model: int = 32
    d_hidden: int = 32
    n_experts: int = 4
    top_k: int = 1
    capacity_factor: float = 1.25
    renormalize: bool = True
    aux_loss_weight: float = 0.01
    dense_hidden: int = 128

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is outside its permitted range.
        """

        if min(self.d_in, self.d_out, self.d_model, self.d_hidden, self.dense_hidden) <= 0:
            raise ValueError("widths must be positive")
        if self.n_experts < 2:
            raise ValueError("n_experts must be at least 2")
        if not 1 <= self.top_k <= self.n_experts:
            raise ValueError(f"top_k must lie in [1, {self.n_experts}]")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")
        if self.aux_loss_weight < 0:
            raise ValueError("aux_loss_weight must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "layer": self.layer,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "capacity_factor": self.capacity_factor,
            "renormalize": self.renormalize,
            "aux_loss_weight": self.aux_loss_weight,
            "dense_hidden": self.dense_hidden,
        }


@dataclass(frozen=True, slots=True)
class MoEExperimentConfig:
    """Settings for the mixture-of-experts experiment suite.

    Attributes:
        seeds: Seeds used for every model.
        n_sequences: Sequences generated, split into train/validation/test.
        seq_len: Tokens per sequence.
        n_functions: Generating functions in the mixture.
        d_selector: Width of the half that decides which function applies.
        d_value: Width of the half the function acts on.
        n_experts: Experts in every mixture layer.
        top_k_grid: Values of ``top_k`` compared.
        capacity_grid: Capacity factors compared, including one that forces overflow.
        aux_weight_grid: Load-balancing weights compared, including zero.
        epochs: Training epochs.
        batch_size: Sequences per step.
        learning_rate_grid: Rates searched per architecture on validation.
        data_seed: Seed for the dataset; fixed so every model sees identical tokens.
    """

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_sequences: int = 1200
    seq_len: int = 16
    n_functions: int = 4
    d_selector: int = 4
    d_value: int = 4
    n_experts: int = 4
    top_k_grid: tuple[int, ...] = (1, 2)
    capacity_grid: tuple[float, ...] = (1.25, 0.5)
    aux_weight_grid: tuple[float, ...] = (0.01, 0.0)
    epochs: int = 60
    batch_size: int = 32
    learning_rate_grid: tuple[float, ...] = (3e-3, 1e-2)
    data_seed: int = 1729
