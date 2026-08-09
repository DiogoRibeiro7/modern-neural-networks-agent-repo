"""Shared typed contracts for experiments and model outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias, runtime_checkable

from torch import Tensor, nn

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class Batch:
    """A supervised batch used by shared experiment utilities.

    Attributes:
        inputs: Input tensor. Shape is task-specific and documented by each task.
        targets: Target tensor. Shape is task-specific and documented by each task.
        mask: Optional boolean/float mask aligned to the task contract.
        metadata: JSON-serializable metadata only.
    """

    inputs: Tensor
    targets: Tensor
    mask: Tensor | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Standard model output.

    Attributes:
        predictions: Main prediction tensor.
        auxiliary: Optional tensors such as router logits or memory diagnostics.
    """

    predictions: Tensor
    auxiliary: dict[str, Tensor] = field(default_factory=dict)


@runtime_checkable
class NeuralModel(Protocol):
    """Structural contract implemented by trainable models used in experiments."""

    def __call__(self, inputs: Tensor, **kwargs: Any) -> ModelOutput | Tensor:
        """Run a forward pass."""
        ...

    def parameters(self, recurse: bool = True) -> Any:
        """Return model parameters, following ``torch.nn.Module`` semantics."""
        ...


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters in ``model``.

    Args:
        model: PyTorch module to inspect.

    Returns:
        Number of scalar parameters whose ``requires_grad`` flag is true.
    """

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
