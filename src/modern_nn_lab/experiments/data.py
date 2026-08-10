"""Dataset splits with leakage-safe preprocessing.

Two rules are enforced here rather than left to each track:

1. Preprocessing statistics are estimated on the training split only. Fitting a scaler
   on the full dataset is a quiet form of leakage that flatters every model equally and
   is therefore easy to miss.
2. Every split carries a content fingerprint, so a record can prove which data produced
   it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from modern_nn_lab.experiments.records import tensor_fingerprint

MIN_SPLIT_SAMPLES = 3
"""Fewest examples that can still yield three non-empty splits."""

RATIO_TOLERANCE = 1e-6
"""Permitted floating-point slack when checking that split ratios sum to one."""


@dataclass(frozen=True, slots=True)
class SupervisedSplit:
    """A train/validation/test split of a supervised dataset.

    Attributes:
        name: Human-readable dataset identifier used in records.
        train_inputs: Shape ``(N_train, ...)``.
        train_targets: Shape ``(N_train, ...)``.
        val_inputs: Shape ``(N_val, ...)``.
        val_targets: Shape ``(N_val, ...)``.
        test_inputs: Shape ``(N_test, ...)``.
        test_targets: Shape ``(N_test, ...)``.
        strategy: Description of how the split was produced, for the record.
        metadata: JSON-serializable extra information, such as the generating function.
    """

    name: str
    train_inputs: Tensor
    train_targets: Tensor
    val_inputs: Tensor
    val_targets: Tensor
    test_inputs: Tensor
    test_targets: Tensor
    strategy: str
    metadata: dict[str, object]

    @property
    def fingerprint(self) -> str:
        """Content fingerprint over every tensor in the split."""

        return tensor_fingerprint(
            self.train_inputs,
            self.train_targets,
            self.val_inputs,
            self.val_targets,
            self.test_inputs,
            self.test_targets,
        )

    @property
    def input_dim(self) -> int:
        """Width of the final input dimension."""

        return int(self.train_inputs.shape[-1])

    def sizes(self) -> tuple[int, int, int]:
        """Return ``(n_train, n_val, n_test)``."""

        return (
            int(self.train_inputs.shape[0]),
            int(self.val_inputs.shape[0]),
            int(self.test_inputs.shape[0]),
        )


def random_split_indices(
    count: int,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 1729,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return disjoint train/validation/test index tensors.

    Args:
        count: Number of examples.
        ratios: Train/validation/test proportions; must be positive and sum to one.
        seed: Seed for the permutation, so a split is reproducible on its own.

    Returns:
        Three index tensors covering ``range(count)`` without overlap.

    Raises:
        ValueError: If ``count`` is too small or the ratios are invalid.
    """

    if count < MIN_SPLIT_SAMPLES:
        raise ValueError(
            f"count must be at least {MIN_SPLIT_SAMPLES} to form three non-empty splits"
        )
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("all split ratios must be positive")
    if abs(sum(ratios) - 1.0) > RATIO_TOLERANCE:
        raise ValueError(f"split ratios must sum to 1, got {sum(ratios)}")

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    n_train = max(1, round(ratios[0] * count))
    n_val = max(1, round(ratios[1] * count))
    n_train = min(n_train, count - 2)
    n_val = min(n_val, count - n_train - 1)
    return order[:n_train], order[n_train : n_train + n_val], order[n_train + n_val :]


def standardize(
    train: Tensor, *others: Tensor, eps: float = 1e-8
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Standardize tensors using training-split statistics only.

    Args:
        train: Training tensor of shape ``(N, D)``; statistics come from this tensor.
        *others: Further tensors transformed with the training statistics.
        eps: Floor applied to the standard deviation to avoid dividing by zero on
            constant features.

    Returns:
        ``(standardized_train, standardized_others)``.
    """

    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(eps)
    return (train - mean) / std, tuple((tensor - mean) / std for tensor in others)


def make_function_split(
    function: Callable[[Tensor], Tensor],
    *,
    name: str,
    input_dim: int,
    n_samples: int = 3000,
    noise_std: float = 0.0,
    low: float = -1.0,
    high: float = 1.0,
    seed: int = 1729,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> SupervisedSplit:
    """Sample a regression dataset from an analytic function on a box domain.

    Noise is added to the *targets only*, and the noiseless target is not retained, so
    no model can inadvertently see the clean signal.

    Args:
        function: Maps inputs of shape ``(N, input_dim)`` to targets of shape ``(N, 1)``
            or ``(N,)``.
        name: Dataset identifier recorded with results.
        input_dim: Input width.
        n_samples: Total number of examples before splitting.
        noise_std: Standard deviation of additive Gaussian target noise.
        low: Lower bound of the uniform sampling box.
        high: Upper bound of the uniform sampling box.
        seed: Seed for sampling and splitting.
        ratios: Train/validation/test proportions.

    Returns:
        A :class:`SupervisedSplit` with float32 tensors and targets shaped ``(N, 1)``.

    Raises:
        ValueError: If ``input_dim`` or ``n_samples`` is not positive, if ``noise_std``
            is negative, or if ``low`` is not below ``high``.
    """

    if input_dim <= 0:
        raise ValueError("input_dim must be positive")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if low >= high:
        raise ValueError("low must be strictly less than high")

    generator = torch.Generator().manual_seed(seed)
    inputs = torch.rand((n_samples, input_dim), generator=generator) * (high - low) + low
    targets = function(inputs)
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)
    if noise_std > 0:
        targets = targets + noise_std * torch.randn(targets.shape, generator=generator)

    train_idx, val_idx, test_idx = random_split_indices(n_samples, ratios=ratios, seed=seed)
    return SupervisedSplit(
        name=name,
        train_inputs=inputs[train_idx].float(),
        train_targets=targets[train_idx].float(),
        val_inputs=inputs[val_idx].float(),
        val_targets=targets[val_idx].float(),
        test_inputs=inputs[test_idx].float(),
        test_targets=targets[test_idx].float(),
        strategy=f"iid random split {ratios} on {n_samples} uniform samples in [{low}, {high}]",
        metadata={
            "input_dim": input_dim,
            "noise_std": noise_std,
            "domain": [low, high],
            "n_samples": n_samples,
        },
    )


def make_tabular_split(
    inputs: Tensor,
    targets: Tensor,
    *,
    name: str,
    seed: int = 1729,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    standardize_inputs: bool = True,
    standardize_targets: bool = False,
) -> SupervisedSplit:
    """Split an in-memory tabular dataset and standardize using training statistics.

    Args:
        inputs: Shape ``(N, D)`` feature matrix.
        targets: Shape ``(N,)`` or ``(N, 1)`` targets.
        name: Dataset identifier recorded with results.
        seed: Seed for the split permutation.
        ratios: Train/validation/test proportions.
        standardize_inputs: Standardize features with training statistics.
        standardize_targets: Standardize targets with training statistics. Use for
            regression; never for classification labels.

    Returns:
        A :class:`SupervisedSplit`.

    Raises:
        ValueError: If ``inputs`` and ``targets`` disagree on the first dimension.
    """

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must share the first dimension")

    if targets.ndim == 1 and standardize_targets:
        targets = targets.unsqueeze(-1)

    train_idx, val_idx, test_idx = random_split_indices(
        int(inputs.shape[0]), ratios=ratios, seed=seed
    )
    train_x, val_x, test_x = inputs[train_idx], inputs[val_idx], inputs[test_idx]
    train_y, val_y, test_y = targets[train_idx], targets[val_idx], targets[test_idx]

    notes: list[str] = []
    if standardize_inputs:
        train_x, (val_x, test_x) = standardize(train_x.float(), val_x.float(), test_x.float())
        notes.append("features standardized on train")
    if standardize_targets:
        train_y, (val_y, test_y) = standardize(train_y.float(), val_y.float(), test_y.float())
        notes.append("targets standardized on train")

    strategy = f"iid random split {ratios}"
    if notes:
        strategy = f"{strategy}; " + "; ".join(notes)

    return SupervisedSplit(
        name=name,
        train_inputs=train_x.float(),
        train_targets=train_y,
        val_inputs=val_x.float(),
        val_targets=val_y,
        test_inputs=test_x.float(),
        test_targets=test_y,
        strategy=strategy,
        metadata={"n_features": int(inputs.shape[-1]), "n_samples": int(inputs.shape[0])},
    )
