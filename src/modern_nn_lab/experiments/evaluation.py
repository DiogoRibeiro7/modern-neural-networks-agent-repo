"""Metrics and across-seed aggregation.

Single-seed numbers are not results in this repository. Every headline number is an
aggregate over at least the number of seeds required by ``docs/experiment_contract.md``,
reported with the individual seeds still visible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from modern_nn_lab.experiments.records import ExperimentRecord

_PROBABILITY_NDIM = 2
"""Class-probability tensors are always ``(N, C)``."""


def mean_squared_error(predictions: Tensor, targets: Tensor) -> float:
    """Return the mean squared error between ``predictions`` and ``targets``.

    Args:
        predictions: Tensor of any shape broadcastable to ``targets``.
        targets: Ground-truth tensor.

    Returns:
        Scalar MSE.
    """

    return float(torch.mean((predictions - targets) ** 2))


def root_mean_squared_error(predictions: Tensor, targets: Tensor) -> float:
    """Return the root mean squared error."""

    return float(mean_squared_error(predictions, targets) ** 0.5)


def mean_absolute_error(predictions: Tensor, targets: Tensor) -> float:
    """Return the mean absolute error."""

    return float(torch.mean(torch.abs(predictions - targets)))


def r2_score(predictions: Tensor, targets: Tensor) -> float:
    """Return the coefficient of determination.

    Args:
        predictions: Predicted values.
        targets: Ground-truth values.

    Returns:
        ``1 - SS_res / SS_tot``. Returns ``0.0`` when the targets are constant, since
        the score is undefined there and a division would produce an infinity that
        silently poisons downstream aggregation.
    """

    residual = torch.sum((targets - predictions) ** 2)
    total = torch.sum((targets - targets.mean()) ** 2)
    if float(total) == 0.0:
        return 0.0
    return float(1.0 - residual / total)


def accuracy(logits: Tensor, targets: Tensor) -> float:
    """Return top-1 accuracy.

    Args:
        logits: Shape ``(N, C)`` class scores, or ``(N,)`` binary scores.
        targets: Shape ``(N,)`` integer class labels.

    Returns:
        Fraction of correct predictions in ``[0, 1]``.
    """

    predicted = (logits > 0).long() if logits.ndim == 1 else logits.argmax(dim=-1)
    return float((predicted == targets).float().mean())


def negative_log_likelihood(logits: Tensor, targets: Tensor) -> float:
    """Return the mean cross-entropy of ``logits`` against integer ``targets``.

    Args:
        logits: Shape ``(N, C)`` unnormalized class scores.
        targets: Shape ``(N,)`` integer class labels.

    Returns:
        Mean negative log-likelihood in nats.
    """

    return float(torch.nn.functional.cross_entropy(logits, targets))


def expected_calibration_error(probabilities: Tensor, targets: Tensor, *, bins: int = 15) -> float:
    """Return the expected calibration error of predicted probabilities.

    The confidence interval ``[0, 1]`` is split into ``bins`` equal-width buckets. For
    each bucket the absolute gap between mean confidence and empirical accuracy is
    weighted by the bucket's share of the data.

    Args:
        probabilities: Shape ``(N, C)`` class probabilities that sum to one per row.
        targets: Shape ``(N,)`` integer class labels.
        bins: Number of equal-width confidence buckets. Must be positive.

    Returns:
        Expected calibration error in ``[0, 1]``.

    Raises:
        ValueError: If ``bins`` is not positive or shapes are inconsistent.
    """

    if bins <= 0:
        raise ValueError("bins must be positive")
    if probabilities.ndim != _PROBABILITY_NDIM or probabilities.shape[0] != targets.shape[0]:
        raise ValueError("probabilities must have shape (N, C) matching targets of shape (N,)")

    confidence, predicted = probabilities.max(dim=-1)
    correct = (predicted == targets).float()
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)

    error = torch.zeros((), device=probabilities.device)
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        # Upper-inclusive on the last bucket so confidence exactly 1.0 is counted.
        in_bin = (confidence > low) & ((confidence <= high) if index < bins - 1 else True)
        weight = in_bin.float().mean()
        if float(weight) == 0.0:
            continue
        error = error + weight * torch.abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(error)


@dataclass(frozen=True, slots=True)
class Aggregate:
    """Across-seed summary of one metric.

    Attributes:
        name: Metric name.
        values: Individual per-seed values, kept so no seed is ever hidden.
        mean: Arithmetic mean.
        std: Sample standard deviation (``ddof=1``); ``0.0`` for a single seed.
        ci_low: Lower bound of the percentile bootstrap interval.
        ci_high: Upper bound of the percentile bootstrap interval.
        confidence: Nominal coverage of the interval.
        higher_is_better: Metric orientation, propagated from the records.
    """

    name: str
    values: tuple[float, ...]
    mean: float
    std: float
    ci_low: float
    ci_high: float
    confidence: float
    higher_is_better: bool

    @property
    def n_seeds(self) -> int:
        """Number of seeds contributing to the aggregate."""

        return len(self.values)

    def format(self, decimals: int = 4) -> str:
        """Render as ``mean ± std (n=k)``.

        Args:
            decimals: Digits after the decimal point.

        Returns:
            Human-readable summary string.
        """

        return f"{self.mean:.{decimals}f} ± {self.std:.{decimals}f} (n={self.n_seeds})"


def bootstrap_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for the mean of ``values``.

    With very few seeds the interval is wide and should be read as such. It is reported
    anyway, because omitting uncertainty is worse than reporting honest imprecision.

    Args:
        values: Observed per-seed values. Must be non-empty.
        confidence: Nominal coverage in ``(0, 1)``.
        resamples: Number of bootstrap resamples.
        seed: Seed for the bootstrap resampling generator, so intervals are reproducible.

    Returns:
        ``(low, high)`` bounds of the interval.

    Raises:
        ValueError: If ``values`` is empty or ``confidence`` is out of range.
    """

    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    observed = torch.tensor(values, dtype=torch.float64)
    if observed.numel() == 1:
        single = float(observed[0])
        return single, single

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        low=0,
        high=observed.numel(),
        size=(resamples, observed.numel()),
        generator=generator,
    )
    means = observed[indices].mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    low = float(torch.quantile(means, tail))
    high = float(torch.quantile(means, 1.0 - tail))
    return low, high


def aggregate_values(
    name: str,
    values: Sequence[float],
    *,
    higher_is_better: bool,
    confidence: float = 0.95,
    seed: int = 0,
) -> Aggregate:
    """Summarize per-seed values with mean, standard deviation, and a bootstrap interval.

    Args:
        name: Metric name.
        values: Per-seed values. Must be non-empty.
        higher_is_better: Metric orientation.
        confidence: Nominal coverage of the bootstrap interval.
        seed: Bootstrap generator seed.

    Returns:
        An :class:`Aggregate`.

    Raises:
        ValueError: If ``values`` is empty.
    """

    if not values:
        raise ValueError("values must not be empty")

    observed = torch.tensor(values, dtype=torch.float64)
    # The unbiased estimator is undefined for a single observation.
    std = float(observed.std(unbiased=True)) if observed.numel() > 1 else 0.0
    low, high = bootstrap_interval(values, confidence=confidence, seed=seed)
    return Aggregate(
        name=name,
        values=tuple(float(value) for value in values),
        mean=float(observed.mean()),
        std=std,
        ci_low=low,
        ci_high=high,
        confidence=confidence,
        higher_is_better=higher_is_better,
    )


def aggregate_runs(
    records: Iterable[ExperimentRecord],
    *,
    confidence: float = 0.95,
    seed: int = 0,
) -> Aggregate:
    """Aggregate the primary metric across records that differ only by seed.

    Args:
        records: Records for the same architecture, variant, dataset, and configuration.
        confidence: Nominal coverage of the bootstrap interval.
        seed: Bootstrap generator seed.

    Returns:
        An :class:`Aggregate` over the primary metric.

    Raises:
        ValueError: If no records are given, if they disagree on the primary metric
            name or orientation, or if any run did not finish successfully. Silently
            averaging over a diverged run would hide the divergence.
    """

    collected = list(records)
    if not collected:
        raise ValueError("no records to aggregate")

    names = {record.primary_metric.name for record in collected}
    if len(names) != 1:
        raise ValueError(f"records disagree on the primary metric name: {sorted(names)}")

    orientations = {record.primary_metric.higher_is_better for record in collected}
    if len(orientations) != 1:
        raise ValueError("records disagree on whether the primary metric is higher-is-better")

    bad = [record for record in collected if record.status != "success"]
    if bad:
        statuses = sorted({record.status for record in bad})
        raise ValueError(
            f"refusing to aggregate over non-successful runs (statuses: {statuses}). "
            "Report them explicitly instead."
        )

    return aggregate_values(
        names.pop(),
        [record.primary_metric.value for record in collected],
        higher_is_better=orientations.pop(),
        confidence=confidence,
        seed=seed,
    )
