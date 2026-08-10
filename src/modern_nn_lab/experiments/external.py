"""Records for non-Torch baselines.

Several tracks require a strong classical baseline — a gradient-boosted tree ensemble
for tabular data, for example. Those estimators do not go through the shared training
loop, but they must still produce records under the same contract, on the same split,
with the same metric. Otherwise the comparison is not auditable.

This module is the single place where that happens.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor

from modern_nn_lab.experiments.data import SupervisedSplit
from modern_nn_lab.experiments.profiling import Stopwatch
from modern_nn_lab.experiments.records import (
    ExperimentRecord,
    MetricValue,
    current_git_commit,
    describe_hardware,
    save_record,
)
from modern_nn_lab.experiments.runner import RunGroup, RunSpec


@runtime_checkable
class Estimator(Protocol):
    """Minimal scikit-learn-style estimator interface."""

    def fit(self, x: Any, y: Any) -> Any:
        """Fit the estimator on training data."""
        ...

    def predict(self, x: Any) -> Any:
        """Predict targets for new data."""
        ...


EstimatorFactory = Callable[[int], Estimator]
"""Builds a fresh estimator given a seed."""


def _to_numpy(tensor: Tensor) -> np.ndarray[Any, Any]:
    """Convert a tensor to a NumPy array, squeezing a trailing singleton dimension."""

    array = tensor.detach().cpu().numpy()
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    return array


def count_estimator_parameters(estimator: Estimator) -> int:
    """Best-effort capacity measure for a fitted tree ensemble.

    Tree ensembles have no parameter count comparable to a neural network's. Reporting
    the total node count is the closest honest analogue, and reports must say so rather
    than placing it in the same column as a weight count without comment.

    Args:
        estimator: A fitted estimator.

    Returns:
        Total node count for forests and boosted ensembles, otherwise ``0``.
    """

    estimators = getattr(estimator, "estimators_", None)
    if estimators is None:
        return 0

    total = 0
    flat = np.asarray(estimators, dtype=object).ravel()
    for tree in flat:
        inner = getattr(tree, "tree_", None)
        if inner is not None:
            total += int(inner.node_count)
    return total


def run_external_baseline(
    factory: EstimatorFactory,
    split: SupervisedSplit,
    *,
    spec: RunSpec,
    metric_fn: Callable[[Tensor, Tensor], float],
    output_dir: Path | str,
    seeds: Sequence[int],
) -> RunGroup:
    """Fit a non-Torch estimator across seeds and write contract-compliant records.

    The estimator sees the same training split as every neural model and is scored with
    the same metric on the same test split.

    Args:
        factory: Builds a fresh estimator for a given seed.
        split: The shared dataset split.
        spec: Identity and reporting metadata.
        metric_fn: Primary metric, applied to test predictions.
        output_dir: Directory receiving the JSON records.
        seeds: Seeds to run.

    Returns:
        A :class:`~modern_nn_lab.experiments.runner.RunGroup`.

    Raises:
        ValueError: If ``seeds`` is empty.
    """

    if not seeds:
        raise ValueError("seeds must not be empty")

    hardware = describe_hardware("cpu")
    commit = current_git_commit()
    train_x = _to_numpy(split.train_inputs)
    train_y = _to_numpy(split.train_targets)
    test_x = _to_numpy(split.test_inputs)

    records: list[ExperimentRecord] = []
    paths: list[Path] = []

    for seed in seeds:
        estimator = factory(seed)
        with Stopwatch() as watch:
            estimator.fit(train_x, train_y)

        predictions = torch.as_tensor(np.asarray(estimator.predict(test_x)), dtype=torch.float32)
        targets = split.test_targets.float()
        if targets.ndim == 2 and targets.shape[1] == 1:
            predictions = predictions.reshape(-1, 1)

        record = ExperimentRecord(
            track=spec.track,
            architecture=spec.architecture,
            architecture_version=spec.architecture_version,
            variant=spec.variant,
            git_commit=commit,
            dataset=split.name,
            dataset_fingerprint=split.fingerprint,
            split_strategy=split.strategy,
            train_samples=int(split.train_inputs.shape[0]),
            eval_samples=int(split.test_inputs.shape[0]),
            seed=seed,
            optimizer="closed-form / greedy fitting",
            parameter_count=count_estimator_parameters(estimator),
            train_wall_clock_s=watch.elapsed_s,
            primary_metric=MetricValue(
                name=spec.metric_name,
                value=metric_fn(predictions, targets),
                higher_is_better=spec.higher_is_better,
            ),
            hardware=hardware,
            precision="float64",
            config={**split.metadata, **spec.extra_config},
            status="success",
            notes=spec.notes,
        )
        records.append(record)
        paths.append(save_record(record, output_dir))

    return RunGroup(spec=spec, records=tuple(records), paths=tuple(paths))
