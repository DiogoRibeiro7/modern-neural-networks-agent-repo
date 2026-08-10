"""Metric correctness and aggregation invariants."""

from __future__ import annotations

import math

import pytest
import torch

from modern_nn_lab.experiments.evaluation import (
    accuracy,
    aggregate_runs,
    aggregate_values,
    bootstrap_interval,
    expected_calibration_error,
    mean_absolute_error,
    mean_squared_error,
    negative_log_likelihood,
    r2_score,
    root_mean_squared_error,
)
from modern_nn_lab.experiments.records import MetricValue
from tests.conftest import make_record


def test_regression_metrics_on_hand_computable_values() -> None:
    predictions = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([1.0, 4.0, 3.0])
    # Errors are [0, -2, 0]: MSE = 4/3, MAE = 2/3.
    assert mean_squared_error(predictions, targets) == pytest.approx(4.0 / 3.0)
    assert mean_absolute_error(predictions, targets) == pytest.approx(2.0 / 3.0)
    assert root_mean_squared_error(predictions, targets) == pytest.approx(math.sqrt(4.0 / 3.0))


def test_r2_is_one_for_perfect_fit_and_zero_for_constant_targets() -> None:
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert r2_score(values, values) == pytest.approx(1.0)
    constant = torch.full((4,), 2.0)
    assert r2_score(constant, constant) == 0.0


def test_r2_of_mean_predictor_is_zero() -> None:
    targets = torch.tensor([1.0, 2.0, 3.0, 4.0])
    predictions = torch.full_like(targets, float(targets.mean()))
    assert r2_score(predictions, targets) == pytest.approx(0.0, abs=1e-6)


def test_accuracy_multiclass_and_binary() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    targets = torch.tensor([0, 1, 1])
    assert accuracy(logits, targets) == pytest.approx(2.0 / 3.0)
    binary_logits = torch.tensor([1.0, -1.0, 1.0])
    assert accuracy(binary_logits, torch.tensor([1, 0, 1])) == pytest.approx(1.0)


def test_negative_log_likelihood_matches_uniform_baseline() -> None:
    logits = torch.zeros(8, 4)
    targets = torch.zeros(8, dtype=torch.long)
    assert negative_log_likelihood(logits, targets) == pytest.approx(math.log(4.0))


def test_calibration_error_is_zero_for_perfectly_calibrated_certainty() -> None:
    probabilities = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1, 0])
    assert expected_calibration_error(probabilities, targets) == pytest.approx(0.0, abs=1e-6)


def test_calibration_error_detects_overconfidence() -> None:
    probabilities = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    targets = torch.tensor([0, 1])
    # Confidence is 1.0 everywhere but accuracy is 0.5.
    assert expected_calibration_error(probabilities, targets) == pytest.approx(0.5, abs=1e-6)


def test_calibration_error_validates_arguments() -> None:
    probabilities = torch.tensor([[0.5, 0.5]])
    with pytest.raises(ValueError, match="bins"):
        expected_calibration_error(probabilities, torch.tensor([0]), bins=0)
    with pytest.raises(ValueError, match="shape"):
        expected_calibration_error(torch.tensor([0.5]), torch.tensor([0]))


def test_bootstrap_interval_is_deterministic_and_brackets_the_mean() -> None:
    values = [1.0, 1.2, 0.9, 1.1, 1.05]
    first = bootstrap_interval(values, seed=7, resamples=2000)
    second = bootstrap_interval(values, seed=7, resamples=2000)
    assert first == second
    mean = sum(values) / len(values)
    assert first[0] <= mean <= first[1]


def test_bootstrap_interval_of_single_value_is_degenerate() -> None:
    assert bootstrap_interval([2.5]) == (2.5, 2.5)


def test_bootstrap_interval_validates_arguments() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_interval([])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_interval([1.0, 2.0], confidence=1.5)


def test_aggregate_values_reports_every_seed() -> None:
    aggregate = aggregate_values("mse", [1.0, 2.0, 3.0], higher_is_better=False)
    assert aggregate.n_seeds == 3
    assert aggregate.values == (1.0, 2.0, 3.0)
    assert aggregate.mean == pytest.approx(2.0)
    assert aggregate.std == pytest.approx(1.0)
    assert "n=3" in aggregate.format()


def test_aggregate_runs_requires_consistent_metric() -> None:
    good = [make_record(seed=seed) for seed in range(3)]
    assert aggregate_runs(good).n_seeds == 3

    mixed = [
        make_record(seed=0),
        make_record(
            seed=1, primary_metric=MetricValue(name="acc", value=1.0, higher_is_better=True)
        ),
    ]
    with pytest.raises(ValueError, match="primary metric name"):
        aggregate_runs(mixed)


def test_aggregate_runs_refuses_to_hide_diverged_runs() -> None:
    records = [make_record(seed=0), make_record(seed=1, status="diverged")]
    with pytest.raises(ValueError, match="non-successful"):
        aggregate_runs(records)


def test_aggregate_runs_requires_records() -> None:
    with pytest.raises(ValueError, match="no records"):
        aggregate_runs([])
