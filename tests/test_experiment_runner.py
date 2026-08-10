"""Profiling accounting and end-to-end runner behaviour."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from modern_nn_lab.experiments.data import make_function_split
from modern_nn_lab.experiments.evaluation import mean_squared_error
from modern_nn_lab.experiments.profiling import (
    Stopwatch,
    linear_flops,
    profile_inference,
    profile_parameters,
)
from modern_nn_lab.experiments.runner import RunSpec, run_seeded_experiment
from modern_nn_lab.experiments.training import TrainingConfig


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    return torch.mean((predictions - targets) ** 2)


def test_parameter_profile_counts_match_torch() -> None:
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    profile = profile_parameters(model)
    expected = (4 * 8 + 8) + (8 * 1 + 1)
    assert profile.total == expected
    assert profile.trainable == expected
    assert profile.activated == expected
    assert profile.sparsity == 0.0


def test_parameter_profile_reports_conditional_activation() -> None:
    model = nn.Linear(10, 10)
    profile = profile_parameters(model, activated=len(list(model.parameters())) and 55)
    assert profile.activated == 55
    assert profile.sparsity == pytest.approx(1.0 - 55 / profile.total)


def test_parameter_profile_excludes_frozen_from_trainable() -> None:
    model = nn.Linear(4, 4)
    model.bias.requires_grad_(False)
    profile = profile_parameters(model)
    assert profile.trainable == profile.total - 4


def test_profile_inference_reports_positive_throughput() -> None:
    model = nn.Linear(3, 1)
    profile = profile_inference(model, torch.randn(8, 3), repeats=3, warmup=1)
    assert profile.batch_size == 8
    assert profile.repeats == 3
    assert profile.latency_ms_mean >= 0.0
    assert profile.throughput_per_s > 0.0


def test_profile_inference_validates_arguments() -> None:
    with pytest.raises(ValueError, match="repeats"):
        profile_inference(nn.Linear(3, 1), torch.randn(2, 3), repeats=0)
    with pytest.raises(ValueError, match="batch dimension"):
        profile_inference(nn.Linear(3, 1), torch.tensor(1.0))


def test_stopwatch_measures_non_negative_elapsed_time() -> None:
    with Stopwatch() as watch:
        _ = sum(range(10_000))
    assert watch.elapsed_s >= 0.0


def test_linear_flops_counts_multiply_accumulate_as_two_operations() -> None:
    assert linear_flops(4, 8, bias=False) == pytest.approx(64.0)
    assert linear_flops(4, 8, bias=True) == pytest.approx(72.0)


def test_runner_writes_one_record_per_seed(tmp_path) -> None:
    split = make_function_split(
        lambda x: x.sum(dim=-1, keepdim=True), name="sum", input_dim=2, n_samples=200, seed=0
    )
    group = run_seeded_experiment(
        lambda seed: nn.Linear(2, 1),
        split,
        spec=RunSpec(
            track="test", architecture="linear", metric_name="test_mse", higher_is_better=False
        ),
        training_config=TrainingConfig(epochs=3, batch_size=32),
        loss_fn=mse_loss,
        metric_fn=mean_squared_error,
        output_dir=tmp_path,
        seeds=(0, 1),
        profile_inference_cost=False,
    )

    assert len(group.records) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert [record.seed for record in group.records] == [0, 1]
    assert all(record.dataset_fingerprint == split.fingerprint for record in group.records)
    assert all(
        record.git_commit is None or len(record.git_commit) == 40 for record in group.records
    )
    assert group.aggregate().n_seeds == 2


def test_runner_records_capacity_and_provenance(tmp_path) -> None:
    split = make_function_split(lambda x: x, name="identity", input_dim=1, n_samples=90, seed=0)
    group = run_seeded_experiment(
        lambda seed: nn.Linear(1, 1),
        split,
        spec=RunSpec(
            track="test",
            architecture="linear",
            metric_name="test_mse",
            higher_is_better=False,
            variant="ablation",
            notes="unit test",
            extra_config={"custom": 7},
        ),
        training_config=TrainingConfig(epochs=2, batch_size=16),
        loss_fn=mse_loss,
        metric_fn=mean_squared_error,
        output_dir=tmp_path,
        seeds=(0,),
        profile_inference_cost=False,
    )
    record = group.records[0]
    assert record.parameter_count == 2
    assert record.variant == "ablation"
    assert record.notes == "unit test"
    assert record.config["custom"] == 7
    assert record.config["epochs"] == 2
    assert record.split_strategy == split.strategy
    assert record.effective_samples == 2 * split.sizes()[0]


def test_runner_marks_divergence_and_refuses_aggregation(tmp_path) -> None:
    split = make_function_split(lambda x: x, name="identity", input_dim=1, n_samples=60, seed=0)

    def exploding(predictions: Tensor, targets: Tensor) -> Tensor:
        return torch.mean((predictions - targets) ** 2) * float("nan")

    group = run_seeded_experiment(
        lambda seed: nn.Linear(1, 1),
        split,
        spec=RunSpec(
            track="test", architecture="linear", metric_name="test_mse", higher_is_better=False
        ),
        training_config=TrainingConfig(epochs=2),
        loss_fn=exploding,
        metric_fn=mean_squared_error,
        output_dir=tmp_path,
        seeds=(0,),
        profile_inference_cost=False,
    )
    assert group.records[0].status == "diverged"
    assert group.successful == ()
    with pytest.raises(ValueError, match="every seed failed"):
        group.aggregate()


def test_runner_requires_seeds(tmp_path) -> None:
    split = make_function_split(lambda x: x, name="identity", input_dim=1, n_samples=60, seed=0)
    with pytest.raises(ValueError, match="seeds"):
        run_seeded_experiment(
            lambda seed: nn.Linear(1, 1),
            split,
            spec=RunSpec(
                track="test", architecture="linear", metric_name="mse", higher_is_better=False
            ),
            training_config=TrainingConfig(epochs=1),
            loss_fn=mse_loss,
            metric_fn=mean_squared_error,
            output_dir=tmp_path,
            seeds=(),
        )
