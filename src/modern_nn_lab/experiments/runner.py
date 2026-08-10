"""Multi-seed experiment orchestration.

This module is the only place that turns a trained model into an
:class:`~modern_nn_lab.experiments.records.ExperimentRecord`. Centralizing it means a
track cannot accidentally omit a contract field, and the *same* protocol is applied to a
target architecture and its baselines.

Protocol enforced here:

- the validation split drives in-training evaluation;
- the test split is touched exactly once per run, after training;
- every seed produces its own record;
- a diverged run is recorded with ``status="diverged"`` rather than dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from modern_nn_lab.experiments.data import SupervisedSplit
from modern_nn_lab.experiments.evaluation import Aggregate, aggregate_runs
from modern_nn_lab.experiments.profiling import profile_inference, profile_parameters
from modern_nn_lab.experiments.records import (
    ExperimentRecord,
    MetricValue,
    current_git_commit,
    describe_hardware,
    save_record,
)
from modern_nn_lab.experiments.training import (
    LossFn,
    MetricFn,
    TrainingConfig,
    evaluate,
    train_supervised,
)

ModelFactory = Callable[[int], nn.Module]
"""Builds a fresh model given a seed. Must not reuse parameters between calls."""

DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
"""Five seeds, the minimum required by ``docs/experiment_contract.md``."""


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Identity and reporting metadata for a group of runs.

    Attributes:
        track: Track key, matching :mod:`modern_nn_lab.registry`.
        architecture: Model name, for example ``"kan"`` or ``"mlp"``.
        variant: Ablation or configuration label. ``None`` means the default setting.
        metric_name: Name of the primary metric computed on the test split.
        higher_is_better: Orientation of the primary metric.
        architecture_version: Version of the implementation, bumped on behaviour change.
        activated_parameters: Parameters used per example, when it differs from the
            total. Required for conditional computation.
        precision: Numerical precision descriptor.
        notes: Free-text caveats stored with every record in the group.
        extra_config: Additional configuration merged into the record's snapshot.
    """

    track: str
    architecture: str
    metric_name: str
    higher_is_better: bool
    variant: str | None = None
    architecture_version: str = "0.1.0"
    activated_parameters: int | None = None
    precision: str = "float32"
    notes: str | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunGroup:
    """Records produced by one specification across seeds.

    Attributes:
        spec: The specification that produced the records.
        records: One record per seed, in seed order.
        paths: Files written, aligned with ``records``.
    """

    spec: RunSpec
    records: tuple[ExperimentRecord, ...]
    paths: tuple[Path, ...]

    @property
    def successful(self) -> tuple[ExperimentRecord, ...]:
        """Records whose run finished successfully."""

        return tuple(record for record in self.records if record.status == "success")

    def aggregate(self) -> Aggregate:
        """Aggregate the primary metric across successful seeds.

        Returns:
            An :class:`~modern_nn_lab.experiments.evaluation.Aggregate`.

        Raises:
            ValueError: If no run succeeded.
        """

        if not self.successful:
            raise ValueError(
                f"{self.spec.architecture}/{self.spec.variant or 'default'}: every seed failed"
            )
        return aggregate_runs(self.successful)


def run_seeded_experiment(
    model_factory: ModelFactory,
    split: SupervisedSplit,
    *,
    spec: RunSpec,
    training_config: TrainingConfig,
    loss_fn: LossFn,
    metric_fn: MetricFn,
    output_dir: Path | str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    val_metric_fn: MetricFn | None = None,
    profile_inference_cost: bool = True,
    eval_batch_size: int | None = None,
) -> RunGroup:
    """Train one architecture across seeds and write one record per seed.

    Args:
        model_factory: Builds a fresh model for a given seed.
        split: Dataset split; validation drives in-training evaluation, test is scored once.
        spec: Identity and reporting metadata.
        training_config: Optimization settings; the seed field is overridden per run.
        loss_fn: Training loss.
        metric_fn: Primary metric, computed on the test split.
        output_dir: Directory receiving the JSON records.
        seeds: Seeds to run. At least five for a headline result.
        val_metric_fn: Metric tracked on validation during training. Defaults to
            ``metric_fn``.
        profile_inference_cost: Measure inference latency and throughput after training.
        eval_batch_size: Chunk size for evaluation; ``None`` evaluates in one batch.

    Returns:
        A :class:`RunGroup`.

    Raises:
        ValueError: If ``seeds`` is empty.
    """

    if not seeds:
        raise ValueError("seeds must not be empty")

    device = torch.device(training_config.device)
    hardware = describe_hardware(device)
    commit = current_git_commit()
    validation_metric = val_metric_fn or metric_fn

    records: list[ExperimentRecord] = []
    paths: list[Path] = []

    for seed in seeds:
        config = TrainingConfig(**{**_config_kwargs(training_config), "seed": seed})
        model = model_factory(seed)
        outcome = train_supervised(
            model,
            split.train_inputs,
            split.train_targets,
            config=config,
            loss_fn=loss_fn,
            eval_inputs=split.val_inputs,
            eval_targets=split.val_targets,
            metric_fn=validation_metric,
            higher_is_better=spec.higher_is_better,
        )

        parameters = profile_parameters(model, activated=spec.activated_parameters)

        if outcome.status == "diverged":
            test_value = float("nan")
            secondary: dict[str, float] = {}
            latency_ms: float | None = None
            throughput: float | None = None
            peak_bytes: int | None = None
        else:
            test_value = evaluate(
                model, split.test_inputs, split.test_targets, metric_fn, batch_size=eval_batch_size
            )
            secondary = {
                "val_metric_final": outcome.final_metric,
                "val_metric_best": outcome.best_metric,
                "train_loss_final": outcome.train_loss[-1] if outcome.train_loss else float("nan"),
            }
            latency_ms = throughput = None
            peak_bytes = None
            if profile_inference_cost:
                latency = profile_inference(
                    model,
                    split.test_inputs[: min(256, split.test_inputs.shape[0])],
                    device=device,
                )
                latency_ms = latency.latency_ms_mean
                throughput = latency.throughput_per_s
                peak_bytes = latency.peak_memory_bytes

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
            optimizer=config.optimizer,
            scheduler="cosine" if config.cosine_schedule else None,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            epochs=config.epochs,
            steps=outcome.steps,
            effective_samples=outcome.effective_samples,
            parameter_count=parameters.total,
            activated_parameter_count=parameters.activated,
            train_wall_clock_s=outcome.wall_clock_s,
            inference_latency_ms=latency_ms,
            inference_throughput_per_s=throughput,
            peak_memory_bytes=peak_bytes,
            primary_metric=MetricValue(
                name=spec.metric_name,
                value=test_value,
                higher_is_better=spec.higher_is_better,
            ),
            secondary_metrics=secondary,
            train_loss_trajectory=outcome.train_loss,
            eval_metric_trajectory=outcome.eval_metric,
            hardware=hardware,
            precision=spec.precision,
            config={
                **config.as_dict(),
                **split.metadata,
                **spec.extra_config,
                "diverged_epoch": outcome.diverged_epoch,
            },
            status=outcome.status,
            notes=spec.notes,
        )
        records.append(record)
        paths.append(save_record(record, output_dir))

    return RunGroup(spec=spec, records=tuple(records), paths=tuple(paths))


def _config_kwargs(config: TrainingConfig) -> dict[str, Any]:
    """Return the constructor arguments of ``config`` as a mutable dictionary."""

    return {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "optimizer": config.optimizer,
        "grad_clip": config.grad_clip,
        "cosine_schedule": config.cosine_schedule,
        "eval_every": config.eval_every,
        "seed": config.seed,
        "device": config.device,
        "shuffle": config.shuffle,
    }
