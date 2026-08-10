"""A small, explicit supervised training loop shared by every track.

The loop is deliberately minimal. It exists so that a target architecture and its
baseline are optimized by *identical* code, which is a precondition for a fair
comparison; it is not a general-purpose training framework.

Design decisions worth knowing:

- Batching is done with an explicit permutation from a seeded :class:`torch.Generator`
  rather than a ``DataLoader``, so a run is reproducible without worker-count caveats.
- A non-finite training loss stops the run and is reported as ``diverged``. Divergence
  is a result, not an error to be retried away.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

from modern_nn_lab.experiments.profiling import Stopwatch
from modern_nn_lab.reproducibility import seed_everything

LossFn = Callable[[Tensor, Tensor], Tensor]
"""Maps ``(model_output, targets)`` to a scalar loss."""

MetricFn = Callable[[Tensor, Tensor], float]
"""Maps ``(model_output, targets)`` to a scalar evaluation metric."""

OptimizerName = Literal["adam", "adamw", "sgd"]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Optimization settings applied identically to every model in a comparison.

    Attributes:
        epochs: Number of passes over the training set.
        batch_size: Mini-batch size. Values larger than the dataset use full batches.
        learning_rate: Base learning rate.
        weight_decay: L2 / decoupled weight decay, depending on ``optimizer``.
        optimizer: Optimizer identifier.
        grad_clip: Global gradient-norm clip. ``None`` disables clipping.
        cosine_schedule: Whether to decay the learning rate with a cosine schedule.
        eval_every: Evaluate every N epochs. ``1`` evaluates after every epoch.
        seed: Seed applied before initialization and batching.
        device: Torch device string.
        shuffle: Whether to shuffle the training set each epoch.
    """

    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerName = "adam"
    grad_clip: float | None = 1.0
    cosine_schedule: bool = False
    eval_every: int = 1
    seed: int = 1729
    device: str = "cpu"
    shuffle: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If any field is outside its permitted range.
        """

        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.eval_every <= 0:
            raise ValueError("eval_every must be positive")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive when set")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "grad_clip": self.grad_clip,
            "cosine_schedule": self.cosine_schedule,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """Everything a training run produced, including its failure mode.

    Attributes:
        train_loss: Mean training loss per epoch.
        eval_metric: Evaluation metric at each evaluation point.
        eval_epochs: Epoch indices matching ``eval_metric``.
        final_metric: Metric after the last evaluation, or ``nan`` if none ran.
        best_metric: Best metric seen, respecting ``higher_is_better``.
        wall_clock_s: Total training wall-clock time.
        steps: Number of optimizer steps executed.
        effective_samples: Training examples processed.
        status: ``"success"`` or ``"diverged"``.
        diverged_epoch: Epoch at which a non-finite loss appeared, if any.
    """

    train_loss: list[float] = field(default_factory=list)
    eval_metric: list[float] = field(default_factory=list)
    eval_epochs: list[int] = field(default_factory=list)
    final_metric: float = math.nan
    best_metric: float = math.nan
    wall_clock_s: float = 0.0
    steps: int = 0
    effective_samples: int = 0
    status: Literal["success", "diverged"] = "success"
    diverged_epoch: int | None = None


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Construct the optimizer named by ``config``.

    Args:
        model: Model whose parameters are optimized.
        config: Training configuration.

    Returns:
        A configured optimizer.

    Raises:
        ValueError: If ``config.optimizer`` is unknown.
    """

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.optimizer == "adam":
        return torch.optim.Adam(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"unknown optimizer {config.optimizer!r}")


def iterate_minibatches(
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    generator: torch.Generator | None,
    shuffle: bool,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield mini-batches, optionally in a seeded random order.

    Args:
        inputs: Shape ``(N, ...)`` inputs.
        targets: Shape ``(N, ...)`` targets aligned with ``inputs``.
        batch_size: Mini-batch size.
        generator: Seeded generator used for the shuffle permutation.
        shuffle: Whether to shuffle.

    Yields:
        ``(input_batch, target_batch)`` pairs.

    Raises:
        ValueError: If ``inputs`` and ``targets`` disagree on the first dimension.
    """

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError(
            f"inputs and targets must share the first dimension, "
            f"got {inputs.shape[0]} and {targets.shape[0]}"
        )

    count = inputs.shape[0]
    order = (
        torch.randperm(count, generator=generator, device=inputs.device)
        if shuffle
        else torch.arange(count, device=inputs.device)
    )
    for start in range(0, count, batch_size):
        index = order[start : start + batch_size]
        yield inputs[index], targets[index]


@torch.no_grad()
def evaluate(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    metric_fn: MetricFn,
    *,
    batch_size: int | None = None,
) -> float:
    """Evaluate ``model`` on a full split.

    Args:
        model: Model to evaluate; switched to eval mode and restored afterwards.
        inputs: Evaluation inputs.
        targets: Evaluation targets.
        metric_fn: Maps model outputs and targets to a scalar.
        batch_size: Evaluate in chunks of this size. ``None`` uses one full batch.

    Returns:
        The metric over the whole split.
    """

    was_training = model.training
    model.eval()
    try:
        if batch_size is None or batch_size >= inputs.shape[0]:
            return metric_fn(model(inputs), targets)
        chunks = [
            model(inputs[start : start + batch_size]) for start in range(0, len(inputs), batch_size)
        ]
        return metric_fn(torch.cat(chunks, dim=0), targets)
    finally:
        model.train(was_training)


@dataclass(frozen=True, slots=True)
class _EpochResult:
    """Outcome of a single training epoch.

    Attributes:
        mean_loss: Mean loss over completed batches; ``nan`` when none completed.
        steps: Optimizer steps taken.
        samples: Training examples processed.
        diverged: Whether a non-finite loss ended the epoch early.
    """

    mean_loss: float
    steps: int
    samples: int
    diverged: bool


def _run_epoch(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    loss_fn: LossFn,
    batch_generator: torch.Generator,
) -> _EpochResult:
    """Run one epoch of mini-batch optimization.

    Args:
        model: Model in training mode.
        inputs: Training inputs already on the target device.
        targets: Training targets already on the target device.
        optimizer: Optimizer to step.
        config: Training configuration.
        loss_fn: Training loss.
        batch_generator: Generator driving the shuffle permutation.

    Returns:
        An :class:`_EpochResult`. A non-finite loss aborts the epoch immediately and is
        reported through ``diverged`` rather than raised, so the caller can record it.
    """

    total_loss = 0.0
    batches = 0
    samples = 0

    for batch_inputs, batch_targets in iterate_minibatches(
        inputs,
        targets,
        batch_size=config.batch_size,
        generator=batch_generator,
        shuffle=config.shuffle,
    ):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch_inputs), batch_targets)
        if not torch.isfinite(loss):
            return _EpochResult(mean_loss=math.nan, steps=batches, samples=samples, diverged=True)

        # Torch ships `Tensor.backward` without annotations, so strict mypy flags the
        # call itself rather than any misuse on our side.
        loss.backward()  # type: ignore[no-untyped-call]
        if config.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_loss += float(loss.detach())
        batches += 1
        samples += int(batch_inputs.shape[0])

    return _EpochResult(
        mean_loss=total_loss / max(batches, 1),
        steps=batches,
        samples=samples,
        diverged=False,
    )


def train_supervised(
    model: nn.Module,
    train_inputs: Tensor,
    train_targets: Tensor,
    *,
    config: TrainingConfig,
    loss_fn: LossFn,
    eval_inputs: Tensor | None = None,
    eval_targets: Tensor | None = None,
    metric_fn: MetricFn | None = None,
    higher_is_better: bool = False,
    seed_model: bool = True,
) -> TrainingOutcome:
    """Train ``model`` with the shared loop and return the full outcome.

    Args:
        model: Module to train, modified in place.
        train_inputs: Training inputs, shape ``(N, ...)``.
        train_targets: Training targets, shape ``(N, ...)``.
        config: Optimization settings shared with the comparison baselines.
        loss_fn: Maps model outputs and targets to a scalar loss.
        eval_inputs: Optional evaluation inputs.
        eval_targets: Optional evaluation targets.
        metric_fn: Metric applied to the evaluation split. Required when evaluation
            data is supplied.
        higher_is_better: Orientation used to select ``best_metric``.
        seed_model: Seed the global RNG before training. Disable when the caller has
            already seeded initialization and wants to control it precisely.

    Returns:
        A :class:`TrainingOutcome`, with ``status="diverged"`` if the loss became
        non-finite.

    Raises:
        ValueError: If evaluation data is given without a metric, or only one of the
            evaluation tensors is provided.
    """

    has_eval_inputs = eval_inputs is not None
    has_eval_targets = eval_targets is not None
    if has_eval_inputs != has_eval_targets:
        raise ValueError("eval_inputs and eval_targets must be provided together")
    if has_eval_inputs and metric_fn is None:
        raise ValueError("metric_fn is required when evaluation data is provided")

    if seed_model:
        seed_everything(config.seed)

    device = torch.device(config.device)
    model = model.to(device)
    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    if eval_inputs is not None and eval_targets is not None:
        eval_inputs = eval_inputs.to(device)
        eval_targets = eval_targets.to(device)

    optimizer = build_optimizer(model, config)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.cosine_schedule
        else None
    )

    # Batch order is drawn from a dedicated generator so it does not depend on how many
    # random numbers the model's initialization happened to consume.
    batch_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)

    outcome_loss: list[float] = []
    outcome_metric: list[float] = []
    outcome_epochs: list[int] = []
    steps = 0
    samples = 0
    status: Literal["success", "diverged"] = "success"
    diverged_epoch: int | None = None

    model.train()
    with Stopwatch() as watch:
        for epoch in range(config.epochs):
            epoch_result = _run_epoch(
                model,
                train_inputs,
                train_targets,
                optimizer=optimizer,
                config=config,
                loss_fn=loss_fn,
                batch_generator=batch_generator,
            )
            steps += epoch_result.steps
            samples += epoch_result.samples

            if epoch_result.diverged:
                status = "diverged"
                diverged_epoch = epoch
                break

            outcome_loss.append(epoch_result.mean_loss)
            if scheduler is not None:
                scheduler.step()

            should_evaluate = (epoch + 1) % config.eval_every == 0 or epoch == config.epochs - 1
            if should_evaluate and eval_inputs is not None and eval_targets is not None:
                assert metric_fn is not None  # guaranteed by the argument validation above
                outcome_metric.append(evaluate(model, eval_inputs, eval_targets, metric_fn))
                outcome_epochs.append(epoch)

    final_metric = outcome_metric[-1] if outcome_metric else math.nan
    if outcome_metric:
        best_metric = max(outcome_metric) if higher_is_better else min(outcome_metric)
    else:
        best_metric = math.nan

    return TrainingOutcome(
        train_loss=outcome_loss,
        eval_metric=outcome_metric,
        eval_epochs=outcome_epochs,
        final_metric=final_metric,
        best_metric=best_metric,
        wall_clock_s=watch.elapsed_s,
        steps=steps,
        effective_samples=samples,
        status=status,
        diverged_epoch=diverged_epoch,
    )
