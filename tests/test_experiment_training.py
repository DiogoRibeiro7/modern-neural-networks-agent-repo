"""Training-loop invariants: determinism, divergence handling, and batching."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from modern_nn_lab.experiments.evaluation import mean_squared_error
from modern_nn_lab.experiments.training import (
    TrainingConfig,
    build_optimizer,
    evaluate,
    iterate_minibatches,
    train_supervised,
)


def linear_problem(n: int = 128) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn((n, 3), generator=generator)
    targets = inputs @ torch.tensor([[1.0], [-2.0], [0.5]])
    return inputs, targets


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    return torch.mean((predictions - targets) ** 2)


def test_config_validates_arguments() -> None:
    with pytest.raises(ValueError, match="epochs"):
        TrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="eval_every"):
        TrainingConfig(eval_every=0)
    with pytest.raises(ValueError, match="grad_clip"):
        TrainingConfig(grad_clip=-1.0)


def test_minibatches_cover_every_example_exactly_once() -> None:
    inputs = torch.arange(10, dtype=torch.float32).unsqueeze(-1)
    targets = inputs.clone()
    generator = torch.Generator().manual_seed(0)
    seen = torch.cat(
        [
            batch_inputs
            for batch_inputs, _ in iterate_minibatches(
                inputs, targets, batch_size=3, generator=generator, shuffle=True
            )
        ]
    )
    assert torch.equal(torch.sort(seen.squeeze(-1)).values, inputs.squeeze(-1))


def test_minibatches_keep_inputs_and_targets_aligned() -> None:
    inputs = torch.arange(12, dtype=torch.float32).unsqueeze(-1)
    targets = inputs * 10.0
    generator = torch.Generator().manual_seed(1)
    for batch_inputs, batch_targets in iterate_minibatches(
        inputs, targets, batch_size=5, generator=generator, shuffle=True
    ):
        assert torch.allclose(batch_targets, batch_inputs * 10.0)


def test_minibatches_reject_mismatched_lengths() -> None:
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="first dimension"):
        list(
            iterate_minibatches(
                torch.zeros(4, 1),
                torch.zeros(3, 1),
                batch_size=2,
                generator=generator,
                shuffle=False,
            )
        )


def test_unknown_optimizer_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(nn.Linear(2, 1), TrainingConfig(optimizer="rmsprop"))  # type: ignore[arg-type]


def test_training_is_deterministic_under_a_fixed_seed() -> None:
    inputs, targets = linear_problem()
    config = TrainingConfig(epochs=5, batch_size=32, seed=4321)

    def run() -> list[float]:
        torch.manual_seed(0)  # deliberately different global state before each run
        model = nn.Linear(3, 1)
        outcome = train_supervised(model, inputs, targets, config=config, loss_fn=mse_loss)
        return outcome.train_loss

    assert run() == run()


def test_training_reduces_loss_on_a_solvable_problem() -> None:
    inputs, targets = linear_problem()
    model = nn.Linear(3, 1)
    outcome = train_supervised(
        model,
        inputs,
        targets,
        config=TrainingConfig(epochs=40, batch_size=32, learning_rate=0.05),
        loss_fn=mse_loss,
    )
    assert outcome.status == "success"
    assert outcome.train_loss[-1] < outcome.train_loss[0] * 0.1
    assert outcome.steps == 40 * 4
    assert outcome.effective_samples == 40 * 128


def test_divergence_is_reported_not_swallowed() -> None:
    inputs, targets = linear_problem()

    def exploding_loss(predictions: Tensor, targets: Tensor) -> Tensor:
        return torch.mean((predictions - targets) ** 2) * float("inf")

    outcome = train_supervised(
        nn.Linear(3, 1),
        inputs,
        targets,
        config=TrainingConfig(epochs=3),
        loss_fn=exploding_loss,
    )
    assert outcome.status == "diverged"
    assert outcome.diverged_epoch == 0


def test_evaluation_requires_a_metric_and_paired_tensors() -> None:
    inputs, targets = linear_problem()
    with pytest.raises(ValueError, match="together"):
        train_supervised(
            nn.Linear(3, 1),
            inputs,
            targets,
            config=TrainingConfig(epochs=1),
            loss_fn=mse_loss,
            eval_inputs=inputs,
        )
    with pytest.raises(ValueError, match="metric_fn"):
        train_supervised(
            nn.Linear(3, 1),
            inputs,
            targets,
            config=TrainingConfig(epochs=1),
            loss_fn=mse_loss,
            eval_inputs=inputs,
            eval_targets=targets,
        )


def test_evaluate_restores_training_mode_and_batches_consistently() -> None:
    inputs, targets = linear_problem()
    model = nn.Linear(3, 1)
    model.train()
    whole = evaluate(model, inputs, targets, mean_squared_error)
    chunked = evaluate(model, inputs, targets, mean_squared_error, batch_size=16)
    assert whole == pytest.approx(chunked, rel=1e-6)
    assert model.training


def test_eval_trajectory_is_recorded_at_requested_epochs() -> None:
    inputs, targets = linear_problem()
    outcome = train_supervised(
        nn.Linear(3, 1),
        inputs,
        targets,
        config=TrainingConfig(epochs=6, eval_every=2),
        loss_fn=mse_loss,
        eval_inputs=inputs,
        eval_targets=targets,
        metric_fn=mean_squared_error,
    )
    assert outcome.eval_epochs == [1, 3, 5]
    assert len(outcome.eval_metric) == 3
    assert outcome.final_metric == outcome.eval_metric[-1]
