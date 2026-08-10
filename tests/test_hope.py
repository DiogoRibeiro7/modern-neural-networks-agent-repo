"""Level-by-level invariants of the Nested Learning track.

The track's acceptance criterion is that **every level has an explicit state transition and
a test**. These tests pin each level down independently: what its state is, what it
computes, and how often it updates. The composition tests then check that stacking levels
reproduces algorithms that already have names — which is the falsifiable content of the
source's claim.
"""

from __future__ import annotations

import pytest
import torch

from modern_nn_lab.experiments.tracks.hope import (
    StreamingEstimator,
    make_continual_stream,
    stream_to_split,
)
from modern_nn_lab.tracks.hope import (
    DataMemory,
    DataMemoryConfig,
    GradientMemory,
    LearnerConfig,
    NestedLearner,
    NestedTrace,
    SelfReferentialLearner,
    local_surprise_signal,
    weight_gradient,
)

# --------------------------------------------------------------------------------------
# L0: the data memory
# --------------------------------------------------------------------------------------


def test_local_surprise_signal_is_the_prediction_error() -> None:
    """Equation 8 factors the weight gradient into this signal times the input."""

    weights = torch.randn(3, 4)
    inputs = torch.randn(4)
    targets = torch.randn(3)
    assert torch.allclose(
        local_surprise_signal(weights, inputs, targets), weights @ inputs - targets, atol=1e-6
    )


def test_weight_gradient_is_the_outer_product_of_equation_8() -> None:
    weights, inputs, targets = torch.randn(3, 4), torch.randn(4), torch.randn(3)
    expected = torch.outer(weights @ inputs - targets, inputs)
    assert torch.allclose(weight_gradient(weights, inputs, targets), expected, atol=1e-6)

    # And it really is the gradient of 1/2 ||Wx - t||^2 with respect to W.
    tracked = weights.clone().requires_grad_(True)
    loss = 0.5 * ((tracked @ inputs - targets) ** 2).sum()
    (autograd,) = torch.autograd.grad(loss, tracked)
    assert torch.allclose(weight_gradient(weights, inputs, targets), autograd, atol=1e-6)


def test_data_memory_subtracts_what_it_is_given() -> None:
    level = DataMemory()
    state = level.initial_state((2, 3))
    signal = torch.randn(2, 3)
    state, weights = level.update(state, signal)
    assert torch.allclose(weights, -signal, atol=1e-7)
    assert state.steps == 1


def test_hebbian_rule_matches_equation_64() -> None:
    level = DataMemory(DataMemoryConfig(rule="hebbian", learning_rate=0.3, decay=0.8))
    state = level.initial_state((3, 4))
    state.tensors["weights"] = torch.randn(3, 4)
    before = state.tensors["weights"].clone()

    key, value = torch.randn(4), torch.randn(3)
    state = level.associative_update(state, key, value)
    expected = 0.8 * before + 0.3 * torch.outer(value, key)
    assert torch.allclose(state.tensors["weights"], expected, atol=1e-6)


def test_delta_rule_matches_equation_65() -> None:
    """M <- (I - eta k k^T) M + eta v k^T: the old value for this key is removed first."""

    level = DataMemory(DataMemoryConfig(rule="delta", learning_rate=0.25))
    state = level.initial_state((3, 4))
    state.tensors["weights"] = torch.randn(3, 4)
    before = state.tensors["weights"].clone()

    key, value = torch.randn(4), torch.randn(3)
    state = level.associative_update(state, key, value)

    identity = torch.eye(4)
    expected = before @ (identity - 0.25 * torch.outer(key, key)) + 0.25 * torch.outer(value, key)
    assert torch.allclose(state.tensors["weights"], expected, atol=1e-5)


def test_delta_rule_differs_from_hebbian() -> None:
    torch.manual_seed(0)
    key, value = torch.randn(4), torch.randn(3)
    outcomes = []
    for rule in ("hebbian", "delta"):
        level = DataMemory(DataMemoryConfig(rule=rule, learning_rate=0.25))
        state = level.initial_state((3, 4))
        state.tensors["weights"] = torch.ones(3, 4)
        outcomes.append(level.associative_update(state, key, value).tensors["weights"])
    assert not torch.allclose(outcomes[0], outcomes[1], atol=1e-4)


def test_associative_update_rejects_the_plain_gradient_rule() -> None:
    level = DataMemory()
    with pytest.raises(ValueError, match="hebbian or delta"):
        level.associative_update(level.initial_state((2, 2)), torch.randn(2), torch.randn(2))


# --------------------------------------------------------------------------------------
# L1: the gradient memory, and its update frequency
# --------------------------------------------------------------------------------------


def test_gradient_memory_matches_equation_11() -> None:
    level = GradientMemory(learning_rate=0.2, decay=0.9)
    state = level.initial_state((2, 3))
    first, second = torch.randn(2, 3), torch.randn(2, 3)

    state, out_first = level.update(state, first)
    assert torch.allclose(out_first, 0.2 * first, atol=1e-6)

    state, out_second = level.update(state, second)
    assert torch.allclose(out_second, 0.9 * (0.2 * first) + 0.2 * second, atol=1e-6)


def test_gradient_memory_period_controls_its_update_frequency() -> None:
    """A slower level must actually hold its value between updates."""

    level = GradientMemory(learning_rate=1.0, decay=1.0, period=3)
    state = level.initial_state((1, 2))
    signal = torch.ones(1, 2)

    outputs = []
    for _ in range(9):
        state, output = level.update(state, signal)
        outputs.append(output.clone())

    # Updates land on samples 0, 3, 6 only.
    assert state.steps == 3
    assert torch.allclose(outputs[0], outputs[1])
    assert torch.allclose(outputs[1], outputs[2])
    assert not torch.allclose(outputs[2], outputs[3])


def test_level_period_must_be_positive() -> None:
    with pytest.raises(ValueError, match="period"):
        GradientMemory(period=0)
    with pytest.raises(ValueError, match="learning_rate"):
        GradientMemory(learning_rate=0.0)
    with pytest.raises(ValueError, match="decay"):
        GradientMemory(decay=1.5)


# --------------------------------------------------------------------------------------
# Composition: adding a level must equal using momentum
# --------------------------------------------------------------------------------------


def test_one_level_learner_is_exactly_gradient_descent() -> None:
    torch.manual_seed(0)
    config = LearnerConfig(d_in=4, d_out=2, learning_rate=0.1, use_gradient_memory=False)
    learner = NestedLearner(config)

    weights = learner.weights.clone()
    for _ in range(5):
        inputs, targets = torch.randn(4), torch.randn(2)
        learner.step(inputs, targets)
        weights = weights - 0.1 * weight_gradient(weights, inputs, targets)
        assert torch.allclose(learner.weights, weights, atol=1e-6)


def test_two_level_learner_matches_torch_sgd_with_momentum() -> None:
    """The two-level composition must reproduce an optimizer that already has a name."""

    torch.manual_seed(0)
    config = LearnerConfig(
        d_in=4, d_out=2, learning_rate=0.05, use_gradient_memory=True, momentum_decay=0.9
    )
    learner = NestedLearner(config)

    reference = learner.weights.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([reference], lr=0.05, momentum=0.9)

    for _ in range(6):
        inputs, targets = torch.randn(4), torch.randn(2)
        learner.step(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * ((reference @ inputs - targets) ** 2).sum()
        loss.backward()
        optimizer.step()

        assert torch.allclose(learner.weights, reference.detach(), atol=1e-5)


def test_removing_the_gradient_memory_recovers_the_one_level_learner() -> None:
    """'Add a level' and 'use momentum' must be the same edit, so removal is exact."""

    torch.manual_seed(0)
    samples = [(torch.randn(4), torch.randn(2)) for _ in range(6)]

    without = NestedLearner(
        LearnerConfig(d_in=4, d_out=2, use_gradient_memory=False, init_scale=0.0)
    )
    disabled = NestedLearner(
        LearnerConfig(d_in=4, d_out=2, use_gradient_memory=False, init_scale=0.0)
    )
    for inputs, targets in samples:
        without.step(inputs, targets)
        disabled.step(inputs, targets)
    assert torch.equal(without.weights, disabled.weights)

    with_memory = NestedLearner(
        LearnerConfig(d_in=4, d_out=2, use_gradient_memory=True, init_scale=0.0)
    )
    for inputs, targets in samples:
        with_memory.step(inputs, targets)
    assert not torch.allclose(with_memory.weights, without.weights, atol=1e-4)


def test_resetting_the_gradient_memory_clears_only_that_level() -> None:
    torch.manual_seed(0)
    learner = NestedLearner(LearnerConfig(d_in=4, d_out=2, use_gradient_memory=True))
    for _ in range(5):
        learner.step(torch.randn(4), torch.randn(2))

    weights_before = learner.weights.clone()
    assert learner.gradient_state is not None
    assert float(learner.gradient_state.tensors["momentum"].norm()) > 0

    learner.reset_gradient_memory()
    assert torch.equal(learner.weights, weights_before)
    assert float(learner.gradient_state.tensors["momentum"].norm()) == 0.0
    assert learner.gradient_state.steps == 0


def test_trace_records_each_level_update_count() -> None:
    learner = NestedLearner(
        LearnerConfig(d_in=3, d_out=2, use_gradient_memory=True, momentum_period=2)
    )
    trace = NestedTrace()
    for _ in range(6):
        learner.step(torch.randn(3), torch.randn(2), trace)

    assert trace.level_steps["data_memory"] == [1, 2, 3, 4, 5, 6]
    # The slower level updates half as often, and the trace proves it.
    assert trace.level_steps["gradient_memory"] == [1, 1, 2, 2, 3, 3]


def test_learner_validates_shapes_and_configuration() -> None:
    learner = NestedLearner(LearnerConfig(d_in=4, d_out=2))
    with pytest.raises(ValueError, match="expected inputs"):
        learner.step(torch.randn(3), torch.randn(2))
    with pytest.raises(ValueError, match="positive"):
        LearnerConfig(d_in=0)
    with pytest.raises(ValueError, match="momentum_period"):
        LearnerConfig(momentum_period=0)


# --------------------------------------------------------------------------------------
# The self-referential level
# --------------------------------------------------------------------------------------


def test_self_generated_value_reduces_to_equation_58() -> None:
    """u_t = f_{W_t}(x_t) = -grad_y L for the L2 instance, so the step equals plain GD."""

    torch.manual_seed(0)
    inputs, targets = torch.randn(4), torch.randn(2)

    referential = SelfReferentialLearner(
        LearnerConfig(d_in=4, d_out=2, use_gradient_memory=False, init_scale=0.0)
    )
    value = referential.self_generated_value(inputs, targets)
    assert torch.allclose(value, -local_surprise_signal(referential.weights, inputs, targets))

    plain = NestedLearner(LearnerConfig(d_in=4, d_out=2, use_gradient_memory=False, init_scale=0.0))
    referential.step(inputs, targets)
    plain.step(inputs, targets)
    assert torch.allclose(referential.weights, plain.weights, atol=1e-7)


def test_a_different_value_function_changes_the_learner() -> None:
    """Definition 5 generalizes eq. 58 by letting the value generator be anything."""

    torch.manual_seed(0)
    inputs, targets = torch.randn(4), torch.randn(2)

    custom = SelfReferentialLearner(
        LearnerConfig(d_in=4, d_out=2, use_gradient_memory=False, init_scale=0.0),
        value_fn=lambda weights, x: torch.tanh(weights @ x),
    )
    plain = NestedLearner(LearnerConfig(d_in=4, d_out=2, use_gradient_memory=False, init_scale=0.0))
    custom.step(inputs, targets)
    plain.step(inputs, targets)
    assert not torch.allclose(custom.weights, plain.weights, atol=1e-5)


# --------------------------------------------------------------------------------------
# The continual stream
# --------------------------------------------------------------------------------------


def test_stream_preserves_task_order_and_labels() -> None:
    stream = make_continual_stream(n_tasks=3, samples_per_task=10, test_samples=4, seed=0)
    assert stream.inputs.shape[0] == 30
    assert stream.task_index.tolist() == [0] * 10 + [1] * 10 + [2] * 10
    assert stream.test_task_index.tolist() == [0] * 4 + [1] * 4 + [2] * 4


def test_conflicting_stream_negates_alternate_tasks() -> None:
    """Task 1 must demand exactly the opposite mapping from task 0."""

    # Enough held-out samples to identify the map: with fewer samples than input
    # dimensions the least-squares fit is underdetermined and recovers nothing.
    stream = make_continual_stream(
        n_tasks=2, samples_per_task=8, test_samples=40, d_in=4, d_out=2, seed=0, conflicting=True
    )
    first = stream.test_task_index == 0
    second = stream.test_task_index == 1
    map_first = torch.linalg.lstsq(stream.test_inputs[first], stream.test_targets[first]).solution
    map_second = torch.linalg.lstsq(
        stream.test_inputs[second], stream.test_targets[second]
    ).solution
    assert torch.allclose(map_first, -map_second, atol=1e-4)


def test_stream_split_keeps_order_and_scores_every_task() -> None:
    stream = make_continual_stream(n_tasks=3, samples_per_task=10, test_samples=4, seed=0)
    split = stream_to_split(stream, name="continual")
    assert torch.equal(split.train_inputs, stream.inputs)
    assert split.test_inputs.shape[0] == 12
    assert split.val_inputs.shape[0] == 4  # first task only
    assert "without boundaries" in split.strategy


def test_stream_validates_dimensions() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_continual_stream(n_tasks=0)


def test_streaming_estimator_fits_and_predicts() -> None:
    stream = make_continual_stream(n_tasks=2, samples_per_task=40, test_samples=8, seed=0)
    estimator = StreamingEstimator(LearnerConfig(d_in=8, d_out=4, learning_rate=0.05))
    estimator.fit(stream.inputs.numpy(), stream.targets.numpy())
    predictions = estimator.predict(stream.test_inputs.numpy())
    assert predictions.shape == (16, 4)


def test_streaming_estimator_requires_fit_before_predict() -> None:
    estimator = StreamingEstimator(LearnerConfig(d_in=4, d_out=2))
    with pytest.raises(RuntimeError, match="before fit"):
        estimator.predict(torch.randn(3, 4).numpy())
