"""Mathematical invariants of the Titans track.

The update rule has three interacting gates, and the point of these tests is that each
one is checked *in isolation* against the source's equations, so an ablation that changes
the architecture's behaviour cannot be confused with one that changes its arithmetic.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from modern_nn_lab.experiments.tasks.sequence import IGNORE_INDEX, make_needle_task
from modern_nn_lab.tracks.titans import (
    MemoryTrace,
    NeuralMemory,
    SlidingWindowAttention,
    TitansConfig,
    TitansMAG,
)

# --------------------------------------------------------------------------------------
# The memory update rule: source equations 12-15
# --------------------------------------------------------------------------------------


def memory(**overrides: object) -> NeuralMemory:
    torch.manual_seed(0)
    settings: dict[str, object] = {"d_memory": 4, "depth": 1}
    settings.update(overrides)
    return NeuralMemory(8, **settings)  # type: ignore[arg-type]


def test_associative_loss_is_the_reconstruction_objective() -> None:
    module = memory()
    step_input = torch.randn(3, 8)
    state = module.initial_state(3)

    predicted = module.read(module.key(step_input), state.weights)
    expected = ((predicted - module.value(step_input)) ** 2).mean(dim=-1).sum()
    assert torch.allclose(module.associative_loss(step_input, state.weights), expected, atol=1e-6)


def test_first_write_matches_the_hand_computed_update() -> None:
    """M_1 = (1 - alpha) M_0 + S_1 with S_1 = -theta * grad, since S_0 = 0."""

    module = memory()
    step_input = torch.randn(2, 8)
    state = module.initial_state(2)
    theta, _, alpha = module.gates(step_input)

    weights = tuple(w.detach().clone().requires_grad_(True) for w in state.weights)
    loss = module.associative_loss(step_input, weights)
    gradient = torch.autograd.grad(loss, weights)[0]

    expected = (1.0 - alpha.view(-1, 1, 1)) * state.weights[0] - theta.view(-1, 1, 1) * gradient
    _, updated = module.step(step_input, state)
    assert torch.allclose(updated.weights[0], expected, atol=1e-6)


def test_momentum_accumulates_past_surprise() -> None:
    """S_2 = eta_2 S_1 - theta_2 grad_2, so the second step must remember the first."""

    module = memory()
    first, second = torch.randn(2, 8), torch.randn(2, 8)
    state = module.initial_state(2)
    _, after_first = module.step(first, state)
    _, after_second = module.step(second, after_first)

    theta, eta, _ = module.gates(second)
    weights = tuple(w.detach().clone().requires_grad_(True) for w in after_first.weights)
    loss = module.associative_loss(second, weights)
    gradient = torch.autograd.grad(loss, weights)[0]
    expected_momentum = (
        eta.view(-1, 1, 1) * after_first.momentum[0] - theta.view(-1, 1, 1) * gradient
    )
    assert torch.allclose(after_second.momentum[0], expected_momentum, atol=1e-6)


def test_disabling_momentum_reduces_the_update_to_a_bare_gradient_step() -> None:
    """Source equation 8: without past surprise, S_t is just -theta * grad."""

    module = memory(use_momentum=False)
    state = module.initial_state(2)
    _, after_first = module.step(torch.randn(2, 8), state)
    step_input = torch.randn(2, 8)
    _, after_second = module.step(step_input, after_first)

    theta, eta, _ = module.gates(step_input)
    assert torch.equal(eta, torch.zeros_like(eta))
    weights = tuple(w.detach().clone().requires_grad_(True) for w in after_first.weights)
    gradient = torch.autograd.grad(module.associative_loss(step_input, weights), weights)[0]
    assert torch.allclose(after_second.momentum[0], -theta.view(-1, 1, 1) * gradient, atol=1e-6)


def test_forgetting_gate_decays_the_memory() -> None:
    module = memory()
    state = module.initial_state(2)
    step_input = torch.randn(2, 8)

    # With alpha forced to 1 the previous memory is erased entirely (source eq. 13).
    with torch.no_grad():
        module.gate_projection.bias[2] = 40.0
    _, updated = module.step(step_input, state)
    _, _, forced = module.gates(step_input)
    assert float(forced.mean().detach()) == pytest.approx(1.0, abs=1e-6)
    # What remains is purely the new surprise term, with no trace of M_0.
    assert not torch.allclose(updated.weights[0], state.weights[0], atol=1e-3)


def test_disabling_forgetting_makes_writes_purely_additive() -> None:
    module = memory(use_forgetting=False)
    state = module.initial_state(2)
    step_input = torch.randn(2, 8)
    _, _, alpha = module.gates(step_input)
    assert torch.equal(alpha, torch.zeros_like(alpha))

    _, updated = module.step(step_input, state)
    assert torch.allclose(updated.weights[0], state.weights[0] + updated.momentum[0], atol=1e-6)


def test_frozen_memory_never_writes() -> None:
    module = memory(updates_enabled=False)
    state = module.initial_state(2)
    for _ in range(5):
        _, state = module.step(torch.randn(2, 8), state)
    base = module.initial_state(2)
    assert torch.equal(state.weights[0], base.weights[0])


def test_learning_rate_scale_is_the_update_rate_knob() -> None:
    fast = memory()
    slow = memory(learning_rate_scale=0.1)
    slow.load_state_dict(fast.state_dict())

    step_input = torch.randn(2, 8)
    fast_theta, _, _ = fast.gates(step_input)
    slow_theta, _, _ = slow.gates(step_input)
    assert torch.allclose(slow_theta, 0.1 * fast_theta, atol=1e-7)


def test_memory_stays_finite_over_long_sequences() -> None:
    """The bounded learning rate and normalized keys must prevent runaway writes."""

    module = memory(depth=2)
    state = module.initial_state(2)
    for _ in range(200):
        output, state = module.step(torch.randn(2, 8) * 3.0, state)
        assert torch.isfinite(output).all()
    assert all(torch.isfinite(w).all() for w in state.weights)
    assert max(float(w.abs().max()) for w in state.weights) < 100.0


def test_memory_reset_is_exact() -> None:
    module = memory()
    sequence = torch.randn(2, 6, 8)
    first = module(sequence)
    _ = module(torch.randn(2, 6, 8))
    assert torch.allclose(first, module(sequence), atol=1e-7)


def test_each_example_carries_its_own_memory() -> None:
    module = memory()
    first, second = torch.randn(1, 6, 8), torch.randn(1, 6, 8)
    alone = module(first)
    together = module(torch.cat([first, second], dim=0))
    assert torch.allclose(alone[0], together[0], atol=1e-6)


def test_memory_construction_is_validated() -> None:
    with pytest.raises(ValueError, match="d_model"):
        NeuralMemory(0)
    with pytest.raises(ValueError, match="depth"):
        NeuralMemory(8, depth=0)
    with pytest.raises(ValueError, match="learning_rate_scale"):
        NeuralMemory(8, learning_rate_scale=-1.0)
    with pytest.raises(ValueError, match="expected shape"):
        NeuralMemory(8)(torch.randn(2, 8))


# --------------------------------------------------------------------------------------
# The write/read diagnostics that this track exists to produce
# --------------------------------------------------------------------------------------


def test_trace_records_one_entry_per_token() -> None:
    module = memory()
    trace = MemoryTrace()
    module(torch.randn(2, 7, 8), trace)
    for series in trace.as_dict().values():
        assert len(series) == 7


def test_trace_reports_no_writes_when_the_memory_is_frozen() -> None:
    module = memory(updates_enabled=False)
    trace = MemoryTrace()
    module(torch.randn(2, 5, 8), trace)
    assert all(value == 0.0 for value in trace.write_norm)
    assert all(value == 0.0 for value in trace.surprise_norm)
    # The associative loss is still recorded: the memory is read even when not written.
    assert all(value > 0.0 for value in trace.loss)


def test_trace_reports_writes_when_the_memory_is_live() -> None:
    module = memory()
    trace = MemoryTrace()
    module(torch.randn(2, 5, 8), trace)
    assert all(value > 0.0 for value in trace.write_norm)
    assert all(0.0 <= value <= 1.0 for value in trace.forget_gate)
    assert all(0.0 <= value <= 1.0 for value in trace.momentum_gate)


# --------------------------------------------------------------------------------------
# Sliding-window attention: the short-term branch
# --------------------------------------------------------------------------------------


def test_sliding_window_sees_only_the_last_w_positions() -> None:
    attention = SlidingWindowAttention(8, n_heads=2, window=3)
    mask = attention.band_mask(6, torch.device("cpu"))
    # The mask uses a large finite penalty rather than -inf, so "allowed" means zero.
    allowed = mask == 0.0
    # Row t may attend to t, t-1, t-2 and nothing else.
    assert allowed[4].tolist() == [False, False, True, True, True, False]
    assert allowed[0].tolist() == [True, False, False, False, False, False]


def test_sliding_window_is_causal_and_window_limited() -> None:
    torch.manual_seed(0)
    attention = SlidingWindowAttention(8, n_heads=2, window=3).eval()
    sequence = torch.randn(1, 8, 8)

    with torch.no_grad():
        original = attention(sequence)
        edited = sequence.clone()
        edited[:, 2] += 5.0
        modified = attention(edited)

    # Position 1 is before the change; position 6 is further back than the window.
    assert torch.allclose(original[:, 1], modified[:, 1], atol=1e-6)
    assert torch.allclose(original[:, 6], modified[:, 6], atol=1e-6)
    assert not torch.allclose(original[:, 3], modified[:, 3], atol=1e-6)


def test_sliding_window_validates_construction() -> None:
    with pytest.raises(ValueError, match="divisible"):
        SlidingWindowAttention(9, n_heads=2)
    with pytest.raises(ValueError, match="window"):
        SlidingWindowAttention(8, window=0)


# --------------------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------------------


def build(**overrides: object) -> TitansMAG:
    torch.manual_seed(0)
    config = TitansConfig(d_model=16, n_heads=2, window=4, memory_depth=1)
    return TitansMAG(8, replace(config, **overrides))  # type: ignore[arg-type]


def test_model_is_causal() -> None:
    model = build().eval()
    tokens = torch.randint(0, 8, (2, 10))
    with torch.no_grad():
        original = model(tokens)
        edited = tokens.clone()
        edited[:, 7] = (edited[:, 7] + 1) % 8
        modified = model(edited)
    assert torch.allclose(original[:, :7], modified[:, :7], atol=1e-6)
    assert not torch.allclose(original[:, 7:], modified[:, 7:], atol=1e-6)


def test_model_shapes_determinism_and_gradients() -> None:
    first, second = build(), build()
    tokens = torch.randint(0, 8, (3, 9))
    assert first(tokens).shape == (3, 9, 8)
    assert torch.equal(first(tokens), second(tokens))

    first(tokens).pow(2).mean().backward()
    grads = [p.grad for p in first.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_persistent_tokens_are_stripped_from_the_output() -> None:
    model = build(persistent_tokens=3)
    assert model(torch.randint(0, 8, (2, 6))).shape == (2, 6, 8)


def test_branch_ablations_remove_exactly_one_branch() -> None:
    assert build(use_long_term=False).memory is None
    assert build(use_long_term=False).attention is not None
    assert build(use_short_term=False).attention is None
    assert build(use_short_term=False).memory is not None
    with pytest.raises(ValueError, match="at least one"):
        build(use_short_term=False, use_long_term=False)


def test_memory_trace_requires_a_memory() -> None:
    with pytest.raises(ValueError, match="no long-term memory"):
        build(use_long_term=False).memory_trace(torch.randint(0, 8, (2, 5)))


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        TitansConfig(d_model=0)
    with pytest.raises(ValueError, match="divisible"):
        TitansConfig(d_model=9, n_heads=2)
    with pytest.raises(ValueError, match="window"):
        TitansConfig(window=0)
    with pytest.raises(ValueError, match="memory_depth"):
        TitansConfig(memory_depth=0)
    with pytest.raises(ValueError, match="persistent_tokens"):
        TitansConfig(persistent_tokens=-1)


# --------------------------------------------------------------------------------------
# The needle task
# --------------------------------------------------------------------------------------


def test_needle_distance_shrinks_as_the_needle_moves_later() -> None:
    far = make_needle_task(n_sequences=10, n_pairs=5, needle_index=0, n_keys=8, seed=0)
    near = make_needle_task(n_sequences=10, n_pairs=5, needle_index=4, n_keys=8, seed=0)
    assert far.metadata["distance"] > near.metadata["distance"]
    assert far.name != near.name


def test_needle_target_is_the_value_written_at_the_needle_index() -> None:
    for index in range(4):
        split = make_needle_task(n_sequences=8, n_pairs=4, needle_index=index, n_keys=6, seed=1)
        inputs, targets = split.train_inputs, split.train_targets
        answer_index = int(split.metadata["answer_index"])  # type: ignore[arg-type]
        assert torch.equal(targets[:, answer_index], inputs[:, 2 * index + 1])
        assert torch.equal(inputs[:, answer_index], inputs[:, 2 * index])
        assert int((targets != IGNORE_INDEX).sum(dim=1).unique().item()) == 1


def test_repeated_needles_duplicate_the_binding() -> None:
    split = make_needle_task(n_sequences=6, n_pairs=5, needle_index=0, n_keys=8, repeats=3, seed=0)
    keys = split.train_inputs[:, 0:10:2]
    values = split.train_inputs[:, 1:10:2]
    assert torch.equal(keys[:, 0], keys[:, 1])
    assert torch.equal(keys[:, 1], keys[:, 2])
    assert torch.equal(values[:, 0], values[:, 2])


def test_needle_task_validates_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_needle_task(n_sequences=0)
    with pytest.raises(ValueError, match="n_keys"):
        make_needle_task(n_pairs=8, n_keys=4)
    with pytest.raises(ValueError, match="needle_index"):
        make_needle_task(n_pairs=4, needle_index=4, n_keys=8)
    with pytest.raises(ValueError, match="repeats"):
        make_needle_task(n_pairs=4, needle_index=3, repeats=2, n_keys=8)
