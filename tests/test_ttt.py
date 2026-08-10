"""Mathematical invariants of the Test-Time Training track.

The two tests that carry the most weight are the source's Theorem 1 — a linear inner model
with batch gradient descent, ``eta = 1/2`` and ``W_0 = 0`` *is* linear attention — and the
assertion that a forward pass performs gradient descent without mutating a single
``nn.Parameter``, which is the operational content of "test-time training of the hidden
state" as opposed to ordinary training.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from modern_nn_lab.experiments.tasks.sequence import IGNORE_INDEX, make_rebinding_task
from modern_nn_lab.tracks.ttt import TTT, TTTConfig, TTTLayer

# --------------------------------------------------------------------------------------
# The inner loop is gradient descent
# --------------------------------------------------------------------------------------


def bare_linear_layer(**overrides: object) -> TTTLayer:
    """A TTT-Linear layer with ``f(x) = W^T x``: no LayerNorm, no residual.

    This is the form the source's Theorem 1 and the hand-computed update assume.
    """

    settings: dict[str, object] = {
        "d_inner": 4,
        "inner_model": "linear",
        "layernorm_residual": False,
        "eta_base": 1.0,
    }
    settings.update(overrides)
    torch.manual_seed(0)
    return TTTLayer(8, **settings)  # type: ignore[arg-type]


def test_one_inner_step_matches_manual_gradient_descent() -> None:
    """W_1 = W_0 - eta * grad, with the gradient computed by hand.

    For ``f(x) = W^T x`` and ``l = ||W^T k - v||^2`` the gradient is ``2 k (W^T k - v)^T``.
    """

    layer = bare_linear_layer()
    step_input = torch.randn(3, 8)

    state = layer.initial_state(3)
    _, updated = layer.step(step_input, state, state)

    keys = layer.train_view(step_input)  # (B, d_inner)
    values = layer.label_view(step_input)
    weights = state.weights[0]  # (B, d_inner, d_inner)
    residual = torch.einsum("bij,bi->bj", weights, keys) - values
    manual_gradient = 2.0 * keys.unsqueeze(-1) * residual.unsqueeze(-2)
    eta = layer.inner_learning_rate(step_input).view(-1, 1, 1)

    assert torch.allclose(updated.weights[0], weights - eta * manual_gradient, atol=1e-6)


def test_inner_learning_rate_follows_the_source_formula() -> None:
    layer = bare_linear_layer(eta_base=0.7)
    step_input = torch.randn(4, 8)
    expected = 0.7 * torch.sigmoid(layer.learning_rate_projection(step_input))
    assert torch.allclose(layer.inner_learning_rate(step_input), expected, atol=1e-7)


def test_inner_loss_is_the_reconstruction_objective_of_equation_four() -> None:
    layer = bare_linear_layer()
    step_input = torch.randn(2, 8)
    state = layer.initial_state(2)

    prediction = torch.einsum("bij,bi->bj", state.weights[0], layer.train_view(step_input))
    expected = ((prediction - layer.label_view(step_input)) ** 2).sum()
    assert torch.allclose(layer.inner_loss(step_input, state.weights), expected, atol=1e-6)


def test_batch_gradient_descent_is_linear_attention() -> None:
    """Source Theorem 1, asserted directly.

    With a linear inner model, batch GD (every gradient taken at ``W_0``), ``eta = 1/2``
    and ``W_0 = 0``, the output rule reduces to ``z_t = sum_{s<=t} v_s (k_s . q_t)``.
    """

    layer = bare_linear_layer(update_rule="batch")
    with torch.no_grad():
        layer.initial_weights[0].zero_()  # W_0 = 0
        layer.learning_rate_projection.weight.zero_()
        layer.learning_rate_projection.bias.zero_()  # sigmoid(0) * 1.0 = 1/2

    sequence = torch.randn(2, 6, 8)
    keys = layer.train_view(sequence)
    values = layer.label_view(sequence)
    queries = layer.test_view(sequence)

    # Causal linear attention, written out.
    scores = torch.einsum("bsd,btd->bts", keys, queries)
    mask = torch.tril(torch.ones(6, 6))
    attended = torch.einsum("bts,bsd->btd", scores * mask, values)

    assert torch.allclose(layer(sequence), layer.output_projection(attended), atol=1e-5)


def test_online_and_batch_rules_agree_on_the_first_token_only() -> None:
    # Both take their first gradient at W_0, so step 1 matches; afterwards they diverge.
    online = bare_linear_layer(update_rule="online")
    batch = bare_linear_layer(update_rule="batch")
    batch.load_state_dict(online.state_dict())

    sequence = torch.randn(2, 4, 8)
    first_online, first_batch = online(sequence), batch(sequence)
    assert torch.allclose(first_online[:, 0], first_batch[:, 0], atol=1e-6)
    assert not torch.allclose(first_online[:, 1:], first_batch[:, 1:], atol=1e-4)


# --------------------------------------------------------------------------------------
# Inner loop versus outer loop: the acceptance criterion of this track
# --------------------------------------------------------------------------------------


def test_forward_pass_never_mutates_outer_parameters() -> None:
    """Test-time training changes the hidden state, never the model's parameters."""

    torch.manual_seed(0)
    model = TTT(6, TTTConfig(d_model=16, n_blocks=1))
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    model(torch.randint(0, 6, (3, 9)))

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), f"{name} was mutated by the forward pass"


def test_hidden_state_does_change_during_a_forward_pass() -> None:
    """The mirror image of the previous test: the *state* must move."""

    layer = bare_linear_layer()
    sequence = torch.randn(2, 5, 8)
    base = layer.initial_state(2)

    state = base
    snapshots = []
    for index in range(sequence.shape[1]):
        _, state = layer.step(sequence[:, index], state, base)
        snapshots.append(state.weights[0].detach().clone())

    assert not torch.allclose(snapshots[0], base.weights[0], atol=1e-6)
    for earlier, later in itertools.pairwise(snapshots):
        assert not torch.allclose(earlier, later, atol=1e-6)


def test_outer_gradients_flow_through_the_inner_update() -> None:
    """The views and W_0 must receive gradient *through* the inner gradient step."""

    layer = bare_linear_layer()
    layer.train()
    layer(torch.randn(2, 4, 8)).pow(2).mean().backward()

    assert layer.train_view.weight.grad is not None
    assert layer.label_view.weight.grad is not None
    assert layer.initial_weights[0].grad is not None
    # The training and label views only ever enter through the inner loss, so a non-zero
    # gradient on them proves the outer loop differentiated through the inner step.
    assert float(layer.train_view.weight.grad.abs().sum()) > 0
    assert float(layer.label_view.weight.grad.abs().sum()) > 0


def test_inner_loop_runs_under_no_grad() -> None:
    """Evaluation wraps the forward pass in `no_grad`; TTT must still learn there."""

    layer = bare_linear_layer()
    layer.eval()
    sequence = torch.randn(2, 5, 8)

    with torch.no_grad():
        live = layer(sequence)

    frozen = bare_linear_layer(learner_updates=False)
    frozen.load_state_dict(layer.state_dict())
    frozen.eval()
    with torch.no_grad():
        without = frozen(sequence)

    # If the inner loop had silently stopped under no_grad the two would coincide.
    assert not torch.allclose(live[:, 1:], without[:, 1:], atol=1e-5)


# --------------------------------------------------------------------------------------
# The required ablation
# --------------------------------------------------------------------------------------


def test_frozen_learner_keeps_the_state_at_w0_exactly() -> None:
    layer = bare_linear_layer(learner_updates=False)
    base = layer.initial_state(2)
    state = base
    for _ in range(5):
        _, state = layer.step(torch.randn(2, 8), state, base)
        assert torch.equal(state.weights[0], base.weights[0])


def test_frozen_learner_is_position_invariant() -> None:
    """With no inner updates the layer is a fixed function of each token alone."""

    layer = bare_linear_layer(learner_updates=False)
    repeated = torch.randn(2, 1, 8).expand(-1, 6, -1).contiguous()
    output = layer(repeated)
    for index in range(1, 6):
        assert torch.allclose(output[:, 0], output[:, index], atol=1e-6)


def test_live_learner_is_not_position_invariant() -> None:
    layer = bare_linear_layer()
    repeated = torch.randn(2, 1, 8).expand(-1, 6, -1).contiguous()
    output = layer(repeated)
    assert not torch.allclose(output[:, 0], output[:, 5], atol=1e-5)


# --------------------------------------------------------------------------------------
# Causality and reset semantics
# --------------------------------------------------------------------------------------


def test_no_information_leaks_from_future_tokens() -> None:
    torch.manual_seed(0)
    model = TTT(6, TTTConfig(d_model=16, n_blocks=1)).eval()
    tokens = torch.randint(0, 6, (2, 9))

    with torch.no_grad():
        original = model(tokens)
        edited = tokens.clone()
        edited[:, 6] = (edited[:, 6] + 1) % 6
        modified = model(edited)

    assert torch.allclose(original[:, :6], modified[:, :6], atol=1e-6)
    assert not torch.allclose(original[:, 6:], modified[:, 6:], atol=1e-6)


def test_reset_semantics_are_exact() -> None:
    """A fresh sequence must start from W_0 with no trace of the previous one."""

    layer = bare_linear_layer()
    sequence = torch.randn(2, 6, 8)

    first = layer(sequence)
    _ = layer(torch.randn(2, 6, 8))  # an unrelated sequence in between
    second = layer(sequence)
    assert torch.allclose(first, second, atol=1e-7)


def test_each_example_carries_its_own_learner() -> None:
    """Batching must not couple examples: one row's tokens cannot affect another's."""

    layer = bare_linear_layer()
    first = torch.randn(1, 6, 8)
    second = torch.randn(1, 6, 8)

    alone = layer(first)
    together = layer(torch.cat([first, second], dim=0))
    assert torch.allclose(alone[0], together[0], atol=1e-6)


# --------------------------------------------------------------------------------------
# Model-level behaviour and the task
# --------------------------------------------------------------------------------------


def test_mlp_inner_model_has_the_documented_shapes() -> None:
    layer = TTTLayer(16, d_inner=4, inner_model="mlp")
    state = layer.initial_state(3)
    assert len(state.weights) == 2
    assert state.weights[0].shape == (3, 4, 16)
    assert state.weights[1].shape == (3, 16, 4)
    assert layer(torch.randn(3, 5, 16)).shape == (3, 5, 16)


def test_mlp_inner_update_moves_both_weight_matrices() -> None:
    layer = TTTLayer(16, d_inner=4, inner_model="mlp")
    base = layer.initial_state(2)
    _, updated = layer.step(torch.randn(2, 16), base, base)
    for before, after in zip(base.weights, updated.weights, strict=True):
        assert not torch.allclose(before, after, atol=1e-7)


def test_model_determinism_and_serialization() -> None:
    torch.manual_seed(3)
    first = TTT(6, TTTConfig(d_model=16))
    torch.manual_seed(3)
    second = TTT(6, TTTConfig(d_model=16))
    tokens = torch.randint(0, 6, (2, 5))
    assert torch.equal(first(tokens), second(tokens))

    restored = TTT(6, TTTConfig(d_model=16))
    restored.load_state_dict(first.state_dict())
    assert torch.allclose(restored(tokens), first(tokens), atol=1e-7)


def test_construction_is_validated() -> None:
    with pytest.raises(ValueError, match="d_model"):
        TTTLayer(0)
    with pytest.raises(ValueError, match="d_inner"):
        TTTLayer(8, d_inner=0)
    with pytest.raises(ValueError, match="expected shape"):
        TTTLayer(8)(torch.randn(2, 8))
    with pytest.raises(ValueError, match="positive"):
        TTTConfig(d_model=0)
    with pytest.raises(ValueError, match="d_inner"):
        TTTConfig(d_inner=0)


def test_rebinding_task_overwrites_the_binding_it_queries() -> None:
    split = make_rebinding_task(n_sequences=20, n_pairs=3, n_keys=6, n_values=6, seed=0)
    inputs, targets = split.train_inputs, split.train_targets
    first_index = int(split.metadata["first_answer_index"])  # type: ignore[arg-type]
    second_index = int(split.metadata["second_answer_index"])  # type: ignore[arg-type]

    scored = (targets != IGNORE_INDEX).sum(dim=1)
    assert bool((scored == 2).all())

    for row in range(inputs.shape[0]):
        queried = int(inputs[row, first_index])
        keys_before = inputs[row, 0 : first_index - 1 : 2]
        values_before = inputs[row, 1 : first_index - 1 : 2]
        position = int((keys_before == queried).nonzero()[0])
        assert int(targets[row, first_index]) == int(values_before[position])

        rebind = first_index + 1
        keys_after = inputs[row, rebind : second_index - 1 : 2]
        values_after = inputs[row, rebind + 1 : second_index - 1 : 2]
        position_after = int((keys_after == queried).nonzero()[0])
        assert int(targets[row, second_index]) == int(values_after[position_after])


def test_rebinding_task_validates_arguments() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_rebinding_task(n_sequences=0)
    with pytest.raises(ValueError, match="n_keys"):
        make_rebinding_task(n_pairs=8, n_keys=4)
