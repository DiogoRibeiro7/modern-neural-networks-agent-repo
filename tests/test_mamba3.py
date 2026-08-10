"""Mathematical invariants of the Mamba-3 track.

The tests that matter here check the implementation against the *source's own* equations:
the discretization coefficients of Proposition 1 and Table 1, the equivalence between the
block-rotation form (Proposition 2) and the RoPE form (Proposition 3) that the
implementation actually uses, and the MIMO decomposition of equations (12)-(14).
"""

from __future__ import annotations

import math

import pytest
import torch

from modern_nn_lab.experiments.tasks.sequence import make_state_tracking_task
from modern_nn_lab.tracks.mamba3 import (
    Mamba3,
    Mamba3Config,
    SelectiveSSM,
    parity_reference,
    rotate_pairs,
)

# --------------------------------------------------------------------------------------
# Rotations
# --------------------------------------------------------------------------------------


def test_rotate_pairs_matches_an_explicit_rotation_matrix() -> None:
    # The source defines R(t) = [[cos, -sin], [sin, cos]] and applies its transpose.
    vectors = torch.tensor([[[1.0], [0.0]]])  # one pair, rank 1
    angle = torch.tensor([[math.pi / 2]])
    rotated = rotate_pairs(vectors, angle)[0, :, 0]
    # R(-pi/2) @ [1, 0] = [cos, -sin] applied as transpose -> [0, -1]
    assert rotated.tolist() == pytest.approx([0.0, -1.0], abs=1e-6)


def test_rotations_compose_additively_and_invert() -> None:
    vectors = torch.randn(2, 6, 3)
    first = torch.randn(2, 3)
    second = torch.randn(2, 3)
    composed = rotate_pairs(rotate_pairs(vectors, first), second)
    combined = rotate_pairs(vectors, first + second)
    assert torch.allclose(composed, combined, atol=1e-5)
    assert torch.allclose(rotate_pairs(rotate_pairs(vectors, first), -first), vectors, atol=1e-5)


def test_rotations_preserve_pair_norms() -> None:
    vectors = torch.randn(3, 8, 2)
    angle = torch.randn(3, 4)
    before = vectors.reshape(3, 4, 2, 2).pow(2).sum(dim=2)
    after = rotate_pairs(vectors, angle).reshape(3, 4, 2, 2).pow(2).sum(dim=2)
    assert torch.allclose(before, after, atol=1e-5)


def test_rotate_pairs_validates_shapes() -> None:
    with pytest.raises(ValueError, match="even"):
        rotate_pairs(torch.randn(1, 3, 1), torch.randn(1, 1))
    with pytest.raises(ValueError, match="angles"):
        rotate_pairs(torch.randn(1, 4, 1), torch.randn(1, 3))


# --------------------------------------------------------------------------------------
# Discretization coefficients: source Table 1 and Proposition 1
# --------------------------------------------------------------------------------------


def test_euler_variant_reproduces_the_mamba2_coefficients() -> None:
    # Table 1, exponential-Euler row: alpha = exp(dt A), beta absent, gamma = dt.
    torch.manual_seed(0)
    layer = SelectiveSSM(8, heads=2, state_size=4, trapezoidal=False, rotary=False)
    coefficients = layer.coefficients(torch.randn(3, 8))

    assert torch.allclose(coefficients["lam"], torch.ones_like(coefficients["lam"]))
    assert torch.allclose(coefficients["beta"], torch.zeros_like(coefficients["beta"]), atol=1e-7)
    assert torch.allclose(coefficients["gamma"], coefficients["delta"], atol=1e-6)
    assert torch.allclose(
        coefficients["alpha"],
        torch.exp(coefficients["delta"] * coefficients["a"]),
        atol=1e-6,
    )


def test_trapezoidal_coefficients_match_proposition_one() -> None:
    # Proposition 1: beta = (1 - lambda) dt exp(dt A), gamma = lambda dt.
    torch.manual_seed(0)
    layer = SelectiveSSM(8, heads=2, state_size=4, trapezoidal=True, rotary=False)
    coefficients = layer.coefficients(torch.randn(4, 8))

    delta, alpha, lam = coefficients["delta"], coefficients["alpha"], coefficients["lam"]
    assert torch.allclose(coefficients["gamma"], lam * delta, atol=1e-6)
    assert torch.allclose(coefficients["beta"], (1.0 - lam) * delta * alpha, atol=1e-6)
    assert bool(((lam > 0.0) & (lam < 1.0)).all())


def test_lambda_one_recovers_the_euler_recurrence_exactly() -> None:
    # Remark 2: the exponential-trapezoidal rule generalizes Mamba-2's Euler rule, which
    # is recovered at lambda = 1.
    torch.manual_seed(0)
    trapezoidal = SelectiveSSM(8, heads=2, state_size=4, rotary=False)
    euler = SelectiveSSM(8, heads=2, state_size=4, trapezoidal=False, rotary=False)
    # The trapezoidal layer owns an extra projection, so identical seeds do NOT give
    # identical shared weights. Copy them instead, so lambda is the only difference.
    trapezoidal.load_state_dict(euler.state_dict(), strict=False)

    with torch.no_grad():
        # Force sigmoid(.) -> 1 so that beta vanishes.
        assert trapezoidal.lambda_projection is not None
        trapezoidal.lambda_projection.weight.zero_()
        trapezoidal.lambda_projection.bias.fill_(40.0)

    sequence = torch.randn(2, 6, 8)
    assert torch.allclose(trapezoidal(sequence), euler(sequence), atol=1e-5)


def test_trapezoidal_term_actually_changes_the_output() -> None:
    torch.manual_seed(0)
    trapezoidal = SelectiveSSM(8, heads=2, state_size=4, rotary=False)
    euler = SelectiveSSM(8, heads=2, state_size=4, trapezoidal=False, rotary=False)
    trapezoidal.load_state_dict(euler.state_dict(), strict=False)
    with torch.no_grad():
        assert trapezoidal.lambda_projection is not None
        trapezoidal.lambda_projection.weight.zero_()
        trapezoidal.lambda_projection.bias.zero_()  # lambda = 1/2, classical trapezoid

    sequence = torch.randn(2, 6, 8)
    assert not torch.allclose(trapezoidal(sequence), euler(sequence), atol=1e-4)


# --------------------------------------------------------------------------------------
# Proposition 2 versus Proposition 3: the RoPE trick must be an identity
# --------------------------------------------------------------------------------------


def block_rotation_reference(layer: SelectiveSSM, sequence: torch.Tensor) -> torch.Tensor:
    """Run source equation (9): rotate the *state*, leave ``B`` and ``C`` alone.

    The implementation instead runs equation (10), which rotates ``B`` and ``C`` by the
    accumulated rotation and leaves the state alone. Propositions 2 and 3 say the two are
    the same function; this reference is what makes that claim testable.
    """

    batch, length, _ = sequence.shape
    state = torch.zeros(batch, layer.heads, layer.state_size, layer.head_dim)
    outputs = []

    for index in range(length):
        step_input = sequence[:, index]
        coefficients = layer.coefficients(step_input)
        alpha, delta = coefficients["alpha"], coefficients["delta"]

        b_matrix = layer.b_projection(step_input).view(
            batch, layer.heads, layer.state_size, layer.rank
        )
        c_matrix = layer.c_projection(step_input).view(
            batch, layer.heads, layer.state_size, layer.rank
        )
        assert layer.theta_projection is not None
        theta = layer.theta_projection(step_input).view(batch, layer.heads, layer.state_size // 2)

        # Rotate the carried state by R(delta * theta) — equation (9).
        rotated_state = rotate_pairs(state, -delta.unsqueeze(-1) * theta)

        head_inputs = step_input.view(batch, layer.heads, layer.head_dim)
        expanded = head_inputs.unsqueeze(-1) * layer.rank_scale.t()
        outer = torch.einsum("bhnr,bhpr->bhnp", b_matrix, expanded)

        state = alpha[..., None, None] * rotated_state + delta[..., None, None] * outer
        read_out = torch.einsum("bhnr,bhnp->bhp", c_matrix, state)
        outputs.append(
            layer.output_projection(
                read_out.reshape(batch, layer.d_model) + layer.skip * step_input
            )
        )

    return torch.stack(outputs, dim=1)


def test_rope_form_equals_the_block_rotation_form() -> None:
    """Proposition 2 and Proposition 3 must describe the same function."""

    torch.manual_seed(0)
    # Euler discretization, so the reference above matches term for term.
    layer = SelectiveSSM(8, heads=2, state_size=4, rank=1, trapezoidal=False, rotary=True)
    sequence = torch.randn(3, 7, 8)
    assert torch.allclose(layer(sequence), block_rotation_reference(layer, sequence), atol=1e-5)


def test_rope_form_equals_block_rotation_form_with_mimo() -> None:
    torch.manual_seed(1)
    layer = SelectiveSSM(8, heads=2, state_size=4, rank=3, trapezoidal=False, rotary=True)
    sequence = torch.randn(2, 5, 8)
    assert torch.allclose(layer(sequence), block_rotation_reference(layer, sequence), atol=1e-5)


# --------------------------------------------------------------------------------------
# MIMO: source equations (12)-(14)
# --------------------------------------------------------------------------------------


def test_mimo_equals_the_sum_of_rank_one_recurrences() -> None:
    """Equations (12)-(14): a rank-R update is R SISO updates sharing alpha and delta."""

    torch.manual_seed(0)
    layer = SelectiveSSM(8, heads=2, state_size=4, rank=3, trapezoidal=False, rotary=False)
    sequence = torch.randn(2, 5, 8)

    batch, length, _ = sequence.shape
    per_rank = torch.zeros(batch, layer.heads, layer.rank, layer.state_size, layer.head_dim)
    outputs = []
    for index in range(length):
        step_input = sequence[:, index]
        coefficients = layer.coefficients(step_input)
        alpha, delta = coefficients["alpha"], coefficients["delta"]
        b_matrix = layer.b_projection(step_input).view(batch, layer.heads, layer.state_size, -1)
        c_matrix = layer.c_projection(step_input).view(batch, layer.heads, layer.state_size, -1)
        head_inputs = step_input.view(batch, layer.heads, layer.head_dim)

        for rank_index in range(layer.rank):
            scaled = head_inputs * layer.rank_scale[rank_index]
            outer = b_matrix[..., rank_index].unsqueeze(-1) * scaled.unsqueeze(-2)
            per_rank[:, :, rank_index] = (
                alpha[..., None, None] * per_rank[:, :, rank_index] + delta[..., None, None] * outer
            )

        summed = per_rank.sum(dim=2)  # equation (13)
        read_out = sum(
            torch.einsum("bhn,bhnp->bhp", c_matrix[..., i], summed) for i in range(layer.rank)
        )
        outputs.append(
            layer.output_projection(
                read_out.reshape(batch, layer.d_model) + layer.skip * step_input
            )
        )

    assert torch.allclose(layer(sequence), torch.stack(outputs, dim=1), atol=1e-5)


def test_rank_one_mimo_is_the_siso_recurrence() -> None:
    torch.manual_seed(0)
    layer = SelectiveSSM(8, heads=2, state_size=4, rank=1, trapezoidal=False, rotary=False)
    with torch.no_grad():
        layer.rank_scale.fill_(1.0)

    sequence = torch.randn(2, 5, 8)
    batch = 2
    state = torch.zeros(batch, layer.heads, layer.state_size, layer.head_dim)
    outputs = []
    for index in range(sequence.shape[1]):
        step_input = sequence[:, index]
        coefficients = layer.coefficients(step_input)
        b_vector = layer.b_projection(step_input).view(batch, layer.heads, layer.state_size)
        c_vector = layer.c_projection(step_input).view(batch, layer.heads, layer.state_size)
        head_inputs = step_input.view(batch, layer.heads, layer.head_dim)
        outer = b_vector.unsqueeze(-1) * head_inputs.unsqueeze(-2)
        state = (
            coefficients["alpha"][..., None, None] * state
            + coefficients["delta"][..., None, None] * outer
        )
        read_out = torch.einsum("bhn,bhnp->bhp", c_vector, state)
        outputs.append(
            layer.output_projection(
                read_out.reshape(batch, layer.d_model) + layer.skip * step_input
            )
        )
    assert torch.allclose(layer(sequence), torch.stack(outputs, dim=1), atol=1e-5)


# --------------------------------------------------------------------------------------
# Structure of the ablations
# --------------------------------------------------------------------------------------


def test_without_rotation_the_transition_is_a_positive_scalar() -> None:
    """A real SSM's state map is a non-negative scalar: no mixing across state pairs.

    This is the property the source identifies as the reason real transitions cannot
    represent parity, so it is worth asserting rather than assuming.
    """

    torch.manual_seed(0)
    layer = SelectiveSSM(8, heads=2, state_size=4, trapezoidal=False, rotary=False)
    assert layer.theta_projection is None

    step_input = torch.randn(1, 8)
    base = layer.initial_state(1)
    perturbation = torch.randn_like(base.state)

    _, without = layer.step(step_input, base)
    perturbed = type(base)(state=base.state + perturbation, pending=base.pending, angle=base.angle)
    _, with_perturbation = layer.step(step_input, perturbed)

    response = with_perturbation.state - without.state
    alpha = layer.coefficients(step_input)["alpha"]
    assert torch.allclose(response, alpha[..., None, None] * perturbation, atol=1e-6)
    assert bool((alpha > 0).all())


def test_rotation_mixes_state_pairs() -> None:
    torch.manual_seed(0)
    layer = SelectiveSSM(8, heads=2, state_size=4, trapezoidal=False, rotary=True)
    real = SelectiveSSM(8, heads=2, state_size=4, trapezoidal=False, rotary=False)
    layer.load_state_dict(real.state_dict(), strict=False)
    sequence = torch.randn(2, 5, 8)
    assert not torch.allclose(layer(sequence), real(sequence), atol=1e-4)


def test_parity_is_representable_by_an_explicit_rotation() -> None:
    """The source's motivating construction: h_t = R(pi x_t) h_{t-1} solves parity."""

    split = make_state_tracking_task(n_sequences=8, seq_len=12, n_states=2, seed=0)
    tokens = split.train_inputs
    assert torch.equal(parity_reference(tokens), split.train_targets)


# --------------------------------------------------------------------------------------
# Model-level behaviour
# --------------------------------------------------------------------------------------


def build_model(vocab_size: int = 6, **overrides: object) -> Mamba3:
    torch.manual_seed(0)
    settings = {"d_model": 16, "n_blocks": 1, "heads": 2, "state_size": 4, "rank": 2}
    settings.update(overrides)
    return Mamba3(vocab_size, Mamba3Config(**settings))  # type: ignore[arg-type]


def test_model_is_causal() -> None:
    model = build_model().eval()
    tokens = torch.randint(0, 6, (2, 9))
    with torch.no_grad():
        original = model(tokens)
        edited = tokens.clone()
        edited[:, 6] = (edited[:, 6] + 1) % 6
        modified = model(edited)
    assert torch.allclose(original[:, :6], modified[:, :6], atol=1e-6)
    assert not torch.allclose(original[:, 6:], modified[:, 6:], atol=1e-6)


def test_model_shapes_and_determinism() -> None:
    first = build_model()
    second = build_model()
    tokens = torch.randint(0, 6, (3, 7))
    assert first(tokens).shape == (3, 7, 6)
    assert torch.equal(first(tokens), second(tokens))


def test_model_gradients_are_finite_and_reach_every_mechanism() -> None:
    model = build_model()
    model(torch.randint(0, 6, (2, 8))).pow(2).mean().backward()
    grads = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads.values())
    assert any("theta_projection" in name for name in grads)
    assert any("lambda_projection" in name for name in grads)
    assert any("rank_scale" in name for name in grads)


def test_state_stays_finite_over_long_sequences() -> None:
    torch.manual_seed(0)
    layer = SelectiveSSM(8, heads=2, state_size=4, rank=2)
    sequence = torch.randn(2, 400, 8) * 5.0
    output = layer(sequence)
    assert torch.isfinite(output).all()


def test_serialization_round_trip() -> None:
    model = build_model()
    tokens = torch.randint(0, 6, (2, 5))
    expected = model(tokens)
    restored = build_model()
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(restored(tokens), expected, atol=1e-7)


def test_construction_is_validated() -> None:
    with pytest.raises(ValueError, match="positive"):
        SelectiveSSM(0)
    with pytest.raises(ValueError, match="divisible"):
        SelectiveSSM(9, heads=2)
    with pytest.raises(ValueError, match="even"):
        SelectiveSSM(8, heads=2, state_size=5, rotary=True)
    with pytest.raises(ValueError, match="dt_min"):
        SelectiveSSM(8, heads=2, dt_min=0.5, dt_max=0.1)
    layer = SelectiveSSM(8, heads=2, state_size=4)
    state = layer.initial_state(2)
    with pytest.raises(ValueError, match="expected shape"):
        layer.step(torch.zeros(2, 5), state)
    with pytest.raises(ValueError, match="expected shape"):
        layer(torch.zeros(2, 8))


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        Mamba3Config(d_model=0)
    with pytest.raises(ValueError, match="divisible"):
        Mamba3Config(d_model=9, heads=2)
    with pytest.raises(ValueError, match="even"):
        Mamba3Config(state_size=5, rotary=True)
    with pytest.raises(ValueError, match="positive"):
        Mamba3Config(rank=0)
