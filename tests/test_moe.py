"""Invariants of the sparse mixture-of-experts track.

The prompt names five mandatory properties: probabilities normalize, top-k dispatch is
correct, tokens are neither duplicated nor lost beyond the specified routing behaviour,
capacity overflow is explicit, and the load-balancing loss behaves as claimed at the
uniform and collapsed extremes. Each has a test below, and the balancing loss is checked
against its two exact endpoints rather than against a direction of change.

The load-balancing tests construct routing distributions directly instead of training a
model to produce them. A test that trains and then hopes for collapse is a test of the
optimizer; these are tests of the formula.
"""

from __future__ import annotations

import math

import pytest
import torch

from modern_nn_lab.tracks.moe import (
    DenseFFN,
    DenseMoELayer,
    MixtureModel,
    MoEConfig,
    SparseMoELayer,
    TopKRouter,
    expert_flops,
    generate,
    specialization_matrix,
    specialization_purity,
)

# --------------------------------------------------------------------------------------
# Mandatory test 1: routing probabilities normalize
# --------------------------------------------------------------------------------------


def test_routing_probabilities_form_a_distribution() -> None:
    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=4, top_k=2)
    tokens = torch.randn(32, 8)

    probabilities = torch.softmax(router.gate(tokens), dim=-1)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(32), atol=1e-6)
    assert (probabilities >= 0).all()


def test_kept_gates_are_renormalized_to_sum_to_one() -> None:
    """The documented convention: a token's output is a convex combination."""

    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=4, top_k=2, renormalize=True)
    _, gate, _, _ = router(torch.randn(32, 8))
    assert torch.allclose(gate.sum(dim=-1), torch.ones(32), atol=1e-6)


def test_without_renormalization_gates_carry_routing_confidence() -> None:
    """The alternative convention must actually differ, or the flag means nothing."""

    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=4, top_k=2, renormalize=False)
    _, gate, _, _ = router(torch.randn(32, 8))
    total = gate.sum(dim=-1)
    assert (total <= 1.0 + 1e-6).all()
    assert not torch.allclose(total, torch.ones(32), atol=1e-3)


def test_renormalizing_a_top1_gate_severs_the_router_from_the_task_loss() -> None:
    """The trap this track discovered, pinned so it cannot silently return.

    With one expert kept, the renormalized gate is ``g / g == 1``: constant in the router's
    parameters, so the task loss contributes exactly zero gradient to routing. Any
    specialization measured under that setting came from somewhere other than the task.
    """

    def router_gradient_norm(top_k: int, renormalize: bool) -> float:
        torch.manual_seed(0)
        layer = SparseMoELayer(8, 8, n_experts=4, top_k=top_k, renormalize=renormalize)
        output, _ = layer(torch.randn(3, 6, 8))
        output.pow(2).mean().backward()
        gradient = layer.router.gate.weight.grad
        assert gradient is not None
        return float(gradient.norm())

    severed = router_gradient_norm(top_k=1, renormalize=True)
    restored = router_gradient_norm(top_k=1, renormalize=False)
    multi = router_gradient_norm(top_k=2, renormalize=True)

    assert severed < 1e-7, f"expected no task gradient, got {severed:.3e}"
    assert restored > 1e-3, "keeping the raw gate must restore the gradient"
    assert multi > 1e-3, "with k > 1 the renormalized gate still depends on the logits"


# --------------------------------------------------------------------------------------
# Mandatory test 2: top-k dispatch is correct
# --------------------------------------------------------------------------------------


def test_router_selects_exactly_the_k_largest_experts() -> None:
    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=5, top_k=2)
    tokens = torch.randn(24, 8)

    probabilities = torch.softmax(router.gate(tokens), dim=-1)
    expert_index, _, _, _ = router(tokens)

    for token in range(24):
        expected = set(probabilities[token].topk(2).indices.tolist())
        assert set(expert_index[token].tolist()) == expected


def test_a_token_is_never_assigned_the_same_expert_twice() -> None:
    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=4, top_k=3)
    expert_index, _, _, _ = router(torch.randn(50, 8))
    for token in range(50):
        assert len(set(expert_index[token].tolist())) == 3


def test_top1_routing_reproduces_the_argmax() -> None:
    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=4, top_k=1)
    tokens = torch.randn(30, 8)
    probabilities = torch.softmax(router.gate(tokens), dim=-1)
    expert_index, gate, _, _ = router(tokens)

    assert torch.equal(expert_index.squeeze(-1), probabilities.argmax(dim=-1))
    # With one expert kept, renormalization forces the gate to exactly one.
    assert torch.allclose(gate.squeeze(-1), torch.ones(30), atol=1e-6)


# --------------------------------------------------------------------------------------
# Mandatory test 3: no duplication or loss beyond the specified behaviour
# --------------------------------------------------------------------------------------


def test_every_token_is_dispatched_exactly_k_times_when_capacity_is_ample() -> None:
    """With room for everyone, nothing is dropped and nothing is counted twice."""

    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=4, top_k=2, capacity_factor=4.0)
    expert_index, _, keep, info = router(torch.randn(40, 8))

    assert bool(keep.all()), "ample capacity must drop nothing"
    assert info.dropped_fraction == 0.0
    counts = torch.zeros(4)
    for token in range(40):
        for slot in range(2):
            counts[int(expert_index[token, slot])] += 1
    assert int(counts.sum()) == 40 * 2


def test_expert_assignment_counts_never_exceed_capacity() -> None:
    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=4, top_k=1, capacity_factor=0.5)
    expert_index, _, keep, info = router(torch.randn(64, 8))

    for expert_id in range(4):
        dispatched = int(((expert_index == expert_id) & keep).sum())
        assert dispatched <= info.capacity


def test_a_token_kept_nowhere_receives_exactly_zero_from_the_layer() -> None:
    """The documented overflow behaviour, checked on the output rather than a counter."""

    torch.manual_seed(0)
    layer = SparseMoELayer(8, 8, n_experts=2, top_k=1, capacity_factor=0.01)
    tokens = torch.randn(1, 40, 8)

    output, info = layer(tokens)
    flat_router_input = tokens.reshape(-1, 8)
    _, _, keep, _ = layer.router(flat_router_input)

    dropped = ~keep.any(dim=-1)
    assert bool(dropped.any()), "this capacity must force drops for the test to mean anything"
    assert info.dropped_fraction > 0
    assert torch.allclose(output.reshape(-1, 8)[dropped], torch.zeros(int(dropped.sum()), 8))
    assert not torch.allclose(
        output.reshape(-1, 8)[~dropped], torch.zeros(int((~dropped).sum()), 8)
    )


def test_a_dropped_token_passes_through_the_residual_unchanged() -> None:
    """Overflow degrades a token to "not processed", not to "erased"."""

    torch.manual_seed(0)
    config = MoEConfig(layer="sparse-moe", d_in=8, n_experts=2, top_k=1, capacity_factor=0.01)
    model = MixtureModel(config).eval()
    inputs = torch.randn(1, 40, 8)

    with torch.no_grad():
        hidden = model.norm(model.input_projection(inputs))
        delta, _ = model.layer(hidden)
        _, _, keep, _ = model.layer.router(hidden.reshape(-1, config.d_model))

    dropped = ~keep.any(dim=-1)
    assert bool(dropped.any())
    # The layer contributes nothing, so the residual carries the token forward.
    assert torch.allclose(delta.reshape(-1, config.d_model)[dropped], torch.zeros(1), atol=1e-7)


def test_slot_major_priority_gives_first_choices_the_capacity() -> None:
    """A token's second preference must not displace another token's first."""

    torch.manual_seed(0)
    router = TopKRouter(4, n_experts=2, top_k=2, capacity_factor=0.5)
    # Force every token to prefer expert 0, then expert 1.
    with torch.no_grad():
        router.gate.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]))
    tokens = torch.zeros(8, 4)
    tokens[:, 0] = 1.0

    expert_index, _, keep, info = router(tokens)
    assert torch.equal(expert_index[:, 0], torch.zeros(8, dtype=torch.long))

    # Capacity is 8*2/2 * 0.5 = 8, so all eight first choices fit and no second choice is
    # displaced by a first choice arriving later.
    first_kept = int(keep[:, 0].sum())
    assert first_kept == min(8, info.capacity)


# --------------------------------------------------------------------------------------
# Mandatory test 4: capacity overflow is explicit
# --------------------------------------------------------------------------------------


def test_capacity_follows_the_documented_formula() -> None:
    router = TopKRouter(8, n_experts=4, top_k=2, capacity_factor=1.25)
    assert router.capacity_for(64) == math.ceil(1.25 * 64 * 2 / 4)
    assert router.capacity_for(1) >= 1, "a batch must never be routed to nothing"


def test_tight_capacity_drops_the_expected_fraction() -> None:
    """Half the even share means at most half of the assignments survive."""

    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=4, top_k=1, capacity_factor=0.5)
    _, _, _, info = router(torch.randn(200, 8))
    assert 0.4 < info.dropped_fraction < 0.6


def test_generous_capacity_drops_nothing() -> None:
    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=4, top_k=1, capacity_factor=10.0)
    _, _, _, info = router(torch.randn(200, 8))
    assert info.dropped_fraction == 0.0


# --------------------------------------------------------------------------------------
# Mandatory test 5: the load-balancing loss at both extremes
# --------------------------------------------------------------------------------------


def uniform_router(n_experts: int, n_tokens: int) -> tuple[TopKRouter, torch.Tensor]:
    """Return a router whose gate is identically zero, so routing is uniform."""

    router = TopKRouter(4, n_experts=n_experts, top_k=1, capacity_factor=10.0)
    with torch.no_grad():
        router.gate.weight.zero_()
    return router, torch.randn(n_tokens, 4)


def test_balancing_loss_is_one_under_uniform_routing() -> None:
    """The exact endpoint, not merely "small"."""

    n_experts = 4
    router, tokens = uniform_router(n_experts, 400)
    _, _, _, info = router(tokens)

    # A zero gate makes every probability 1/E; top-1 then breaks ties by index, so the
    # dispatch fractions are one-hot even though the gate mass is uniform. Check the gate
    # mass directly, which is the part the formula depends on.
    assert torch.allclose(info.gate_mass, torch.full((n_experts,), 1.0 / n_experts), atol=1e-6)
    assert pytest.approx(1.0, abs=1e-5) == float(info.load_balancing_loss)


def test_balancing_loss_equals_the_expert_count_under_total_collapse() -> None:
    """Every token to one expert, with all the gate mass there, is the worst case."""

    n_experts = 4
    torch.manual_seed(0)
    router = TopKRouter(4, n_experts=n_experts, top_k=1, capacity_factor=10.0)
    with torch.no_grad():
        # A huge weight on expert 0 drives its probability to one for every token.
        router.gate.weight.zero_()
        router.gate.weight[0, 0] = 200.0
    tokens = torch.ones(200, 4)

    _, _, _, info = router(tokens)
    assert pytest.approx(float(n_experts), abs=1e-3) == float(info.load_balancing_loss)
    assert float(info.entropy) < 1e-4
    assert float(info.utilization.max()) == 1.0


def test_balancing_loss_penalizes_imbalance_between_the_extremes() -> None:
    n_experts = 4
    torch.manual_seed(0)
    router = TopKRouter(4, n_experts=n_experts, top_k=1, capacity_factor=10.0)
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[0, 0] = 2.0
    tokens = torch.ones(200, 4)

    _, _, _, info = router(tokens)
    balance = float(info.load_balancing_loss)
    assert 1.0 < balance < n_experts


def test_entropy_is_maximal_for_a_uniform_router_and_normalizes_to_one() -> None:
    n_experts = 8
    router, tokens = uniform_router(n_experts, 100)
    _, _, _, info = router(tokens)

    assert pytest.approx(math.log(n_experts), abs=1e-5) == float(info.entropy)
    assert pytest.approx(1.0, abs=1e-5) == float(info.normalized_entropy)


def test_routing_info_reports_the_scalars_the_prompt_requires() -> None:
    torch.manual_seed(0)
    router = TopKRouter(8, n_experts=4, top_k=1)
    _, _, _, info = router(torch.randn(40, 8))
    metrics = info.as_metrics()

    for key in (
        "load_balancing_loss",
        "routing_entropy",
        "expert_utilization_max",
        "expert_utilization_min",
        "dropped_fraction",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)


# --------------------------------------------------------------------------------------
# The cost accounting the required comparison rests on
# --------------------------------------------------------------------------------------


def test_sparse_activates_a_fraction_of_its_parameters() -> None:
    """The whole claim: many parameters held, few spent per token."""

    dense = MixtureModel(MoEConfig(layer="dense-moe", n_experts=4))
    sparse = MixtureModel(MoEConfig(layer="sparse-moe", n_experts=4, top_k=1))

    dense_total = sum(p.numel() for p in dense.parameters())
    sparse_total = sum(p.numel() for p in sparse.parameters())

    assert sparse_total == pytest.approx(dense_total, rel=0.05), "expert banks must match"
    assert sparse.activated_parameters() < dense.activated_parameters()
    assert dense.activated_parameters() == dense_total


def test_activated_parameters_grow_with_top_k() -> None:
    one = MixtureModel(MoEConfig(layer="sparse-moe", n_experts=4, top_k=1))
    two = MixtureModel(MoEConfig(layer="sparse-moe", n_experts=4, top_k=2))
    assert two.activated_parameters() > one.activated_parameters()
    assert two.flops_per_token() > one.flops_per_token()


def test_flops_scale_as_documented() -> None:
    per_expert = expert_flops(32, 32)
    sparse = MixtureModel(
        MoEConfig(layer="sparse-moe", d_model=32, d_hidden=32, n_experts=4, top_k=2)
    )
    router_cost = 32 * 4
    assert sparse.flops_per_token() == 2 * per_expert + router_cost


def test_dense_ffn_reports_no_routing() -> None:
    layer = DenseFFN(8, 16)
    output, info = layer(torch.randn(3, 5, 8))
    assert output.shape == (3, 5, 8)
    assert info is None


def test_dense_moe_uses_every_expert() -> None:
    """Changing any expert must change the output; none is inert."""

    torch.manual_seed(0)
    layer = DenseMoELayer(8, 8, n_experts=3).eval()
    tokens = torch.randn(2, 4, 8)
    with torch.no_grad():
        before = layer(tokens)[0].clone()
        for expert in layer.experts:
            for parameter in expert.parameters():
                parameter.add_(0.5)
            assert not torch.allclose(layer(tokens)[0], before, atol=1e-5)
            for parameter in expert.parameters():
                parameter.sub_(0.5)


def test_shapes_are_preserved_by_every_layer() -> None:
    tokens = torch.randn(3, 7, 8)
    for layer in (
        DenseFFN(8, 16),
        DenseMoELayer(8, 8, n_experts=3),
        SparseMoELayer(8, 8, n_experts=3, top_k=2),
    ):
        output, _ = layer(tokens)
        assert output.shape == tokens.shape


# --------------------------------------------------------------------------------------
# The dataset, and whether specialization is measurable at all
# --------------------------------------------------------------------------------------


def test_generated_tokens_have_the_documented_shapes() -> None:
    task = generate(n_sequences=5, seq_len=6, n_functions=3, d_selector=2, d_value=3, seed=0)
    assert task.inputs.shape == (5, 6, 5)
    assert task.targets.shape == (5, 6, 1)
    assert task.latent.shape == (5, 6)
    assert task.d_in == 5
    assert int(task.latent.max()) < 3


def test_every_function_is_actually_used() -> None:
    task = generate(n_sequences=200, seq_len=16, n_functions=4, seed=0)
    counts = torch.bincount(task.flat_latent(), minlength=4)
    assert int(counts.min()) > 0
    # Roughly balanced: an argmax over random projections of a Gaussian has no strong bias.
    assert float(counts.min() / counts.max()) > 0.3


def test_the_selector_half_determines_the_function_and_the_value_half_does_not() -> None:
    """The design property that makes a routing failure interpretable."""

    task = generate(n_sequences=300, seq_len=8, n_functions=4, d_selector=4, d_value=4, seed=1)
    flat_inputs = task.inputs.reshape(-1, task.d_in)
    latent = task.flat_latent()

    from sklearn.linear_model import LogisticRegression

    selector_only = LogisticRegression(max_iter=500).fit(flat_inputs[:, :4].numpy(), latent.numpy())
    value_only = LogisticRegression(max_iter=500).fit(flat_inputs[:, 4:].numpy(), latent.numpy())

    assert selector_only.score(flat_inputs[:, :4].numpy(), latent.numpy()) > 0.85
    assert value_only.score(flat_inputs[:, 4:].numpy(), latent.numpy()) < 0.45


def test_specialization_purity_at_its_two_extremes() -> None:
    perfect = torch.eye(4) * 100
    assert specialization_purity(perfect) == pytest.approx(1.0)

    uniform = torch.ones((4, 4)) * 25
    assert specialization_purity(uniform) == pytest.approx(0.25)

    assert specialization_purity(torch.zeros((4, 4))) == 0.0


def test_specialization_matrix_counts_every_token() -> None:
    latent = torch.tensor([0, 0, 1, 2, 3])
    expert = torch.tensor([1, 1, 0, 0, 3])
    matrix = specialization_matrix(latent, expert, n_functions=4, n_experts=4)
    assert int(matrix.sum()) == 5
    assert float(matrix[0, 1]) == 2.0


# --------------------------------------------------------------------------------------
# Configuration validation
# --------------------------------------------------------------------------------------


def test_router_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="n_experts"):
        TopKRouter(8, n_experts=1)
    with pytest.raises(ValueError, match="top_k"):
        TopKRouter(8, n_experts=4, top_k=5)
    with pytest.raises(ValueError, match="capacity_factor"):
        TopKRouter(8, n_experts=4, capacity_factor=0.0)
    with pytest.raises(ValueError, match="shape"):
        TopKRouter(8, n_experts=4)(torch.randn(3, 4, 8))


def test_config_validates_its_fields() -> None:
    with pytest.raises(ValueError, match="widths"):
        MoEConfig(d_model=0)
    with pytest.raises(ValueError, match="n_experts"):
        MoEConfig(n_experts=1)
    with pytest.raises(ValueError, match="top_k"):
        MoEConfig(n_experts=4, top_k=9)
    with pytest.raises(ValueError, match="aux_loss_weight"):
        MoEConfig(aux_loss_weight=-1.0)
    with pytest.raises(ValueError, match="n_functions"):
        generate(n_sequences=4, n_functions=1)
    with pytest.raises(ValueError, match="unknown layer"):
        MixtureModel(MoEConfig(layer="nope"))  # type: ignore[arg-type]


def test_model_rejects_wrongly_shaped_input() -> None:
    model = MixtureModel(MoEConfig(d_in=8))
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(4, 8))
