"""Mathematical invariants of the Kolmogorov-Arnold track.

These tests target the mechanism, not the plumbing: partition of unity, hand-computable
spline evaluation, function-preserving grid updates, and the exact meaning of each
ablation flag.
"""

from __future__ import annotations

import math

import pytest
import torch

from modern_nn_lab.tracks.kan import (
    KAN,
    MLP,
    KANConfig,
    KANLayer,
    adaptive_grid,
    b_spline_basis,
    build_grid,
    count_parameters,
    curve_to_coefficients,
    match_parameter_budget,
)


def interior_points(n: int = 64, low: float = -0.95, high: float = 0.95) -> torch.Tensor:
    """Points strictly inside the default domain, where partition of unity holds."""

    return torch.linspace(low, high, n).unsqueeze(-1)


# --------------------------------------------------------------------------------------
# Spline basis
# --------------------------------------------------------------------------------------


def test_grid_shape_and_spacing() -> None:
    grid = build_grid(3, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0))
    assert grid.shape == (3, 5 + 2 * 3 + 1)
    spacing = grid[:, 1:] - grid[:, :-1]
    assert torch.allclose(spacing, torch.full_like(spacing, 0.4))
    # Every input dimension starts from the same uniform grid.
    assert torch.allclose(grid[0], grid[1])


def test_grid_validates_arguments() -> None:
    with pytest.raises(ValueError, match="in_features"):
        build_grid(0, grid_size=5, spline_order=3)
    with pytest.raises(ValueError, match="grid_size"):
        build_grid(1, grid_size=0, spline_order=3)
    with pytest.raises(ValueError, match="spline_order"):
        build_grid(1, grid_size=5, spline_order=-1)
    with pytest.raises(ValueError, match="grid_range"):
        build_grid(1, grid_size=5, spline_order=3, grid_range=(1.0, -1.0))


@pytest.mark.parametrize("order", [1, 2, 3])
def test_basis_is_a_partition_of_unity_inside_the_domain(order: int) -> None:
    grid = build_grid(1, grid_size=7, spline_order=order)
    basis = b_spline_basis(interior_points(), grid, spline_order=order)
    assert basis.shape == (64, 1, 7 + order)
    assert torch.allclose(basis.sum(dim=-1), torch.ones(64, 1), atol=1e-5)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_basis_is_non_negative(order: int) -> None:
    grid = build_grid(1, grid_size=7, spline_order=order)
    basis = b_spline_basis(interior_points(), grid, spline_order=order)
    assert (basis >= -1e-6).all()


def test_basis_has_local_support() -> None:
    # A B-spline of order k is supported on k + 1 knot cells, so at any point at most
    # k + 1 basis functions can be non-zero.
    order = 3
    grid = build_grid(1, grid_size=10, spline_order=order)
    basis = b_spline_basis(interior_points(200), grid, spline_order=order)
    assert int((basis > 1e-8).sum(dim=-1).max()) <= order + 1


def test_degree_one_basis_matches_the_hat_function() -> None:
    # With G = 2 on [-1, 1] and k = 1 the knots are [-1.5, -1, 0, 1, 1.5]. The basis
    # function centred at 0 is the hat that peaks at x = 0 and vanishes at +-1.
    grid = build_grid(1, grid_size=2, spline_order=1, grid_range=(-1.0, 1.0))
    points = torch.tensor([[-1.0], [-0.5], [0.0], [0.5]])
    basis = b_spline_basis(points, grid, spline_order=1)[:, 0, :]
    hat = basis[:, 1]
    assert hat.tolist() == pytest.approx([0.0, 0.5, 1.0, 0.5], abs=1e-6)


def test_basis_validates_shapes() -> None:
    grid = build_grid(2, grid_size=5, spline_order=3)
    with pytest.raises(ValueError, match="inputs must have shape"):
        b_spline_basis(torch.zeros(4), grid, spline_order=3)
    with pytest.raises(ValueError, match="features"):
        b_spline_basis(torch.zeros(4, 3), grid, spline_order=3)
    with pytest.raises(ValueError, match="too few"):
        b_spline_basis(torch.zeros(4, 2), grid, spline_order=8)


def test_curve_to_coefficients_reproduces_a_known_curve() -> None:
    # A cubic spline's approximation error scales as h^4, so the grid has to be fine
    # enough that the fit residual is below the tolerance being asserted.
    grid = build_grid(1, grid_size=24, spline_order=3)
    points = interior_points(200)
    values = torch.sin(2.0 * math.pi * points).unsqueeze(-1)  # (N, 1, 1)
    coefficients = curve_to_coefficients(points, values, grid, spline_order=3)
    basis = b_spline_basis(points, grid, spline_order=3)
    reconstructed = torch.einsum("bip,oip->bio", basis, coefficients)
    assert torch.allclose(reconstructed, values, atol=1e-3)


def test_curve_to_coefficients_validates_shapes() -> None:
    grid = build_grid(1, grid_size=5, spline_order=3)
    with pytest.raises(ValueError, match="outputs must have shape"):
        curve_to_coefficients(torch.zeros(4, 1), torch.zeros(4, 1), grid, spline_order=3)
    with pytest.raises(ValueError, match="agree on batch"):
        curve_to_coefficients(torch.zeros(4, 1), torch.zeros(5, 1, 1), grid, spline_order=3)


def test_adaptive_grid_is_monotone_and_correctly_sized() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(500, 3) * 2.0
    grid = adaptive_grid(inputs, grid_size=6, spline_order=3)
    assert grid.shape == (3, 6 + 2 * 3 + 1)
    assert (grid[:, 1:] - grid[:, :-1] > 0).all()


def test_adaptive_grid_survives_a_constant_feature() -> None:
    inputs = torch.cat([torch.randn(100, 1), torch.full((100, 1), 0.5)], dim=1)
    grid = adaptive_grid(inputs, grid_size=5, spline_order=3)
    assert torch.isfinite(grid).all()
    assert (grid[:, 1:] - grid[:, :-1] > 0).all()


def test_adaptive_grid_validates_arguments() -> None:
    with pytest.raises(ValueError, match="shape"):
        adaptive_grid(torch.zeros(5), grid_size=5, spline_order=3)
    with pytest.raises(ValueError, match="at least one sample"):
        adaptive_grid(torch.zeros(0, 2), grid_size=5, spline_order=3)
    with pytest.raises(ValueError, match="uniform_mixture"):
        adaptive_grid(torch.zeros(4, 2), grid_size=5, spline_order=3, uniform_mixture=2.0)


# --------------------------------------------------------------------------------------
# Layer
# --------------------------------------------------------------------------------------


def test_layer_shapes_and_parameter_accounting() -> None:
    layer = KANLayer(3, 4, grid_size=5, spline_order=3)
    assert layer(torch.zeros(7, 3)).shape == (7, 4)
    # 3 * 4 edges, each holding G + k spline coefficients plus one base weight.
    assert count_parameters(layer) == 3 * 4 * (5 + 3) + 3 * 4


def test_layer_edge_function_matches_hand_computed_coefficients() -> None:
    layer = KANLayer(1, 1, grid_size=4, spline_order=1, use_base_branch=False)
    with torch.no_grad():
        # Degree-1 splines interpolate their coefficients at the interior knots, so a
        # coefficient vector is exactly the function's value at those knots.
        layer.spline_weight.copy_(torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]]))

    knots = layer.grid[0, layer.spline_order : -layer.spline_order]
    values = layer(knots.unsqueeze(-1))[:, 0]
    assert values.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0], abs=1e-5)


def test_layer_forward_equals_the_documented_decomposition() -> None:
    torch.manual_seed(0)
    layer = KANLayer(2, 3, grid_size=5, spline_order=3)
    inputs = torch.rand(6, 2) * 1.6 - 0.8

    basis = layer.basis(inputs)
    spline = torch.einsum("bip,oip->bo", basis, layer.spline_weight)
    base = torch.nn.functional.silu(inputs) @ layer.base_weight.T
    assert torch.allclose(layer(inputs), base + spline, atol=1e-6)


def test_edge_functions_are_consistent_with_the_forward_pass() -> None:
    torch.manual_seed(0)
    layer = KANLayer(2, 3, grid_size=5, spline_order=3)
    samples = torch.linspace(-0.9, 0.9, 11)
    edges = layer.edge_functions(samples)
    assert edges.shape == (3, 2, 11)

    # Feeding the same value into every input reduces the layer to the sum of edges.
    repeated = samples.unsqueeze(-1).expand(-1, 2)
    assert torch.allclose(layer(repeated), edges.sum(dim=1).T, atol=1e-5)


def test_layer_validates_input_shape() -> None:
    layer = KANLayer(2, 2)
    with pytest.raises(ValueError, match="expected shape"):
        layer(torch.zeros(4))
    with pytest.raises(ValueError, match="input features"):
        layer(torch.zeros(4, 5))


def test_layer_validates_construction() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        KANLayer(0, 2)
    with pytest.raises(ValueError, match="grid_size"):
        KANLayer(2, 2, grid_size=0)
    with pytest.raises(ValueError, match="spline_order"):
        KANLayer(2, 2, spline_order=0)
    with pytest.raises(ValueError, match="spline_noise_scale"):
        KANLayer(2, 2, spline_noise_scale=-1.0)


def test_initialization_is_deterministic_under_a_fixed_seed() -> None:
    torch.manual_seed(7)
    first = KANLayer(3, 2)
    torch.manual_seed(7)
    second = KANLayer(3, 2)
    assert torch.equal(first.base_weight, second.base_weight)
    assert torch.equal(first.spline_weight, second.spline_weight)

    torch.manual_seed(8)
    third = KANLayer(3, 2)
    assert not torch.equal(first.spline_weight, third.spline_weight)


def test_gradients_are_finite_and_reach_both_branches() -> None:
    torch.manual_seed(0)
    layer = KANLayer(2, 2)
    output = layer(torch.rand(16, 2) * 1.6 - 0.8)
    output.pow(2).mean().backward()

    assert layer.base_weight.grad is not None
    assert layer.spline_weight.grad is not None
    assert torch.isfinite(layer.base_weight.grad).all()
    assert torch.isfinite(layer.spline_weight.grad).all()
    assert layer.spline_weight.grad.abs().sum() > 0


def test_frozen_edges_ablation_stops_spline_gradients() -> None:
    layer = KANLayer(2, 2, learnable_spline=False)
    layer(torch.rand(8, 2) * 1.6 - 0.8).sum().backward()
    assert layer.spline_weight.grad is None
    assert layer.base_weight.grad is not None


def test_no_base_branch_ablation_removes_the_residual_path() -> None:
    layer = KANLayer(2, 2, use_base_branch=False)
    assert torch.count_nonzero(layer.base_weight) == 0
    inputs = torch.rand(8, 2) * 1.6 - 0.8
    spline_only = torch.einsum("bip,oip->bo", layer.basis(inputs), layer.spline_weight)
    assert torch.allclose(layer(inputs), spline_only, atol=1e-6)


def test_grid_update_moves_knots_but_preserves_the_function() -> None:
    torch.manual_seed(0)
    layer = KANLayer(1, 2, grid_size=8, spline_order=3)
    inputs = torch.randn(512, 1) * 0.3  # concentrated well inside the initial domain

    before_grid = layer.grid.clone()
    before_output = layer(inputs).clone()
    layer.update_grid(inputs)

    assert not torch.allclose(before_grid, layer.grid)
    # The refit is a projection onto a *different* spline space, so it is preserving up
    # to least-squares residual, not exactly. The residual must stay small relative to
    # the signal; see "Known approximations" in the track README.
    drift = (before_output - layer(inputs)).detach().abs().max()
    assert float(drift) < 0.05 * float(before_output.detach().abs().max())


def test_grid_update_validates_input_shape() -> None:
    layer = KANLayer(2, 2)
    with pytest.raises(ValueError, match="expected shape"):
        layer.update_grid(torch.zeros(4, 3))


def test_regularization_is_non_negative_and_responds_to_coefficients() -> None:
    layer = KANLayer(2, 2)
    with torch.no_grad():
        layer.spline_weight.zero_()
    assert float(layer.regularization().detach()) == pytest.approx(0.0)

    with torch.no_grad():
        layer.spline_weight.fill_(2.0)
    assert float(layer.regularization().detach()) == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# Network, baseline, and budget matching
# --------------------------------------------------------------------------------------


def test_network_shapes_and_serialization_round_trip() -> None:
    torch.manual_seed(0)
    config = KANConfig(layer_widths=(2, 5, 1))
    model = KAN(config)
    inputs = torch.rand(9, 2) * 1.6 - 0.8
    expected = model(inputs)
    assert expected.shape == (9, 1)

    restored = KAN(config)
    restored.load_state_dict(model.state_dict())
    assert torch.allclose(restored(inputs), expected, atol=1e-7)


def test_network_grid_update_preserves_outputs() -> None:
    torch.manual_seed(0)
    model = KAN(KANConfig(layer_widths=(2, 4, 1), grid_size=8))
    inputs = torch.randn(400, 2) * 0.3
    before = model(inputs).clone()
    model.update_grids(inputs)
    assert torch.allclose(before, model(inputs), atol=5e-3)


def test_network_can_fit_a_univariate_analytic_function() -> None:
    torch.manual_seed(0)
    model = KAN(KANConfig(layer_widths=(1, 1), grid_size=12))
    inputs = interior_points(256)
    targets = torch.sin(2.0 * math.pi * inputs)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    initial = float(torch.mean((model(inputs) - targets) ** 2).detach())
    for _ in range(300):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(inputs) - targets) ** 2)
        loss.backward()
        optimizer.step()

    assert float(loss) < initial * 0.01
    assert float(loss) < 1e-3


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="at least an input"):
        KANConfig(layer_widths=(3,))
    with pytest.raises(ValueError, match="width must be positive"):
        KANConfig(layer_widths=(2, 0, 1))
    with pytest.raises(ValueError, match="grid_size"):
        KANConfig(layer_widths=(2, 1), grid_size=0)
    with pytest.raises(ValueError, match="spline_order"):
        KANConfig(layer_widths=(2, 1), spline_order=0)
    with pytest.raises(ValueError, match="grid_range"):
        KANConfig(layer_widths=(2, 1), grid_range=(1.0, 1.0))
    with pytest.raises(ValueError, match="regularization_weight"):
        KANConfig(layer_widths=(2, 1), regularization_weight=-1.0)


def test_config_reports_parameters_per_edge() -> None:
    assert KANConfig(layer_widths=(2, 1), grid_size=5, spline_order=3).parameters_per_edge == 9
    assert (
        KANConfig(
            layer_widths=(2, 1), grid_size=5, spline_order=3, use_base_branch=False
        ).parameters_per_edge
        == 8
    )


def test_matched_budget_mlp_is_close_to_the_kan_parameter_count() -> None:
    kan = KAN(KANConfig(layer_widths=(2, 5, 1)))
    target = count_parameters(kan)
    hidden = match_parameter_budget(target, 2, 1, depth=1)
    baseline = MLP(2, 1, hidden_widths=hidden)
    assert abs(count_parameters(baseline) - target) <= target * 0.05


def test_matched_budget_validates_arguments() -> None:
    with pytest.raises(ValueError, match="depth"):
        match_parameter_budget(100, 2, 1, depth=-1)
    with pytest.raises(ValueError, match="max_width"):
        match_parameter_budget(100, 2, 1, max_width=0)
    assert match_parameter_budget(100, 2, 1, depth=0) == []


def test_mlp_validates_widths() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        MLP(0, 1, hidden_widths=[4])
    with pytest.raises(ValueError, match="hidden widths"):
        MLP(2, 1, hidden_widths=[0])
