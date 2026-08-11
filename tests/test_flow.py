"""Invariants of the flow-matching track.

The prompt names four mandatory properties: the conditional vector field is correct on
analytic toy paths, ODE shapes and dtypes are right, endpoints and reversibility hold under
small numerical error, and sampling is deterministic under a fixed seed. Each has a test
here.

Two of them are checked more strongly than the prompt requires, because the weaker version
would pass on broken code. The solver's *convergence order* is measured rather than its
mere accuracy, so a midpoint method that had silently degraded to first order would fail.
And the closed-form marginal field is verified two independent ways — against the defining
orthogonality property of a conditional expectation, and by integrating it to confirm it
actually transports the source onto the target.
"""

from __future__ import annotations

import math

import pytest
import torch

from modern_nn_lab.tracks.flow import (
    DATASETS,
    MIXTURE_CENTRES,
    FlowConfig,
    GaussianEndpoints,
    LinearPath,
    TrigonometricPath,
    VectorField,
    build_path,
    energy_distance,
    flow_matching_loss,
    integrate,
    marginal_velocity,
    mode_coverage,
    projection_residual,
    sample_source,
    sample_target,
)

# --------------------------------------------------------------------------------------
# Mandatory test 1: the conditional vector field on analytic toy paths
# --------------------------------------------------------------------------------------


def test_linear_path_has_a_constant_conditional_velocity() -> None:
    """For the straight-line path the target is exactly ``x1 - x0``, at every time."""

    path = LinearPath()
    source = torch.randn(16, 2)
    target = torch.randn(16, 2)

    for t_value in (0.0, 0.25, 0.5, 0.75, 1.0):
        times = torch.full((16, 1), t_value)
        velocity = path.conditional_velocity(source, target, times)
        assert torch.allclose(velocity, target - source, atol=1e-6)


def test_trigonometric_path_matches_its_derivative_by_hand() -> None:
    """The quarter-circle path's velocity, written out independently of the code."""

    path = TrigonometricPath()
    source = torch.randn(8, 2)
    target = torch.randn(8, 2)
    times = torch.rand(8, 1)

    expected = (
        0.5
        * math.pi
        * (torch.cos(0.5 * math.pi * times) * target - torch.sin(0.5 * math.pi * times) * source)
    )
    assert torch.allclose(path.conditional_velocity(source, target, times), expected, atol=1e-6)


@pytest.mark.parametrize("name", ["linear", "trigonometric"])
def test_the_conditional_velocity_is_the_derivative_of_the_interpolant(name: str) -> None:
    """The defining relation, checked numerically rather than trusted."""

    path = build_path(name)
    # Double precision: a central difference divides an O(1) subtraction by 2e-4, which
    # amplifies float32 round-off to about 1e-3 — larger than the agreement being checked.
    # The property under test is mathematical, so the arithmetic should not be the limit.
    source = torch.randn(32, 2, dtype=torch.float64)
    target = torch.randn(32, 2, dtype=torch.float64)
    times = 0.2 + 0.6 * torch.rand(32, 1, dtype=torch.float64)

    step = 1e-6
    forward = path.interpolate(source, target, times + step)
    backward = path.interpolate(source, target, times - step)
    numerical = (forward - backward) / (2 * step)

    analytic = path.conditional_velocity(source, target, times)
    assert torch.allclose(numerical, analytic, atol=1e-7)


@pytest.mark.parametrize("name", ["linear", "trigonometric"])
def test_every_path_starts_at_the_source_and_ends_at_the_target(name: str) -> None:
    """A path that failed this would be interpolating between the wrong things."""

    path = build_path(name)
    (alpha_0, sigma_0), (alpha_1, sigma_1) = path.endpoints()

    assert alpha_0 == pytest.approx(0.0, abs=1e-6)
    assert sigma_0 == pytest.approx(1.0, abs=1e-6)
    assert alpha_1 == pytest.approx(1.0, abs=1e-6)
    assert sigma_1 == pytest.approx(0.0, abs=1e-6)

    source = torch.randn(8, 2)
    target = torch.randn(8, 2)
    assert torch.allclose(path.interpolate(source, target, torch.zeros(8, 1)), source, atol=1e-6)
    assert torch.allclose(path.interpolate(source, target, torch.ones(8, 1)), target, atol=1e-6)


def test_the_trigonometric_path_preserves_variance_for_a_standard_normal_target() -> None:
    """The property the path is named for, stated precisely and checked."""

    path = TrigonometricPath()
    generator = torch.Generator().manual_seed(0)
    source = torch.randn((40000, 1), generator=generator)
    target = torch.randn((40000, 1), generator=generator)

    for t_value in (0.0, 0.3, 0.6, 1.0):
        interpolated = path.interpolate(source, target, torch.full((40000, 1), t_value))
        assert float(interpolated.var()) == pytest.approx(1.0, abs=0.05)


def test_path_registry_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown path"):
        build_path("nope")


def test_paths_reject_badly_shaped_times() -> None:
    path = LinearPath()
    with pytest.raises(ValueError, match="expected times"):
        path.interpolate(torch.randn(4, 2), torch.randn(4, 2), torch.rand(4, 2, 1))


# --------------------------------------------------------------------------------------
# The closed-form marginal field, verified two independent ways
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["linear", "trigonometric"])
def test_the_marginal_field_satisfies_the_conditional_expectation_property(name: str) -> None:
    """A conditional expectation's residual is orthogonal to every function of its argument."""

    generator = torch.Generator().manual_seed(0)
    residual = projection_residual(
        build_path(name),
        GaussianEndpoints(mean=2.0, scale=0.5),
        n_samples=200_000,
        generator=generator,
    )
    for statistic in residual.values():
        assert abs(statistic) < 0.05


def test_the_orthogonality_check_rejects_a_wrong_field() -> None:
    """Otherwise the test above would pass on anything."""

    from modern_nn_lab.tracks.flow import analytic

    generator = torch.Generator().manual_seed(0)
    endpoints = GaussianEndpoints(mean=2.0, scale=0.5)
    original = analytic.marginal_velocity
    try:
        analytic.marginal_velocity = lambda x, t, p, e: original(x, t, p, e) * 1.10
        residual = analytic.projection_residual(
            build_path("linear"), endpoints, n_samples=200_000, generator=generator
        )
    finally:
        analytic.marginal_velocity = original

    assert max(abs(value) for value in residual.values()) > 0.05


@pytest.mark.parametrize("name", ["linear", "trigonometric"])
def test_integrating_the_exact_field_transports_source_onto_target(name: str) -> None:
    """The decisive check: the field is only right if it moves the distribution correctly."""

    path = build_path(name)
    endpoints = GaussianEndpoints(mean=2.0, scale=0.5)
    generator = torch.Generator().manual_seed(0)
    initial = torch.randn((20000, 2), generator=generator)

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return marginal_velocity(x, t, path, endpoints)

    final = integrate(field, initial, steps=800, method="midpoint").final
    assert torch.allclose(final.mean(dim=0), torch.full((2,), 2.0), atol=0.05)
    assert torch.allclose(final.std(dim=0), torch.full((2,), 0.5), atol=0.05)


def test_gaussian_endpoints_validate_their_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        GaussianEndpoints(mean=0.0, scale=0.0)


# --------------------------------------------------------------------------------------
# Mandatory test 2: ODE shape and dtype correctness
# --------------------------------------------------------------------------------------


def test_integration_preserves_shape_and_dtype() -> None:
    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x

    for dtype in (torch.float32, torch.float64):
        initial = torch.randn(7, 3, dtype=dtype)
        run = integrate(field, initial, steps=5, method="euler")
        assert run.final.shape == initial.shape
        assert run.final.dtype == dtype


def test_saved_trajectories_have_one_state_more_than_steps() -> None:
    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    initial = torch.zeros(4, 2)
    run = integrate(field, initial, steps=6, method="euler", save_trajectory=True)

    assert run.states.shape == (7, 4, 2)
    assert run.times.shape == (7,)
    assert torch.allclose(run.states[0], initial)
    assert torch.allclose(run.states[-1], run.final)
    assert float(run.times[0]) == 0.0
    assert float(run.times[-1]) == pytest.approx(1.0)


def test_field_evaluations_are_counted_per_method() -> None:
    """Comparing solvers at equal steps would flatter the more expensive one."""

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x

    initial = torch.randn(3, 2)
    assert integrate(field, initial, steps=10, method="euler").n_evaluations == 10
    assert integrate(field, initial, steps=10, method="midpoint").n_evaluations == 20


def test_the_field_receives_a_column_of_times() -> None:
    """A solver passing a scalar or a wrongly shaped time would break broadcasting."""

    seen: list[tuple[int, ...]] = []

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        seen.append(tuple(t.shape))
        return torch.zeros_like(x)

    integrate(field, torch.zeros(5, 2), steps=3, method="euler")
    assert seen == [(5, 1)] * 3


def test_integrate_validates_its_arguments() -> None:
    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x

    with pytest.raises(ValueError, match="steps must be positive"):
        integrate(field, torch.zeros(2, 2), steps=0)
    with pytest.raises(ValueError, match="unknown method"):
        integrate(field, torch.zeros(2, 2), steps=2, method="rk4")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Mandatory test 3: endpoints, reversibility, and convergence order
# --------------------------------------------------------------------------------------


def test_solvers_reproduce_a_known_exact_solution() -> None:
    """``dx/dt = -x`` from ``x(0) = 1`` has the exact solution ``e^{-1}`` at ``t = 1``."""

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x

    exact = math.exp(-1.0)
    for method, tolerance in (("euler", 2e-3), ("midpoint", 1e-5)):
        final = integrate(field, torch.ones(1, 1), steps=1000, method=method).final  # type: ignore[arg-type]
        assert float(final) == pytest.approx(exact, abs=tolerance)


@pytest.mark.parametrize(("method", "expected_order"), [("euler", 1.0), ("midpoint", 2.0)])
def test_solver_convergence_order_is_what_it_claims(method: str, expected_order: float) -> None:
    """Measured, not asserted: a silently first-order midpoint method must fail this."""

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x

    exact = math.exp(-1.0)
    errors = []
    for steps in (16, 32, 64, 128):
        final = integrate(field, torch.ones(1, 1), steps=steps, method=method).final  # type: ignore[arg-type]
        errors.append(abs(float(final) - exact))

    # Halving the step size should divide the error by 2^order.
    orders = [math.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    measured = sum(orders) / len(orders)
    assert measured == pytest.approx(expected_order, abs=0.15)


def test_integration_is_reversible_under_a_small_step_size() -> None:
    """Forward then backward must return where it started, up to discretization error."""

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + t

    initial = torch.randn(16, 2)
    forward = integrate(field, initial, steps=400, method="midpoint", t_start=0.0, t_end=1.0)
    back = integrate(field, forward.final, steps=400, method="midpoint", t_start=1.0, t_end=0.0)

    assert torch.allclose(back.final, initial, atol=1e-4)


def test_a_coarse_round_trip_does_not_return_exactly() -> None:
    """Otherwise the reversibility test could be passing for a trivial reason."""

    def field(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) + t

    initial = torch.randn(16, 2)
    forward = integrate(field, initial, steps=2, method="euler", t_start=0.0, t_end=1.0)
    back = integrate(field, forward.final, steps=2, method="euler", t_start=1.0, t_end=0.0)

    assert not torch.allclose(back.final, initial, atol=1e-4)


# --------------------------------------------------------------------------------------
# Mandatory test 4: deterministic sampling under a fixed seed
# --------------------------------------------------------------------------------------


def test_sampling_is_deterministic_under_a_fixed_seed() -> None:
    torch.manual_seed(0)
    field = VectorField(FlowConfig()).eval()

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return field(x, t)

    first = integrate(
        velocity, sample_source(64, 2, torch.Generator().manual_seed(7)), steps=16
    ).final
    second = integrate(
        velocity, sample_source(64, 2, torch.Generator().manual_seed(7)), steps=16
    ).final

    assert torch.equal(first, second)


def test_a_different_seed_gives_different_samples() -> None:
    torch.manual_seed(0)
    field = VectorField(FlowConfig()).eval()

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return field(x, t)

    first = integrate(
        velocity, sample_source(64, 2, torch.Generator().manual_seed(7)), steps=16
    ).final
    other = integrate(
        velocity, sample_source(64, 2, torch.Generator().manual_seed(8)), steps=16
    ).final

    assert not torch.allclose(first, other)


def test_data_generation_is_deterministic_under_a_fixed_seed() -> None:
    for dataset in DATASETS:
        first = sample_target(dataset, 128, torch.Generator().manual_seed(3))
        second = sample_target(dataset, 128, torch.Generator().manual_seed(3))
        assert torch.equal(first, second)


# --------------------------------------------------------------------------------------
# The network and its objective
# --------------------------------------------------------------------------------------


def test_the_field_network_produces_a_velocity_per_point() -> None:
    field = VectorField(FlowConfig(dim=2))
    assert field(torch.randn(9, 2), torch.rand(9, 1)).shape == (9, 2)
    assert field(torch.randn(9, 2), torch.rand(9)).shape == (9, 2)


def test_the_field_network_actually_depends_on_time() -> None:
    """Without this, the Fourier embedding could be dead and nothing would say so."""

    torch.manual_seed(0)
    field = VectorField(FlowConfig()).eval()
    points = torch.randn(32, 2)
    with torch.no_grad():
        early = field(points, torch.zeros(32, 1))
        late = field(points, torch.ones(32, 1))
    assert not torch.allclose(early, late, atol=1e-4)


def test_the_field_network_validates_shapes() -> None:
    field = VectorField(FlowConfig(dim=2))
    with pytest.raises(ValueError, match="dimensions"):
        field(torch.randn(4, 5), torch.rand(4, 1))
    with pytest.raises(ValueError, match="batch size"):
        field(torch.randn(4, 2), torch.rand(3, 1))


def test_the_loss_is_zero_when_the_field_predicts_the_conditional_target() -> None:
    """A sanity check on the objective itself, using a field that cheats perfectly."""

    path = LinearPath()
    source = torch.randn(64, 2)
    target = torch.randn(64, 2)

    class Perfect(VectorField):
        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return target - source

    perfect = Perfect(FlowConfig())
    loss = flow_matching_loss(perfect, path, source, target, torch.Generator().manual_seed(0))
    assert float(loss) == pytest.approx(0.0, abs=1e-10)


def test_training_reduces_the_flow_matching_loss() -> None:
    from modern_nn_lab.experiments.tracks.flow import train_field
    from modern_nn_lab.tracks.flow import FlowExperimentConfig

    settings = FlowExperimentConfig(steps=200, batch_size=128)
    _, losses = train_field("gaussian", LinearPath(), settings, seed=0)
    assert sum(losses[-20:]) / 20 < sum(losses[:20]) / 20


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        FlowConfig(d_hidden=0)
    with pytest.raises(ValueError, match="positive"):
        FlowConfig(n_frequencies=0)


# --------------------------------------------------------------------------------------
# The evaluation metric
# --------------------------------------------------------------------------------------


def test_energy_distance_is_near_zero_for_two_draws_of_the_same_distribution() -> None:
    generator = torch.Generator().manual_seed(0)
    first = torch.randn((2000, 2), generator=generator)
    second = torch.randn((2000, 2), generator=generator)
    assert energy_distance(first, second) < 0.02


def test_energy_distance_grows_with_separation() -> None:
    generator = torch.Generator().manual_seed(0)
    reference = torch.randn((2000, 2), generator=generator)

    previous = -1.0
    for shift in (0.0, 0.5, 1.0, 2.0):
        distance = energy_distance(reference + shift, reference)
        assert distance > previous
        previous = distance


def test_energy_distance_is_symmetric_and_non_negative() -> None:
    generator = torch.Generator().manual_seed(0)
    first = torch.randn((500, 2), generator=generator)
    second = 1.5 + torch.randn((500, 2), generator=generator)

    assert energy_distance(first, second) == pytest.approx(energy_distance(second, first), abs=1e-5)
    assert energy_distance(first, first) >= 0.0


def test_mode_coverage_detects_a_dropped_mode() -> None:
    all_modes = MIXTURE_CENTRES.repeat(10, 1)
    assert mode_coverage(all_modes, MIXTURE_CENTRES) == pytest.approx(1.0)

    three_modes = MIXTURE_CENTRES[:3].repeat(10, 1)
    assert mode_coverage(three_modes, MIXTURE_CENTRES) == pytest.approx(0.75)


def test_datasets_have_the_documented_shapes_and_are_distinct() -> None:
    generator = torch.Generator().manual_seed(0)
    samples = {name: sample_target(name, 500, generator) for name in DATASETS}
    for name, points in samples.items():
        assert points.shape == (500, 2), name
        assert torch.isfinite(points).all(), name

    assert energy_distance(samples["moons"], samples["mixture"]) > 0.1


def test_unknown_dataset_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        sample_target("spiral", 4, torch.Generator())  # type: ignore[arg-type]
