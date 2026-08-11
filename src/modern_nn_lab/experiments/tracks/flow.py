"""Experiment suite for Track 10 — flow matching.

The acceptance criterion is the organizing principle: **separate vector-field approximation
error from ODE discretization error**. Sample quality alone cannot do that, because a poor
sample set is equally consistent with a badly learned field and with a solver that took too
few steps.

The Gaussian target resolves it, because there the marginal velocity field is known in
closed form:

- **discretization error alone** — integrate the *exact* field at each step count. The model
  is not involved, so whatever error remains is the solver's.
- **approximation error alone** — compare the *learned* field to the exact one pointwise,
  over the marginal the model will actually be asked about. No solver is involved, so
  whatever error remains is the network's.
- **both together** — integrate the learned field. If the separation is honest, this should
  behave like the discretization curve until it flattens onto the approximation floor.

The two curved targets carry no closed form and are reported on sample quality alone, which
is stated rather than glossed: for those, the two error sources are *not* separated here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from modern_nn_lab.experiments.external import run_meta_evaluation
from modern_nn_lab.experiments.records import ARTEFACT_DIRNAME, RESULT_SCHEMA_VERSION, fingerprint
from modern_nn_lab.experiments.runner import RunGroup, RunSpec
from modern_nn_lab.reproducibility import seed_everything
from modern_nn_lab.tracks.flow import (
    DATASETS,
    MIXTURE_CENTRES,
    Dataset,
    FlowConfig,
    FlowExperimentConfig,
    GaussianEndpoints,
    ProbabilityPath,
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
from modern_nn_lab.tracks.flow.solver import EVALUATIONS_PER_STEP, Method, VelocityFn

TRACK = "flow"
ARCHITECTURE_VERSION = "0.1.0"


def train_field(
    dataset: Dataset,
    path: ProbabilityPath,
    settings: FlowExperimentConfig,
    *,
    seed: int,
) -> tuple[VectorField, list[float]]:
    """Fit a velocity field by conditional flow matching.

    Args:
        dataset: Which target distribution.
        path: The probability path defining the conditional target.
        settings: Track settings.
        seed: Seed for initialization and sampling.

    Returns:
        ``(field, loss_trajectory)``.
    """

    seed_everything(seed)
    field = VectorField(FlowConfig())
    optimizer = torch.optim.Adam(field.parameters(), lr=settings.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)

    losses: list[float] = []
    field.train()
    for _ in range(settings.steps):
        source = sample_source(settings.batch_size, 2, generator)
        target = sample_target(
            dataset,
            settings.batch_size,
            generator,
            mean=settings.gaussian_mean,
            scale=settings.gaussian_scale,
        )
        loss = flow_matching_loss(field, path, source, target, generator)
        optimizer.zero_grad(set_to_none=True)
        # Torch ships `Tensor.backward` unannotated; the call is what strict mypy flags.
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.detach()))

    return field, losses


def as_velocity(field: VectorField) -> VelocityFn:
    """Wrap a trained field as a plain velocity function for the solver.

    Args:
        field: The trained network.

    Returns:
        A callable suitable for :func:`~modern_nn_lab.tracks.flow.solver.integrate`.
    """

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            predicted: torch.Tensor = field(x, t)
        return predicted

    return velocity


def generate_samples(
    field: VectorField,
    n: int,
    *,
    steps: int,
    method: Method,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample from a trained flow by integrating from the source.

    Args:
        field: The trained network.
        n: Number of samples.
        steps: Solver steps.
        method: Solver method.
        generator: Seeded generator for the source draw.

    Returns:
        Shape ``(n, 2)`` samples.
    """

    field.eval()
    initial = sample_source(n, 2, generator)
    trajectory = integrate(as_velocity(field), initial, steps=steps, method=method)
    return trajectory.final


def evaluate_dataset(
    dataset: Dataset, path_name: str, settings: FlowExperimentConfig, output_dir: Path
) -> RunGroup:
    """Train and score one (dataset, path) pair across seeds.

    Args:
        dataset: Which target distribution.
        path_name: Which probability path.
        settings: Track settings.
        output_dir: Directory receiving records.

    Returns:
        A :class:`RunGroup`.
    """

    identity = fingerprint(
        {
            "dataset": dataset,
            "path": path_name,
            "n_eval": settings.n_eval,
            "steps": settings.steps,
            "gaussian_mean": settings.gaussian_mean,
            "gaussian_scale": settings.gaussian_scale,
        }
    )
    shared: dict[str, Any] = {
        "dataset": dataset,
        "path": path_name,
        "solver": settings.eval_method,
        "solver_steps": settings.solver_steps[-1],
        "training_steps": settings.steps,
        **FlowConfig().as_dict(),
    }

    def evaluate(seed: int) -> tuple[float, dict[str, float], dict[str, Any]]:
        path = build_path(path_name)
        field, losses = train_field(dataset, path, settings, seed=seed)

        generator = torch.Generator().manual_seed(10_000 + seed)
        reference = sample_target(
            dataset,
            settings.n_eval,
            generator,
            mean=settings.gaussian_mean,
            scale=settings.gaussian_scale,
        )
        samples = generate_samples(
            field,
            settings.n_eval,
            steps=settings.solver_steps[-1],
            method=settings.eval_method,
            generator=generator,
        )

        metrics: dict[str, float] = {
            "final_training_loss": losses[-1],
            # A floor for the metric: two independent draws from the target itself. An
            # energy distance near this value means the samples are as close as sampling
            # noise allows, and reporting the distance without it would suggest otherwise.
            "energy_distance_floor": energy_distance(
                reference,
                sample_target(
                    dataset,
                    settings.n_eval,
                    generator,
                    mean=settings.gaussian_mean,
                    scale=settings.gaussian_scale,
                ),
            ),
        }
        if dataset == "mixture":
            metrics["mode_coverage"] = mode_coverage(samples, MIXTURE_CENTRES)

        return energy_distance(samples, reference), metrics, dict(shared)

    return run_meta_evaluation(
        evaluate,
        spec=RunSpec(
            track=TRACK,
            architecture=path_name,
            metric_name="energy_distance",
            higher_is_better=False,
            variant=dataset,
            architecture_version=ARCHITECTURE_VERSION,
            extra_config=shared,
            notes=(
                "Energy distance against a held-out draw from the target, with the "
                "target-versus-target floor recorded beside it."
            ),
        ),
        dataset=dataset,
        split_strategy=(
            f"{settings.n_eval} samples generated by integrating from the source with "
            f"{settings.solver_steps[-1]} {settings.eval_method} steps, scored against an "
            "independent draw from the target"
        ),
        output_dir=output_dir,
        seeds=settings.seeds,
        dataset_fingerprint=identity,
        parameter_count=sum(p.numel() for p in VectorField(FlowConfig()).parameters()),
    )


def separate_error_sources(settings: FlowExperimentConfig) -> dict[str, Any]:
    """Measure discretization and approximation error independently.

    This is the acceptance criterion, and it is only possible because the Gaussian case has
    a closed-form marginal field.

    Args:
        settings: Track settings.

    Returns:
        A JSON-serializable summary.
    """

    endpoints = GaussianEndpoints(mean=settings.gaussian_mean, scale=settings.gaussian_scale)
    generator = torch.Generator().manual_seed(settings.data_seed)
    reference = endpoints.sample_target(settings.n_eval, 2, generator)
    floor = energy_distance(reference, endpoints.sample_target(settings.n_eval, 2, generator))

    rows: list[dict[str, Any]] = []
    for path_name in settings.paths:
        path = build_path(path_name)
        field, _ = train_field("gaussian", path, settings, seed=0)

        # (a) Approximation error alone: no solver involved. Compare the learned field to
        # the exact one over the marginal the model is actually asked about.
        probe_generator = torch.Generator().manual_seed(4242)
        probe_source = sample_source(8192, 2, probe_generator)
        probe_target = endpoints.sample_target(8192, 2, probe_generator)
        probe_times = torch.rand((8192, 1), generator=probe_generator)
        probe_points = path.interpolate(probe_source, probe_target, probe_times)
        with torch.no_grad():
            learned = field(probe_points, probe_times)
        exact = marginal_velocity(probe_points, probe_times, path, endpoints)
        field_error = float((learned - exact).pow(2).sum(dim=-1).mean().sqrt())
        field_scale = float(exact.pow(2).sum(dim=-1).mean().sqrt())

        methods: tuple[Method, ...] = ("euler", "midpoint")
        for method in methods:
            for steps in settings.solver_steps:
                sample_generator = torch.Generator().manual_seed(777)
                initial = sample_source(settings.n_eval, 2, sample_generator)

                # (b) Discretization error alone: the exact field, so the model cannot
                # contribute anything.
                def exact_field(
                    x: torch.Tensor, t: torch.Tensor, chosen: ProbabilityPath = path
                ) -> torch.Tensor:
                    return marginal_velocity(x, t, chosen, endpoints)

                exact_run = integrate(exact_field, initial, steps=steps, method=method)
                # (c) Both together.
                learned_run = integrate(
                    as_velocity(field),
                    initial,
                    steps=steps,
                    method=method,
                )

                rows.append(
                    {
                        "path": path_name,
                        "method": method,
                        "steps": steps,
                        "n_evaluations": steps * EVALUATIONS_PER_STEP[method],
                        "discretization_only": energy_distance(exact_run.final, reference),
                        "combined": energy_distance(learned_run.final, reference),
                        "field_rmse": field_error,
                        "field_relative_rmse": field_error / max(field_scale, 1e-9),
                    }
                )

    return {
        "note": (
            "`discretization_only` integrates the closed-form marginal field, so the model "
            "contributes nothing and the residual is the solver's. `field_rmse` compares "
            "the learned field to the exact one pointwise with no solver involved, so it is "
            "the network's alone. `combined` integrates the learned field. The floor is the "
            "energy distance between two independent draws from the target, which no method "
            "can beat."
        ),
        "energy_distance_floor": floor,
        "rows": rows,
    }


def write_diagnostics(settings: FlowExperimentConfig, output_dir: Path) -> Path:
    """Serialize the error separation, path checks, and saved trajectories.

    Args:
        settings: Track settings.
        output_dir: Directory receiving the artefact.

    Returns:
        Path of the written file.
    """

    endpoints = GaussianEndpoints(mean=settings.gaussian_mean, scale=settings.gaussian_scale)
    generator = torch.Generator().manual_seed(settings.data_seed)

    path_checks = []
    for path_name in settings.paths:
        path = build_path(path_name)
        residual = projection_residual(path, endpoints, n_samples=200_000, generator=generator)
        start, end = path.endpoints()
        path_checks.append(
            {
                "path": path_name,
                "alpha_sigma_at_0": list(start),
                "alpha_sigma_at_1": list(end),
                "projection_residual": residual,
            }
        )

    # A handful of saved trajectories, for the diagnostic visualization the prompt asks for.
    trajectories: dict[str, Any] = {}
    for dataset in DATASETS:
        path = build_path(settings.paths[0])
        field, _ = train_field(dataset, path, settings, seed=0)
        sample_generator = torch.Generator().manual_seed(99)
        initial = sample_source(64, 2, sample_generator)
        run = integrate(
            as_velocity(field),
            initial,
            steps=32,
            method="midpoint",
            save_trajectory=True,
        )
        trajectories[dataset] = {
            "times": run.times.tolist(),
            "states": run.states.tolist(),
            "path": settings.paths[0],
        }

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "track": TRACK,
        "measurement": "flow_diagnostics",
        "path_checks": path_checks,
        "error_separation": separate_error_sources(settings),
        "trajectories": trajectories,
    }

    artefacts = output_dir / ARTEFACT_DIRNAME
    artefacts.mkdir(parents=True, exist_ok=True)
    destination = artefacts / "flow_diagnostics.json"
    destination.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return destination


def run(output_dir: Path | str | None = None, *, quick: bool = False) -> None:
    """Run the whole flow-matching suite.

    Args:
        output_dir: Destination directory. Defaults to ``results/flow``.
        quick: Reduced seeds, training steps, and evaluation samples.
    """

    destination = Path(output_dir) if output_dir is not None else Path("results") / TRACK
    settings = (
        FlowExperimentConfig(
            seeds=(0, 1),
            steps=300,
            n_eval=512,
            solver_steps=(2, 8, 32),
        )
        if quick
        else FlowExperimentConfig()
    )

    for dataset in DATASETS:
        for path_name in settings.paths:
            evaluate_dataset(dataset, path_name, settings, destination)

    write_diagnostics(settings, destination)
