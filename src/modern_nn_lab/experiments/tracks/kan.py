r"""Experiment suite for Track 01 — Kolmogorov-Arnold Networks.

Protocol, in the order the repository requires of every track:

1. **Diagnostics that isolate the mechanism** — univariate functions with a known
   analytic form, then a multivariate compositional function.
2. **Matched-parameter baseline** — an MLP whose hidden width is searched so its
   parameter count is as close as possible to the KAN's. A KAN edge stores ``G + k + 1``
   numbers where a linear edge stores one, so equal *widths* would be a threefold
   capacity advantage disguised as an architecture result.
3. **Ablations** — frozen edge functions (spline coefficients fixed at initialization)
   and a pure-spline variant with the residual ``SiLU`` branch removed.
4. **Sensitivity** — grid size, spline order, and regularization weight.
5. **Real benchmark** — a public tabular regression dataset, compared against a tree
   ensemble as well as an MLP, because framing KAN against neural baselines only would
   overstate its standing on tabular data.
6. **Interpretability artefact** — learned edge functions sampled and serialized, so
   figures come from saved results rather than from a live model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor, nn

from modern_nn_lab.experiments.data import SupervisedSplit, make_function_split, make_tabular_split
from modern_nn_lab.experiments.evaluation import mean_squared_error
from modern_nn_lab.experiments.external import run_external_baseline
from modern_nn_lab.experiments.records import ARTEFACT_DIRNAME, RESULT_SCHEMA_VERSION
from modern_nn_lab.experiments.runner import RunGroup, RunSpec, run_seeded_experiment
from modern_nn_lab.experiments.training import TrainingConfig, train_supervised
from modern_nn_lab.reproducibility import seed_everything
from modern_nn_lab.tracks.kan import (
    KAN,
    MLP,
    KANConfig,
    KANExperimentConfig,
    count_parameters,
    match_parameter_budget,
)

TRACK = "kan"
ARCHITECTURE_VERSION = "0.1.0"

DEFAULT_GRID_SIZE = 5
"""Grid intervals used by every run that is not part of the grid-size sweep."""

DEFAULT_SPLINE_ORDER = 3
"""Cubic splines, the setting used throughout the primary source's examples."""


def univariate_target(inputs: Tensor) -> Tensor:
    r"""Analytic univariate target :math:`f(x) = \sin(2\pi x)`.

    Chosen because the true function is exactly representable in the limit of a fine
    spline grid, so approximation error is attributable to capacity, not to noise.

    Args:
        inputs: Shape ``(N, 1)`` in ``[-1, 1]``.

    Returns:
        Shape ``(N, 1)`` targets.
    """

    return torch.sin(2.0 * math.pi * inputs[:, :1])


def compositional_target(inputs: Tensor) -> Tensor:
    r"""Compositional target :math:`f(x_1, x_2) = \exp(\sin(\pi x_1) + x_2^2)`.

    A composition of univariate functions with an outer nonlinearity: exactly the
    structural assumption a KAN encodes, and therefore the case where the mechanism
    should show an advantage if it has one.

    Args:
        inputs: Shape ``(N, 2)`` in ``[-1, 1]^2``.

    Returns:
        Shape ``(N, 1)`` targets.
    """

    return torch.exp(torch.sin(math.pi * inputs[:, :1]) + inputs[:, 1:2] ** 2)


def make_kan_factory(config: KANConfig) -> Callable[[int], nn.Module]:
    """Return a factory building a freshly seeded KAN.

    Args:
        config: Network configuration.

    Returns:
        A callable mapping a seed to a new :class:`~modern_nn_lab.tracks.kan.KAN`.
    """

    def factory(seed: int) -> nn.Module:
        seed_everything(seed)
        return KAN(config)

    return factory


def make_mlp_factory(
    in_features: int, out_features: int, hidden_widths: list[int]
) -> Callable[[int], nn.Module]:
    """Return a factory building a freshly seeded MLP baseline.

    Args:
        in_features: Input width.
        out_features: Output width.
        hidden_widths: Hidden widths, typically from :func:`match_parameter_budget`.

    Returns:
        A callable mapping a seed to a new :class:`~modern_nn_lab.tracks.kan.MLP`.
    """

    def factory(seed: int) -> nn.Module:
        seed_everything(seed)
        return MLP(in_features, out_features, hidden_widths=hidden_widths)

    return factory


def regularized_mse(model: nn.Module, weight: float) -> Callable[[Tensor, Tensor], Tensor]:
    """Build an MSE loss with the KAN L1 surrogate attached.

    The penalty is read from the live model, so it is applied to the target architecture
    only where it is defined and never silently to the baseline.

    Args:
        model: Model being trained.
        weight: Regularization coefficient; ``0`` returns plain MSE.

    Returns:
        A loss callable.
    """

    def loss_fn(predictions: Tensor, targets: Tensor) -> Tensor:
        loss = torch.mean((predictions - targets) ** 2)
        if weight > 0.0 and isinstance(model, KAN):
            loss = loss + weight * model.regularization()
        return loss

    return loss_fn


def _training_config(settings: KANExperimentConfig, learning_rate: float) -> TrainingConfig:
    """Translate the track settings into the shared optimizer configuration."""

    return TrainingConfig(
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=learning_rate,
        eval_every=max(1, settings.epochs // 10),
        cosine_schedule=True,
    )


def select_learning_rate(
    factory: Callable[[int], nn.Module],
    split: SupervisedSplit,
    *,
    settings: KANExperimentConfig,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Pick a learning rate for one architecture using the **validation** split.

    Why this exists: an earlier version of this suite trained every model at one shared
    learning rate. At that rate the MLP baseline was badly under-trained on the synthetic
    tasks — it reached a test MSE of about 0.43 on ``sin(2πx)``, close to simply
    predicting the mean, while the same architecture reaches about 0.001 at a larger rate
    and a longer budget. The resulting "KAN wins by three orders of magnitude" was an
    artefact of the optimizer setting, not a property of the architecture.

    Every architecture and every ablation therefore searches the *same* grid, with the
    same number of trials and the same epoch budget, and the choice is made on validation
    data. The test split is never consulted.

    Args:
        factory: Builds a fresh model for a given seed.
        split: Dataset split; only train and validation are used here.
        settings: Track settings supplying the grid and epoch budget.
        loss_fn: Training loss.
        seed: Seed used for every trial, so trials differ only in learning rate.

    Returns:
        ``(best_learning_rate, {rate: validation_mse})``.
    """

    scores: dict[str, float] = {}
    best_rate = settings.learning_rate
    best_score = float("inf")

    for rate in settings.learning_rate_grid:
        model = factory(seed)
        outcome = train_supervised(
            model,
            split.train_inputs,
            split.train_targets,
            config=_training_config(settings, rate),
            loss_fn=loss_fn,
            eval_inputs=split.val_inputs,
            eval_targets=split.val_targets,
            metric_fn=mean_squared_error,
        )
        score = outcome.final_metric if outcome.status == "success" else float("inf")
        if not math.isfinite(score):
            score = float("inf")
        scores[f"{rate:g}"] = score
        if score < best_score:
            best_score, best_rate = score, rate

    return best_rate, scores


def _run_torch_model(
    factory: Callable[[int], nn.Module],
    split: SupervisedSplit,
    *,
    architecture: str,
    variant: str | None,
    settings: KANExperimentConfig,
    output_dir: Path,
    extra_config: dict[str, object],
    regularization_weight: float = 0.0,
    notes: str | None = None,
    seeds: tuple[int, ...] | None = None,
    learning_rate: float | None = None,
) -> RunGroup:
    """Run one architecture/variant on one split with the shared protocol.

    Args:
        factory: Builds a fresh model for a given seed.
        split: Dataset split.
        architecture: Architecture name recorded with the results.
        variant: Ablation or configuration label.
        settings: Track experiment settings.
        output_dir: Directory receiving records.
        extra_config: Extra configuration merged into every record.
        regularization_weight: Weight of the KAN L1 surrogate.
        notes: Free-text caveat stored with every record.
        seeds: Seeds to run; defaults to ``settings.seeds``.
        learning_rate: Skip the search and use this rate. Used by the sensitivity sweep,
            where the point is to vary the architecture, not the optimizer.

    Returns:
        The resulting :class:`RunGroup`.
    """

    # The L1 surrogate is a property of the live model, so the loss closure has to see
    # whichever model the runner most recently built for the current seed.
    current: dict[str, nn.Module] = {}

    def seeded_factory(seed: int) -> nn.Module:
        model = factory(seed)
        current["model"] = model
        return model

    def bound_loss(predictions: Tensor, targets: Tensor) -> Tensor:
        return regularized_mse(current["model"], regularization_weight)(predictions, targets)

    run_seeds = seeds if seeds is not None else settings.seeds

    if learning_rate is None:
        learning_rate, lr_scores = select_learning_rate(
            seeded_factory,
            split,
            settings=settings,
            loss_fn=bound_loss,
            seed=run_seeds[0],
        )
        selection = "validation grid search"
    else:
        lr_scores = {}
        selection = "inherited from the tuned baseline configuration"

    return run_seeded_experiment(
        seeded_factory,
        split,
        spec=RunSpec(
            track=TRACK,
            architecture=architecture,
            metric_name="test_mse",
            higher_is_better=False,
            variant=variant,
            architecture_version=ARCHITECTURE_VERSION,
            extra_config={
                **extra_config,
                "regularization_weight": regularization_weight,
                "learning_rate_selection": selection,
                "learning_rate_grid": list(settings.learning_rate_grid),
                "learning_rate_validation_mse": lr_scores,
            },
            notes=notes,
        ),
        training_config=_training_config(settings, learning_rate),
        loss_fn=bound_loss,
        metric_fn=mean_squared_error,
        output_dir=output_dir,
        seeds=run_seeds,
        profile_inference_cost=True,
    )


def run_function_task(
    split: SupervisedSplit,
    *,
    layer_widths: tuple[int, ...],
    settings: KANExperimentConfig,
    output_dir: Path,
) -> dict[str, RunGroup]:
    """Compare KAN, matched-budget MLP, and two ablations on one synthetic function.

    Args:
        split: The dataset split.
        layer_widths: KAN widths, for example ``(2, 5, 1)``.
        settings: Track experiment settings.
        output_dir: Directory receiving records.

    Returns:
        Mapping from run label to :class:`RunGroup`.
    """

    base_config = KANConfig(
        layer_widths=layer_widths,
        grid_size=DEFAULT_GRID_SIZE,
        spline_order=DEFAULT_SPLINE_ORDER,
    )
    reference = KAN(base_config)
    kan_parameters = count_parameters(reference)
    hidden = match_parameter_budget(
        kan_parameters, layer_widths[0], layer_widths[-1], depth=len(layer_widths) - 2 or 1
    )
    baseline = MLP(layer_widths[0], layer_widths[-1], hidden_widths=hidden)

    shared = {
        "kan_parameters": kan_parameters,
        "mlp_parameters": count_parameters(baseline),
        "mlp_hidden_widths": hidden,
        "layer_widths": list(layer_widths),
    }

    groups: dict[str, RunGroup] = {}
    groups["kan"] = _run_torch_model(
        make_kan_factory(base_config),
        split,
        architecture="kan",
        variant=None,
        settings=settings,
        output_dir=output_dir,
        extra_config={**shared, **base_config.as_dict()},
    )
    groups["mlp"] = _run_torch_model(
        make_mlp_factory(layer_widths[0], layer_widths[-1], hidden),
        split,
        architecture="mlp",
        variant="matched-parameters",
        settings=settings,
        output_dir=output_dir,
        extra_config=shared,
        notes="Hidden width searched to match the KAN parameter count.",
    )

    frozen = replace(base_config, learnable_spline=False)
    groups["kan_frozen_edges"] = _run_torch_model(
        make_kan_factory(frozen),
        split,
        architecture="kan",
        variant="frozen-edge-functions",
        settings=settings,
        output_dir=output_dir,
        extra_config={**shared, **frozen.as_dict()},
        notes=(
            "Ablation: spline coefficients frozen at initialization; only the base branch learns."
        ),
    )

    pure_spline = replace(base_config, use_base_branch=False)
    groups["kan_pure_spline"] = _run_torch_model(
        make_kan_factory(pure_spline),
        split,
        architecture="kan",
        variant="no-base-branch",
        settings=settings,
        output_dir=output_dir,
        extra_config={**shared, **pure_spline.as_dict()},
        notes="Ablation: residual SiLU branch removed; the edge function is a pure spline.",
    )
    return groups


def run_sensitivity(
    split: SupervisedSplit,
    *,
    layer_widths: tuple[int, ...],
    settings: KANExperimentConfig,
    output_dir: Path,
) -> dict[str, RunGroup]:
    """Sweep grid size, spline order, and regularization weight.

    Args:
        split: The dataset split.
        layer_widths: KAN widths.
        settings: Track experiment settings.
        output_dir: Directory receiving records.

    Returns:
        Mapping from run label to :class:`RunGroup`.
    """

    groups: dict[str, RunGroup] = {}

    for grid_size in settings.grid_sweep:
        config = KANConfig(layer_widths=layer_widths, grid_size=grid_size, spline_order=3)
        groups[f"grid_{grid_size}"] = _run_torch_model(
            make_kan_factory(config),
            split,
            architecture="kan",
            variant=f"grid-size-{grid_size}",
            settings=settings,
            output_dir=output_dir,
            extra_config={**config.as_dict(), "sweep": "grid_size"},
            seeds=settings.sweep_seeds,
        )

    for order in settings.order_sweep:
        config = KANConfig(layer_widths=layer_widths, grid_size=5, spline_order=order)
        groups[f"order_{order}"] = _run_torch_model(
            make_kan_factory(config),
            split,
            architecture="kan",
            variant=f"spline-order-{order}",
            settings=settings,
            output_dir=output_dir,
            extra_config={**config.as_dict(), "sweep": "spline_order"},
            seeds=settings.sweep_seeds,
        )

    for weight in settings.regularization_sweep:
        config = KANConfig(layer_widths=layer_widths, grid_size=5, spline_order=3)
        groups[f"reg_{weight}"] = _run_torch_model(
            make_kan_factory(config),
            split,
            architecture="kan",
            variant=f"l1-weight-{weight}",
            settings=settings,
            output_dir=output_dir,
            extra_config={**config.as_dict(), "sweep": "regularization"},
            regularization_weight=weight,
            seeds=settings.sweep_seeds,
        )

    return groups


def run_tabular_benchmark(settings: KANExperimentConfig, output_dir: Path) -> dict[str, RunGroup]:
    """Benchmark on a public tabular regression dataset against neural and tree baselines.

    Uses the scikit-learn *diabetes* dataset, which ships with the library, so the
    benchmark runs without a network download and its fingerprint is stable.

    Args:
        settings: Track experiment settings.
        output_dir: Directory receiving records.

    Returns:
        Mapping from run label to :class:`RunGroup`.
    """

    from sklearn.datasets import load_diabetes
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    raw = load_diabetes()
    split = make_tabular_split(
        torch.as_tensor(raw.data, dtype=torch.float32),
        torch.as_tensor(raw.target, dtype=torch.float32),
        name="sklearn-diabetes",
        seed=1729,
        standardize_targets=True,
    )

    layer_widths = (split.input_dim, 5, 1)
    config = KANConfig(layer_widths=layer_widths, grid_size=5, spline_order=3)
    kan_parameters = count_parameters(KAN(config))
    hidden = match_parameter_budget(kan_parameters, split.input_dim, 1, depth=1)

    shared: dict[str, object] = {
        "kan_parameters": kan_parameters,
        "mlp_hidden_widths": hidden,
        "benchmark": "tabular-regression",
    }

    groups: dict[str, RunGroup] = {}
    groups["kan"] = _run_torch_model(
        make_kan_factory(config),
        split,
        architecture="kan",
        variant=None,
        settings=settings,
        output_dir=output_dir,
        extra_config={**shared, **config.as_dict()},
    )
    groups["mlp"] = _run_torch_model(
        make_mlp_factory(split.input_dim, 1, hidden),
        split,
        architecture="mlp",
        variant="matched-parameters",
        settings=settings,
        output_dir=output_dir,
        extra_config=shared,
    )

    tree_note = (
        "parameter_count reports total tree nodes, which is not commensurable with a "
        "neural weight count; see reports/kan.md."
    )
    groups["random_forest"] = run_external_baseline(
        lambda seed: RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
        split,
        spec=RunSpec(
            track=TRACK,
            architecture="random_forest",
            metric_name="test_mse",
            higher_is_better=False,
            extra_config=shared,
            notes=tree_note,
        ),
        metric_fn=mean_squared_error,
        output_dir=output_dir,
        seeds=settings.seeds,
    )
    groups["gradient_boosting"] = run_external_baseline(
        lambda seed: GradientBoostingRegressor(random_state=seed),
        split,
        spec=RunSpec(
            track=TRACK,
            architecture="gradient_boosting",
            metric_name="test_mse",
            higher_is_better=False,
            extra_config=shared,
            notes=tree_note,
        ),
        metric_fn=mean_squared_error,
        output_dir=output_dir,
        seeds=settings.seeds,
    )
    return groups


def export_edge_functions(
    split: SupervisedSplit,
    *,
    layer_widths: tuple[int, ...],
    settings: KANExperimentConfig,
    output_dir: Path,
    seed: int = 0,
) -> Path:
    """Train one KAN and serialize its learned edge functions for plotting.

    Figures must be regenerable from saved results, so the *values* of every
    ``phi_ji`` are written to disk rather than a rendered image.

    Args:
        split: Dataset used for training.
        layer_widths: KAN widths.
        settings: Track experiment settings.
        output_dir: Directory receiving the JSON file.
        seed: Seed for the exported model.

    Returns:
        Path of the written file.
    """

    config = KANConfig(
        layer_widths=layer_widths,
        grid_size=DEFAULT_GRID_SIZE,
        spline_order=DEFAULT_SPLINE_ORDER,
    )

    def factory(model_seed: int) -> nn.Module:
        seed_everything(model_seed)
        return KAN(config)

    learning_rate, _ = select_learning_rate(
        factory,
        split,
        settings=settings,
        loss_fn=lambda p, t: torch.mean((p - t) ** 2),
        seed=seed,
    )
    seed_everything(seed)
    model = KAN(config)
    train_supervised(
        model,
        split.train_inputs,
        split.train_targets,
        config=TrainingConfig(
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            learning_rate=learning_rate,
            seed=seed,
            cosine_schedule=True,
        ),
        loss_fn=regularized_mse(model, 0.0),
    )

    samples = torch.linspace(-1.0, 1.0, settings.edge_samples)
    edges = model.first_layer_edges(samples)

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "track": TRACK,
        "dataset": split.name,
        "dataset_fingerprint": split.fingerprint,
        "seed": seed,
        "config": config.as_dict(),
        "samples": samples.tolist(),
        "edges": edges.tolist(),
        "edge_layout": "edges[out_feature][in_feature][sample]",
    }
    # Derived diagnostics live beside the records but outside the record schema.
    artefacts = output_dir / ARTEFACT_DIRNAME
    artefacts.mkdir(parents=True, exist_ok=True)
    path = artefacts / "edge_functions__compositional.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run(output_dir: Path | str | None = None, *, quick: bool = False) -> None:
    """Run the whole KAN suite and write every record under ``output_dir``.

    Args:
        output_dir: Destination directory. Defaults to ``results/kan``.
        quick: Use three seeds and far fewer epochs. Intended for smoke-testing the
            protocol, not for reporting; records carry the reduced settings.
    """

    destination = Path(output_dir) if output_dir is not None else Path("results") / TRACK
    settings = (
        KANExperimentConfig(
            seeds=(0, 1, 2),
            sweep_seeds=(0, 1),
            epochs=30,
            learning_rate_grid=(2e-2,),
            n_samples=600,
            grid_sweep=(3, 5),
            order_sweep=(1, 3),
            regularization_sweep=(0.0, 1e-2),
            edge_samples=41,
        )
        if quick
        else KANExperimentConfig()
    )

    univariate = make_function_split(
        univariate_target,
        name="sin-2pix",
        input_dim=1,
        n_samples=settings.n_samples,
        noise_std=settings.noise_std,
        seed=1729,
    )
    compositional = make_function_split(
        compositional_target,
        name="exp-sin-plus-square",
        input_dim=2,
        n_samples=settings.n_samples,
        noise_std=settings.noise_std,
        seed=1729,
    )

    run_function_task(univariate, layer_widths=(1, 3, 1), settings=settings, output_dir=destination)
    run_function_task(
        compositional, layer_widths=(2, 5, 1), settings=settings, output_dir=destination
    )
    run_sensitivity(
        compositional, layer_widths=(2, 5, 1), settings=settings, output_dir=destination
    )
    run_tabular_benchmark(settings, destination)
    export_edge_functions(
        compositional, layer_widths=(2, 5, 1), settings=settings, output_dir=destination
    )
