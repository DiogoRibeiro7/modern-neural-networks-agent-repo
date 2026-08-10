"""Experiment suite for Track 08 — relational foundation-model prototype.

Five regimes, each isolating one kind of relational reasoning, against four comparisons:
the prototype, the homogeneous-GNN baseline the prompt asks for, leakage-safe feature
engineering with a GBDT, and a target-table-only floor. Two ablations remove typing and
time from the prototype, one flag each.

Everything reads its inputs from :mod:`modern_nn_lab.tracks.relational.sampler`, including
the GBDT features, so no comparison here can differ in what it was allowed to see. The
database is generated once per regime from a fixed seed and shared by every model and every
training seed: seed variance in this track is variance over *initialization and training*,
not over datasets, and the report says so rather than letting the intervals be read as
something wider.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from modern_nn_lab.experiments.data import SupervisedSplit
from modern_nn_lab.experiments.evaluation import accuracy
from modern_nn_lab.experiments.external import run_meta_evaluation
from modern_nn_lab.experiments.records import ARTEFACT_DIRNAME, RESULT_SCHEMA_VERSION
from modern_nn_lab.experiments.runner import RunGroup, RunSpec, run_seeded_experiment
from modern_nn_lab.experiments.training import TrainingConfig, train_supervised
from modern_nn_lab.tracks.relational import (
    FEATURE_NAMES,
    REGIMES,
    Normalizer,
    Regime,
    RelationalConfig,
    RelationalEncoder,
    RelationalExperimentConfig,
    RelationalProblem,
    SamplingConfig,
    TargetOnlyModel,
    attribute,
    build_row_sets,
    flatten,
    generate,
    leakage_canary_strength,
    reachable_paths,
    summarize,
)

TRACK = "relational"
ARCHITECTURE_VERSION = "0.1.0"

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

VARIANTS: dict[str, dict[str, bool]] = {
    "relational": {},
    "no-types": {"use_types": False},
    "no-time": {"use_time": False},
    "gnn-flat": {"use_links": False},
}
"""Model name to the flags that differ from the default. Each varies one thing."""


WIDTH_CANDIDATES: tuple[int, ...] = (16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160)
"""Widths searched when matching a variant's parameter count to the prototype's."""


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters.

    Args:
        model: Any module.

    Returns:
        Total trainable parameter count.
    """

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def match_width(kind: str, overrides: dict[str, bool], target: int) -> int:
    """Choose the width whose parameter count is closest to the prototype's.

    Removing typing deletes three of four row encoders, and the target-only model has no
    message passing at all, so at a shared width these models would be several times
    smaller than the prototype. Comparing them at that width would confound the mechanism
    with capacity — the failure this repository corrected in Track 01 — so each is widened
    until its parameter count matches, and the achieved counts are recorded so the residual
    mismatch is visible rather than assumed away.

    Args:
        kind: ``"relational"`` or ``"target_only"``.
        overrides: Flags that differ from the default configuration.
        target: Parameter count to match.

    Returns:
        The best width from :data:`WIDTH_CANDIDATES`.
    """

    best_width = WIDTH_CANDIDATES[0]
    best_gap = float("inf")
    for width in WIDTH_CANDIDATES:
        candidate = _build(kind, RelationalConfig(d_model=width, **overrides))
        gap = abs(count_parameters(candidate) - target)
        if gap < best_gap:
            best_gap = gap
            best_width = width
    return best_width


def build_problem(regime: Regime, settings: RelationalExperimentConfig) -> RelationalProblem:
    """Generate the database for one regime.

    Args:
        regime: Which relational structure the label depends on.
        settings: Track settings.

    Returns:
        The problem.
    """

    return generate(
        regime,
        n_entities=settings.n_entities,
        n_products=settings.n_products,
        orders_per_entity=settings.orders_per_entity,
        signals_per_entity=settings.signals_per_entity,
        horizon=settings.horizon,
        recent_window=settings.recent_window,
        seed=settings.data_seed,
    )


def temporal_split(
    problem: RelationalProblem, settings: RelationalExperimentConfig
) -> tuple[SupervisedSplit, Normalizer]:
    """Split prediction points by time and encode each one's neighbourhood.

    The split is chronological, not random. Training on points interleaved with the test
    points would let a model learn from a period it is later scored on, which is the
    dataset-level version of the leak the sampler prevents at the row level.

    Normalization statistics come from the training rows only, so the split being scored
    never influences its own features.

    Args:
        problem: Database and task.
        settings: Track settings.

    Returns:
        ``(split, normalizer)``.
    """

    rows = build_row_sets(
        problem,
        SamplingConfig(max_events=settings.max_events, max_distractors=settings.max_distractors),
    )
    order = torch.argsort(problem.task.timestamps)
    labels = problem.task.labels.float()

    total = int(order.shape[0])
    n_train = int(total * TRAIN_FRACTION)
    n_val = int(total * VAL_FRACTION)
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    normalizer = Normalizer.fit(rows[train_idx])
    normalized = normalizer(rows)

    split = SupervisedSplit(
        name=f"{problem.task.regime}",
        train_inputs=normalized[train_idx],
        train_targets=labels[train_idx],
        val_inputs=normalized[val_idx],
        val_targets=labels[val_idx],
        test_inputs=normalized[test_idx],
        test_targets=labels[test_idx],
        strategy=(
            f"chronological split of {total} prediction points by timestamp "
            f"({n_train}/{n_val}/{total - n_train - n_val}); each point's neighbourhood "
            "contains only rows strictly earlier than its own prediction time, and "
            "normalization statistics are fitted on the training rows alone"
        ),
        metadata={
            "regime": problem.task.regime,
            "row_counts": problem.database.row_counts(),
            "max_events": settings.max_events,
            "max_distractors": settings.max_distractors,
            "leakage_canary_accuracy": leakage_canary_strength(problem),
        },
    )
    return split, normalizer


def binary_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Binary cross-entropy over logits.

    Args:
        predictions: Shape ``(B,)`` logits.
        targets: Shape ``(B,)`` targets in ``{0, 1}``.

    Returns:
        Scalar loss.
    """

    return nn.functional.binary_cross_entropy_with_logits(predictions, targets)


def binary_accuracy(predictions: Tensor, targets: Tensor) -> float:
    """Accuracy of thresholded logits.

    Args:
        predictions: Shape ``(B,)`` logits.
        targets: Shape ``(B,)`` targets in ``{0, 1}``.

    Returns:
        Accuracy.
    """

    probabilities = torch.sigmoid(predictions)
    stacked = torch.stack([1.0 - probabilities, probabilities], dim=-1)
    return accuracy(stacked, targets.long())


def select_learning_rate(
    factory_kind: str,
    config: RelationalConfig,
    split: SupervisedSplit,
    settings: RelationalExperimentConfig,
) -> float:
    """Choose a learning rate per architecture on the validation split.

    A shared rate compares step sizes rather than architectures; this repository learned
    that the hard way in Track 01 and applies the rule everywhere since.

    Args:
        factory_kind: ``"relational"`` or ``"target_only"``.
        config: Architecture configuration.
        split: The split whose validation portion drives the choice.
        settings: Track settings.

    Returns:
        The best rate from the grid.
    """

    best_rate = settings.learning_rate_grid[0]
    best_score = float("-inf")

    for rate in settings.learning_rate_grid:
        torch.manual_seed(0)
        model = _build(factory_kind, config)
        outcome = train_supervised(
            model,
            split.train_inputs,
            split.train_targets,
            config=TrainingConfig(
                epochs=settings.epochs,
                batch_size=settings.batch_size,
                learning_rate=rate,
                seed=0,
            ),
            loss_fn=binary_loss,
            eval_inputs=split.val_inputs,
            eval_targets=split.val_targets,
            metric_fn=binary_accuracy,
            higher_is_better=True,
        )
        if outcome.best_metric > best_score:
            best_score = outcome.best_metric
            best_rate = rate

    return best_rate


def _build(kind: str, config: RelationalConfig) -> nn.Module:
    """Construct a model of the requested kind.

    Args:
        kind: ``"relational"`` or ``"target_only"``.
        config: Architecture configuration.

    Returns:
        The model.

    Raises:
        ValueError: If the kind is unknown.
    """

    if kind == "relational":
        return RelationalEncoder(config)
    if kind == "target_only":
        return TargetOnlyModel(config)
    raise ValueError(f"unknown model kind {kind!r}")


def run_regime(
    regime: Regime, settings: RelationalExperimentConfig, output_dir: Path
) -> dict[str, RunGroup]:
    """Train every model on one regime and write records.

    Args:
        regime: Which relational structure the label depends on.
        settings: Track settings.
        output_dir: Directory receiving records.

    Returns:
        Mapping from model name to :class:`RunGroup`.
    """

    problem = build_problem(regime, settings)
    split, _ = temporal_split(problem, settings)
    target = count_parameters(RelationalEncoder(RelationalConfig()))
    shared: dict[str, Any] = {
        "regime": regime,
        "n_entities": settings.n_entities,
        "max_events": settings.max_events,
        "leakage_canary_accuracy": leakage_canary_strength(problem),
    }
    groups: dict[str, RunGroup] = {}

    for name, overrides in VARIANTS.items():
        config = RelationalConfig(d_model=match_width("relational", overrides, target), **overrides)
        rate = select_learning_rate("relational", config, split, settings)
        groups[name] = run_seeded_experiment(
            _factory("relational", config),
            split,
            spec=RunSpec(
                track=TRACK,
                architecture=name,
                metric_name="test_accuracy",
                higher_is_better=True,
                variant=regime,
                architecture_version=ARCHITECTURE_VERSION,
                extra_config={**shared, **config.as_dict(), "selected_learning_rate": rate},
                notes=(
                    "Reads its neighbourhood from the shared temporal sampler; no model in "
                    "this track applies its own time filter."
                ),
            ),
            training_config=TrainingConfig(
                epochs=settings.epochs, batch_size=settings.batch_size, learning_rate=rate
            ),
            loss_fn=binary_loss,
            metric_fn=binary_accuracy,
            output_dir=output_dir,
            seeds=settings.seeds,
        )

    target_config = RelationalConfig(d_model=match_width("target_only", {}, target))
    target_rate = select_learning_rate("target_only", target_config, split, settings)
    groups["target-only"] = run_seeded_experiment(
        _factory("target_only", target_config),
        split,
        spec=RunSpec(
            track=TRACK,
            architecture="target-only",
            metric_name="test_accuracy",
            higher_is_better=True,
            variant=regime,
            architecture_version=ARCHITECTURE_VERSION,
            extra_config={**shared, **target_config.as_dict(), "uses_related_tables": False},
            notes="The floor: the entity's own columns and nothing else.",
        ),
        training_config=TrainingConfig(
            epochs=settings.epochs, batch_size=settings.batch_size, learning_rate=target_rate
        ),
        loss_fn=binary_loss,
        metric_fn=binary_accuracy,
        output_dir=output_dir,
        seeds=settings.seeds,
    )

    groups["gbdt-flat"] = _run_gbdt(split, regime, shared, settings, output_dir)
    return groups


def _factory(kind: str, config: RelationalConfig) -> Callable[[int], nn.Module]:
    """Return a model factory that builds a freshly initialized model per seed.

    Args:
        kind: ``"relational"`` or ``"target_only"``.
        config: Architecture configuration.

    Returns:
        A callable taking a seed and returning a new model.
    """

    def build(seed: int) -> nn.Module:
        return _seeded(_build(kind, config), seed)

    return build


def _seeded(model: nn.Module, seed: int) -> nn.Module:
    """Re-initialize a model under a given seed.

    Args:
        model: A freshly constructed model.
        seed: Seed.

    Returns:
        The same model, re-initialized.
    """

    torch.manual_seed(seed)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


def _run_gbdt(
    split: SupervisedSplit,
    regime: Regime,
    shared: dict[str, Any],
    settings: RelationalExperimentConfig,
    output_dir: Path,
) -> RunGroup:
    """Fit gradient boosting on the flattened, leakage-safe feature matrix.

    Args:
        split: The shared split, whose row sets are flattened here.
        regime: Regime label.
        shared: Configuration shared across models.
        settings: Track settings.
        output_dir: Directory receiving records.

    Returns:
        A :class:`RunGroup`.
    """

    from sklearn.ensemble import GradientBoostingClassifier

    train_x = flatten(split.train_inputs).numpy()
    train_y = split.train_targets.long().numpy()
    test_x = flatten(split.test_inputs).numpy()
    test_y = split.test_targets.long().numpy()

    def fit_and_score(seed: int) -> tuple[float, dict[str, float], dict[str, Any]]:
        model = GradientBoostingClassifier(random_state=seed)
        model.fit(train_x, train_y)
        score = float((model.predict(test_x) == test_y).mean())
        importance = dict(
            zip(FEATURE_NAMES, (float(value) for value in model.feature_importances_), strict=True)
        )
        top = sorted(importance.items(), key=lambda item: -item[1])[:3]
        return (
            score,
            {"train_accuracy": float((model.predict(train_x) == train_y).mean())},
            {**shared, "n_features": len(FEATURE_NAMES), "top_features": [name for name, _ in top]},
        )

    return run_meta_evaluation(
        fit_and_score,
        spec=RunSpec(
            track=TRACK,
            architecture="gbdt-flat",
            metric_name="test_accuracy",
            higher_is_better=True,
            variant=regime,
            architecture_version=ARCHITECTURE_VERSION,
            extra_config=shared,
            notes=(
                "Aggregates the same time-gated rows into one feature matrix. The joins "
                "were written by hand and do not adapt to a change of schema."
            ),
        ),
        dataset=split.name,
        split_strategy=split.strategy,
        output_dir=output_dir,
        seeds=settings.seeds,
        dataset_fingerprint=split.fingerprint,
    )


def write_leakage_audit(settings: RelationalExperimentConfig, output_dir: Path) -> Path:
    """Serialize the temporal-leakage audit and an explainability trace.

    This is the artefact behind both acceptance criteria. For every regime it records the
    accuracy a leaking pipeline would reach, the minimum event age actually observed in the
    sampled neighbourhoods, and an example prediction's relational paths with attributions.

    Args:
        settings: Track settings.
        output_dir: Directory receiving the artefact.

    Returns:
        Path of the written file.
    """

    audit: list[dict[str, Any]] = []
    example: dict[str, Any] = {}

    for regime in REGIMES:
        problem = build_problem(regime, settings)
        sampling = SamplingConfig(
            max_events=settings.max_events, max_distractors=settings.max_distractors
        )
        gated = build_row_sets(problem, sampling)
        leaky = build_row_sets(problem, sampling, include_future=True)

        summaries = [summarize(reachable_paths(gated, point)) for point in range(len(problem.task))]
        observed = [entry["min_event_age"] for entry in summaries if entry["min_event_age"]]

        audit.append(
            {
                "regime": regime,
                "leakage_canary_accuracy": leakage_canary_strength(problem),
                "rows_visible_gated": float((gated[..., 4] > 0).float().sum(dim=1).mean()),
                "rows_visible_ungated": float((leaky[..., 4] > 0).float().sum(dim=1).mean()),
                "min_event_age_observed": min(observed) if observed else None,
                "mean_paths_per_prediction": sum(entry["n_paths"] for entry in summaries)
                / max(len(summaries), 1),
                "distinct_paths": sorted(
                    {path for entry in summaries for path in entry["distinct_paths"]}
                ),
            }
        )

        if regime == "multi_hop":
            # Traced on the *unnormalized* rows on purpose. Standardization rescales the
            # elapsed-time channel, so a normalized row reports a negative "age" — which in
            # a leakage audit reads as exactly the alarm this artefact exists to raise.
            # The ages printed here are therefore real elapsed times.
            model = _seeded(RelationalEncoder(RelationalConfig()), 0)
            traces = attribute(model, gated, 0)
            example = {
                "regime": regime,
                "note": (
                    "Ages are elapsed times on unnormalized rows, so they are comparable "
                    "with the audit above. Attributions come from an untrained model, so "
                    "they show what the architecture can reach rather than what a fitted "
                    "model relies on; the path list beside them is exact either way."
                ),
                "paths": [trace.describe() for trace in traces],
            }

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "track": TRACK,
        "measurement": "temporal_leakage_audit",
        "note": (
            "`leakage_canary_accuracy` is the accuracy a pipeline reaches by reading "
            "post-timestamp rows. The sampler cannot reach it, and the gated row counts "
            "beside the ungated ones show why: the future rows are simply absent. "
            "`min_event_age_observed` is over rows carrying a timestamp; static rows are "
            "exempt because they have none."
        ),
        "regimes": audit,
        "explainability_example": example,
    }

    artefacts = output_dir / ARTEFACT_DIRNAME
    artefacts.mkdir(parents=True, exist_ok=True)
    path = artefacts / "leakage_audit.json"
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return path


def run(output_dir: Path | str | None = None, *, quick: bool = False) -> None:
    """Run the whole relational suite.

    Args:
        output_dir: Destination directory. Defaults to ``results/relational``.
        quick: Reduced seeds, entities, and epochs.
    """

    destination = Path(output_dir) if output_dir is not None else Path("results") / TRACK
    settings = (
        RelationalExperimentConfig(
            seeds=(0, 1), n_entities=200, epochs=12, learning_rate_grid=(3e-3,)
        )
        if quick
        else RelationalExperimentConfig()
    )

    for regime in REGIMES:
        run_regime(regime, settings, destination)

    write_leakage_audit(settings, destination)
