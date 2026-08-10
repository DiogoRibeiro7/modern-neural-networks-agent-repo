"""Experiment suite for Track 07 — Prior-Fitted Networks.

**Deliverable A only.** The official TabPFN checkpoint (deliverable B) is not executed
here; see :mod:`modern_nn_lab.tracks.pfn.reference` for why, and the report for what that
means. No number in this suite is compared against a pre-trained checkpoint.

The protocol is a fair one by construction, because every model sees exactly the same
context and is asked exactly the same queries:

- the **PFN** is prior-fitted once, then answers each evaluation task in a single forward
  pass with no gradient step;
- every **baseline** is fitted from scratch on that task's context, which is the normal
  way those models are used.

That asymmetry is the mechanism under study, not a confound: the PFN has paid its training
cost up front on *synthetic tasks from a known prior*, and the experiment asks what that
buys and where it fails. The out-of-prior tasks are where it should fail.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from modern_nn_lab.experiments.evaluation import accuracy, expected_calibration_error
from modern_nn_lab.experiments.external import run_meta_evaluation
from modern_nn_lab.experiments.records import ARTEFACT_DIRNAME, RESULT_SCHEMA_VERSION, fingerprint
from modern_nn_lab.experiments.runner import RunGroup, RunSpec
from modern_nn_lab.reproducibility import seed_everything
from modern_nn_lab.tracks.pfn import (
    PFNConfig,
    PFNExperimentConfig,
    PriorFittedNetwork,
    TaskPrior,
    build_prior,
    pretraining_advantage_note,
    tabpfn_available,
)

TRACK = "pfn"
ARCHITECTURE_VERSION = "0.1.0"

EvaluateFn = Callable[[int], tuple[float, dict[str, float], dict[str, Any]]]


def fit_prior(
    prior: TaskPrior, settings: PFNExperimentConfig, *, seed: int, n_features: int | None = None
) -> tuple[PriorFittedNetwork, list[float]]:
    """Prior-fit a PFN: the only training that happens in this track.

    Args:
        prior: The task distribution to fit.
        settings: Track settings.
        seed: Seed for initialization and task sampling.
        n_features: Input width; defaults to ``settings.n_features``. Must match the
            prior's width.

    Returns:
        ``(model, loss_trajectory)``.
    """

    width = n_features if n_features is not None else settings.n_features
    seed_everything(seed)
    model = PriorFittedNetwork(PFNConfig(n_features=width))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)

    losses: list[float] = []
    model.train()
    for _ in range(settings.prior_fitting_steps):
        batch = prior.sample(
            batch_size=settings.tasks_per_step,
            n_context=settings.train_context,
            n_query=settings.n_query,
            generator=generator,
        )
        logits = model(batch.context_inputs, batch.context_labels, batch.query_inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), batch.query_labels.reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        # Torch ships `Tensor.backward` unannotated; the call itself is what strict mypy
        # flags, not any misuse here.
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    return model, losses


def blank_features(inputs: Tensor, rate: float, generator: torch.Generator) -> Tensor:
    """Blank a fraction of feature entries, imputing with the population mean.

    Inputs are standard normal, so zero *is* the mean of the feature distribution. Every
    model therefore receives the identical imputed matrix and none of them gets a private
    missingness mechanism. This measures robustness to degraded features, not native
    missing-value handling — a real distinction, and the report states it.

    Args:
        inputs: Shape ``(B, n, d)``.
        rate: Fraction of entries to blank, in ``[0, 1)``.
        generator: Seeded generator.

    Returns:
        A new tensor with the selected entries set to zero.
    """

    if rate <= 0:
        return inputs
    missing = torch.rand(inputs.shape, generator=generator) < rate
    return inputs.masked_fill(missing, 0.0)


def evaluate_pfn(
    model: PriorFittedNetwork,
    prior: TaskPrior,
    *,
    n_context: int,
    n_query: int,
    n_tasks: int,
    generator: torch.Generator,
    missing_rate: float = 0.0,
) -> tuple[float, float]:
    """Score the PFN over independently sampled tasks.

    Args:
        model: A prior-fitted network.
        prior: The task distribution to sample from.
        n_context: Context size per task.
        n_query: Queries per task.
        n_tasks: Tasks to sample.
        generator: Seeded generator.
        missing_rate: Fraction of feature entries blanked before prediction.

    Returns:
        ``(accuracy, expected_calibration_error)`` pooled over all queries.
    """

    batch = prior.sample(
        batch_size=n_tasks, n_context=n_context, n_query=n_query, generator=generator
    )
    probabilities = model.predict_proba(
        blank_features(batch.context_inputs, missing_rate, generator),
        batch.context_labels,
        blank_features(batch.query_inputs, missing_rate, generator),
    )
    flat_probs = probabilities.reshape(-1, probabilities.shape[-1])
    flat_labels = batch.query_labels.reshape(-1)
    return (
        accuracy(flat_probs, flat_labels),
        expected_calibration_error(flat_probs, flat_labels),
    )


def baseline_builders(seed: int) -> dict[str, Callable[[], Any]]:
    """Return the per-task baselines, each constructed fresh for every task.

    Args:
        seed: Seed passed to the estimators that accept one.

    Returns:
        Mapping from baseline name to a zero-argument constructor.
    """

    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    return {
        "logistic": lambda: LogisticRegression(max_iter=1000),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=seed),
        # Stands in for the prompt's "CatBoost or another strong GBDT". CatBoost is not a
        # dependency of this repository; scikit-learn's boosting is the same family.
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=seed),
    }


BASELINES = ("logistic", "random_forest", "gradient_boosting", "mlp")
"""Names of the per-task baselines, in the order they are reported."""


def evaluate_baseline(
    name: str,
    prior: TaskPrior,
    *,
    n_context: int,
    n_query: int,
    n_tasks: int,
    generator: torch.Generator,
    seed: int,
    missing_rate: float = 0.0,
) -> tuple[float, float]:
    """Fit a conventional model per task and score it on the same queries.

    Args:
        name: One of ``"logistic"``, ``"random_forest"``, ``"gradient_boosting"``, ``"mlp"``.
        prior: The task distribution.
        n_context: Context size per task.
        n_query: Queries per task.
        n_tasks: Tasks to sample.
        generator: Seeded generator.
        seed: Seed passed to the estimators.
        missing_rate: Fraction of feature entries blanked before fitting.

    Returns:
        ``(accuracy, expected_calibration_error)`` pooled over all queries.

    Raises:
        KeyError: If the baseline name is unknown.
    """

    batch = prior.sample(
        batch_size=n_tasks, n_context=n_context, n_query=n_query, generator=generator
    )
    context_inputs = blank_features(batch.context_inputs, missing_rate, generator)
    query_inputs = blank_features(batch.query_inputs, missing_rate, generator)

    builders = baseline_builders(seed)
    if name not in builders:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(builders)}")

    probabilities: list[Tensor] = []
    labels: list[Tensor] = []
    for task in range(n_tasks):
        context_x = context_inputs[task].numpy()
        context_y = batch.context_labels[task].numpy()
        query_x = query_inputs[task].numpy()

        if len(set(context_y.tolist())) < 2:
            # A degenerate context has one class only; scikit-learn cannot fit, and the
            # honest prediction is the constant it saw. Skipping would quietly drop the
            # hardest tasks from the baseline's score but not the PFN's.
            single = int(context_y[0])
            probs = torch.zeros(query_x.shape[0], 2)
            probs[:, single] = 1.0
        else:
            estimator = builders[name]()
            estimator.fit(context_x, context_y)
            probs = torch.as_tensor(estimator.predict_proba(query_x), dtype=torch.float32)

        probabilities.append(probs)
        labels.append(batch.query_labels[task])

    flat_probs = torch.cat(probabilities)
    flat_labels = torch.cat(labels)
    return (
        accuracy(flat_probs, flat_labels),
        expected_calibration_error(flat_probs, flat_labels),
    )


def run_comparison(
    settings: PFNExperimentConfig,
    output_dir: Path,
    *,
    fit_prior_name: str,
    eval_prior_name: str,
    n_context: int,
    label_noise: float = 0.0,
    missing_rate: float = 0.0,
    n_features: int | None = None,
    dataset: str | None = None,
) -> dict[str, RunGroup]:
    """Compare the PFN against per-task baselines on one evaluation distribution.

    Args:
        settings: Track settings.
        output_dir: Directory receiving records.
        fit_prior_name: Prior the PFN is fitted to.
        eval_prior_name: Prior the models are evaluated on. Differing from
            ``fit_prior_name`` is the out-of-prior test.
        n_context: Context size at evaluation time.
        label_noise: Label-flip probability of the evaluation prior.
        missing_rate: Fraction of feature entries blanked for every model alike.
        n_features: Input width; defaults to ``settings.n_features``. Changing it changes
            the model, so the PFN is prior-fitted again at that width.
        dataset: Dataset label; defaults to a descriptive one.

    Returns:
        Mapping from model name to :class:`RunGroup`.
    """

    width = n_features if n_features is not None else settings.n_features
    label = dataset or (
        f"fit-{fit_prior_name}__eval-{eval_prior_name}__ctx{n_context}__d{width}"
        + (f"__noise{label_noise:g}" if label_noise else "")
        + (f"__missing{missing_rate:g}" if missing_rate else "")
    )
    identity = fingerprint(
        {
            "fit_prior": fit_prior_name,
            "eval_prior": eval_prior_name,
            "n_context": n_context,
            "label_noise": label_noise,
            "missing_rate": missing_rate,
            "n_features": width,
            "eval_tasks": settings.eval_tasks,
            "positive_rate": settings.positive_rate,
        }
    )
    strategy = (
        f"{settings.eval_tasks} tasks sampled from the {eval_prior_name} prior; the PFN was "
        f"prior-fitted on {fit_prior_name}; baselines are fitted per task on the same "
        f"{n_context}-example context and scored on the same queries"
    )
    shared: dict[str, Any] = {
        "fit_prior": fit_prior_name,
        "eval_prior": eval_prior_name,
        "n_context": n_context,
        "label_noise": label_noise,
        "missing_rate": missing_rate,
        "n_features": width,
        "eval_tasks": settings.eval_tasks,
        "in_prior": fit_prior_name == eval_prior_name,
    }

    def evaluation_prior() -> TaskPrior:
        options: dict[str, float] = {}
        if eval_prior_name == "imbalanced":
            options["positive_rate"] = settings.positive_rate
        return build_prior(eval_prior_name, n_features=width, label_noise=label_noise, **options)

    groups: dict[str, RunGroup] = {}

    def pfn_evaluation(seed: int) -> tuple[float, dict[str, float], dict[str, Any]]:
        prior = build_prior(fit_prior_name, n_features=width)
        model, losses = fit_prior_cached(prior, settings, seed, n_features=width)
        generator = torch.Generator().manual_seed(10_000 + seed)
        score, calibration = evaluate_pfn(
            model,
            evaluation_prior(),
            n_context=n_context,
            n_query=settings.n_query,
            n_tasks=settings.eval_tasks,
            generator=generator,
            missing_rate=missing_rate,
        )
        return (
            score,
            {"expected_calibration_error": calibration, "prior_fitting_loss": losses[-1]},
            {**shared, **PFNConfig(n_features=width).as_dict()},
        )

    groups["pfn"] = run_meta_evaluation(
        pfn_evaluation,
        spec=RunSpec(
            track=TRACK,
            architecture="pfn",
            metric_name="query_accuracy",
            higher_is_better=True,
            architecture_version=ARCHITECTURE_VERSION,
            extra_config=shared,
            notes=(
                "Prior-fitted once, then one forward pass per task with no gradient step. "
                "Not comparable to a pre-trained checkpoint; none is used here."
            ),
        ),
        dataset=label,
        split_strategy=strategy,
        output_dir=output_dir,
        seeds=settings.seeds,
        dataset_fingerprint=identity,
        parameter_count=sum(
            p.numel() for p in PriorFittedNetwork(PFNConfig(n_features=width)).parameters()
        ),
    )

    for baseline in BASELINES:

        def baseline_evaluation(
            seed: int, name: str = baseline
        ) -> tuple[float, dict[str, float], dict[str, Any]]:
            generator = torch.Generator().manual_seed(10_000 + seed)
            score, calibration = evaluate_baseline(
                name,
                evaluation_prior(),
                n_context=n_context,
                n_query=settings.n_query,
                n_tasks=settings.eval_tasks,
                generator=generator,
                seed=seed,
                missing_rate=missing_rate,
            )
            return score, {"expected_calibration_error": calibration}, dict(shared)

        groups[baseline] = run_meta_evaluation(
            baseline_evaluation,
            spec=RunSpec(
                track=TRACK,
                architecture=baseline,
                metric_name="query_accuracy",
                higher_is_better=True,
                variant="fitted-per-task",
                architecture_version=ARCHITECTURE_VERSION,
                extra_config=shared,
                notes="Fitted from scratch on each task's context, the usual way.",
            ),
            dataset=label,
            split_strategy=strategy,
            output_dir=output_dir,
            seeds=settings.seeds,
            dataset_fingerprint=identity,
        )

    return groups


_PRIOR_FIT_CACHE: dict[tuple[str, int, int, int], tuple[PriorFittedNetwork, list[float]]] = {}


def fit_prior_cached(
    prior: TaskPrior, settings: PFNExperimentConfig, seed: int, *, n_features: int | None = None
) -> tuple[PriorFittedNetwork, list[float]]:
    """Prior-fit once per ``(prior, seed)`` and reuse across evaluation points.

    Prior fitting does not depend on the evaluation distribution or the evaluation context
    size, so refitting for every sweep point would multiply the cost without changing any
    model. Caching keeps the *same* fitted model across the in-prior, out-of-prior, and
    context-size evaluations, which is also what makes those comparisons meaningful.

    The input width is part of the key: a model fitted at four features cannot answer a
    sixteen-feature task, so sharing a cache entry across widths would silently hand back
    the wrong model.

    Args:
        prior: The fitting prior.
        settings: Track settings.
        seed: Seed.
        n_features: Input width; defaults to ``settings.n_features``.

    Returns:
        ``(model, loss_trajectory)``.
    """

    width = n_features if n_features is not None else settings.n_features
    key = (prior.name, settings.prior_fitting_steps, seed, width)
    if key not in _PRIOR_FIT_CACHE:
        _PRIOR_FIT_CACHE[key] = fit_prior(prior, settings, seed=seed, n_features=width)
    return _PRIOR_FIT_CACHE[key]


def measure_cost(settings: PFNExperimentConfig) -> dict[str, Any]:
    """Time prediction on new datasets, and the up-front cost the PFN pays for it.

    The comparison the track cares about is not raw latency but *where* the cost sits. The
    PFN pays once, before it has seen any of these datasets; a baseline pays again for
    every dataset. The break-even count is the honest way to state that trade-off, and it
    is computed rather than asserted.

    Args:
        settings: Track settings.

    Returns:
        A JSON-serializable summary in milliseconds per task.
    """

    import time

    prior = build_prior("linear", n_features=settings.n_features)
    # Deliberately *not* the cached fit: by this point the cache is warm from the
    # comparisons, and timing a dictionary lookup would report the up-front cost as free —
    # which is precisely the claim this measurement exists to check.
    fitting_start = time.perf_counter()
    model, _ = fit_prior(prior, settings, seed=0)
    prior_fitting_s = time.perf_counter() - fitting_start

    generator = torch.Generator().manual_seed(31_337)
    batch = prior.sample(
        batch_size=settings.cost_tasks,
        n_context=settings.train_context,
        n_query=settings.n_query,
        generator=generator,
    )

    start = time.perf_counter()
    model.predict_proba(batch.context_inputs, batch.context_labels, batch.query_inputs)
    pfn_ms = (time.perf_counter() - start) * 1000 / settings.cost_tasks

    per_task: dict[str, float] = {"pfn": pfn_ms}
    builders = baseline_builders(0)
    for name in BASELINES:
        start = time.perf_counter()
        for task in range(settings.cost_tasks):
            context_y = batch.context_labels[task].numpy()
            if len(set(context_y.tolist())) < 2:
                continue
            estimator = builders[name]()
            estimator.fit(batch.context_inputs[task].numpy(), context_y)
            estimator.predict_proba(batch.query_inputs[task].numpy())
        per_task[name] = (time.perf_counter() - start) * 1000 / settings.cost_tasks

    return {
        "note": (
            "Single seed, CPU, on a shared machine — these are order-of-magnitude figures, "
            "not benchmarks. The PFN's prior fitting is a one-time cost paid before any of "
            "these datasets existed; each baseline's cost recurs per dataset."
        ),
        "prior_fitting_s": prior_fitting_s,
        "prior_fitting_steps": settings.prior_fitting_steps,
        "tasks_timed": settings.cost_tasks,
        "per_task_ms": per_task,
        "break_even_tasks": {
            name: (prior_fitting_s * 1000) / max(per_task[name] - pfn_ms, 1e-9)
            for name in BASELINES
            if per_task[name] > pfn_ms
        },
    }


def write_prior_diagnostics(settings: PFNExperimentConfig, output_dir: Path) -> Path:
    """Serialize what changing the prior does, and where the model is confidently wrong.

    Args:
        settings: Track settings.
        output_dir: Directory receiving the artefact.

    Returns:
        Path of the written file.
    """

    rows: list[dict[str, Any]] = []
    for fit_name in ("linear", "mlp"):
        prior = build_prior(fit_name, n_features=settings.n_features)
        model, losses = fit_prior_cached(prior, settings, 0)
        for eval_name in ("linear", "mlp", "xor"):
            evaluation = build_prior(eval_name, n_features=settings.n_features)
            generator = torch.Generator().manual_seed(777)
            score, calibration = evaluate_pfn(
                model,
                evaluation,
                n_context=settings.train_context,
                n_query=settings.n_query,
                n_tasks=settings.eval_tasks,
                generator=generator,
            )
            rows.append(
                {
                    "fit_prior": fit_name,
                    "eval_prior": eval_name,
                    "in_prior": fit_name == eval_name,
                    "accuracy": score,
                    "expected_calibration_error": calibration,
                    "final_prior_fitting_loss": losses[-1],
                }
            )

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "track": TRACK,
        "measurement": "prior_transfer_matrix",
        "note": (
            "Single seed. Rows where fit_prior != eval_prior are out-of-prior: the model "
            "is being asked about a task family it was never fitted to. The calibration "
            "column is the point — an out-of-prior model that stays confident is worse "
            "than one that becomes uncertain."
        ),
        "tabpfn_reference": {
            "executed": False,
            "reason": (
                "TabPFN 8.2.0 gates its checkpoint behind interactive browser license "
                "acceptance, which cannot be completed non-interactively. No TabPFN "
                "number appears in this track."
            ),
            "adapter": "modern_nn_lab.tracks.pfn.reference",
            "package_importable": tabpfn_available(),
            "pretraining_advantage_note": pretraining_advantage_note(),
        },
        "transfer": rows,
        "cost": measure_cost(settings),
    }
    artefacts = output_dir / ARTEFACT_DIRNAME
    artefacts.mkdir(parents=True, exist_ok=True)
    path = artefacts / "prior_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return path


def run(output_dir: Path | str | None = None, *, quick: bool = False) -> None:
    """Run the whole PFN suite (deliverable A).

    Args:
        output_dir: Destination directory. Defaults to ``results/pfn``.
        quick: Reduced seeds, fitting steps, and evaluation tasks.
    """

    destination = Path(output_dir) if output_dir is not None else Path("results") / TRACK
    settings = (
        PFNExperimentConfig(
            seeds=(0, 1),
            prior_fitting_steps=40,
            eval_tasks=20,
            context_sweep=(5, 20),
            feature_counts=(4, 8),
            cost_tasks=10,
        )
        if quick
        else PFNExperimentConfig()
    )

    # 1. In prior: the family the model was fitted to.
    run_comparison(
        settings,
        destination,
        fit_prior_name="linear",
        eval_prior_name="linear",
        n_context=settings.train_context,
    )

    # 2. Out of prior: curved boundaries, then a family no linear rule can express.
    for eval_prior in ("mlp", "xor"):
        run_comparison(
            settings,
            destination,
            fit_prior_name="linear",
            eval_prior_name=eval_prior,
            n_context=settings.train_context,
        )

    # 3. Small-n: the regime where fitting per dataset should struggle most.
    for context in settings.context_sweep:
        if context == settings.train_context:
            continue
        run_comparison(
            settings,
            destination,
            fit_prior_name="linear",
            eval_prior_name="linear",
            n_context=context,
        )

    # 4. Label noise: irreducible error the model cannot fit away.
    run_comparison(
        settings,
        destination,
        fit_prior_name="linear",
        eval_prior_name="linear",
        n_context=settings.train_context,
        label_noise=settings.label_noise,
    )

    # 5. Class imbalance: the same separator, thresholded so one class is rare.
    run_comparison(
        settings,
        destination,
        fit_prior_name="linear",
        eval_prior_name="imbalanced",
        n_context=settings.train_context,
    )

    # 6. Missingness: identical blanked features for every model, so no model gets a
    # private imputation advantage.
    for rate in settings.missing_rates:
        run_comparison(
            settings,
            destination,
            fit_prior_name="linear",
            eval_prior_name="linear",
            n_context=settings.train_context,
            missing_rate=rate,
        )

    # 7. Feature count. Each width is a different model, prior-fitted separately, so this
    # sweep costs a full round of prior fitting per width.
    for width in settings.feature_counts:
        if width == settings.n_features:
            continue
        run_comparison(
            settings,
            destination,
            fit_prior_name="linear",
            eval_prior_name="linear",
            n_context=settings.train_context,
            n_features=width,
        )

    write_prior_diagnostics(settings, destination)
