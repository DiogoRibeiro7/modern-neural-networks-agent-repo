"""Experiment suite for Track 05 — Titans-style neural long-term memory.

The design is built around one manipulation: **the distance between writing a fact and
being asked for it**, relative to the short-term window. A model with only a sliding
window of size ``W`` should be fine inside ``W`` and helpless outside it; a working
long-term memory should close that gap. The *interaction* is the result, not either
accuracy on its own.

Ablations, covering the four the track prompt requires:

- ``short-term-only`` — no neural memory at all;
- ``frozen-memory`` — the memory exists and is read, but never written;
- ``no-momentum`` — equation 14 reduced to a bare gradient step (the source's eq. 8),
  removing past surprise;
- ``slow-updates`` — ``theta_t`` scaled by 0.1, i.e. an altered update rate.

The suite also writes a **memory-trace artefact**: per-token surprise, write magnitude,
and gate values, plus a forgetting curve. The track's acceptance criterion is explicit
write/read diagnostics rather than task accuracy alone, and that artefact is where they
live.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from torch import nn

from modern_nn_lab.experiments.records import ARTEFACT_DIRNAME, RESULT_SCHEMA_VERSION
from modern_nn_lab.experiments.runner import RunGroup, RunSpec, run_seeded_experiment
from modern_nn_lab.experiments.tasks.sequence import (
    SequenceSplit,
    make_needle_task,
    masked_accuracy,
    masked_cross_entropy,
)
from modern_nn_lab.experiments.training import TrainingConfig, train_supervised
from modern_nn_lab.models.sequence import (
    CausalTransformer,
    RecurrentBaseline,
    count_parameters,
    match_width_to_budget,
)
from modern_nn_lab.reproducibility import seed_everything
from modern_nn_lab.tracks.titans import TitansConfig, TitansExperimentConfig, TitansMAG

TRACK = "titans"
ARCHITECTURE_VERSION = "0.1.0"

ModelFactory = Callable[[int], nn.Module]


def _training_config(settings: TitansExperimentConfig, learning_rate: float) -> TrainingConfig:
    """Translate the track settings into the shared optimizer configuration."""

    return TrainingConfig(
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=learning_rate,
        eval_every=max(1, settings.epochs // 5),
        cosine_schedule=True,
    )


def select_learning_rate(
    factory: ModelFactory,
    split: SequenceSplit,
    *,
    settings: TitansExperimentConfig,
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Pick a learning rate on the validation split, using the same grid for everyone.

    Args:
        factory: Builds a fresh model for a given seed.
        split: Task split; only train and validation are used.
        settings: Track settings.
        seed: Seed used for every trial.

    Returns:
        ``(best_learning_rate, {rate: validation_accuracy})``.
    """

    scores: dict[str, float] = {}
    best_rate = settings.learning_rate_grid[0]
    best_score = -1.0

    for rate in settings.learning_rate_grid:
        outcome = train_supervised(
            factory(seed),
            split.train_inputs,
            split.train_targets,
            config=_training_config(settings, rate),
            loss_fn=masked_cross_entropy,
            eval_inputs=split.val_inputs,
            eval_targets=split.val_targets,
            metric_fn=masked_accuracy,
            higher_is_better=True,
        )
        score = outcome.final_metric if outcome.status == "success" else -1.0
        scores[f"{rate:g}"] = score
        if score > best_score:
            best_score, best_rate = score, rate

    return best_rate, scores


def _run_model(
    factory: ModelFactory,
    split: SequenceSplit,
    *,
    architecture: str,
    variant: str | None,
    settings: TitansExperimentConfig,
    output_dir: Path,
    extra_config: dict[str, object],
    notes: str | None = None,
) -> RunGroup:
    """Run one architecture on one task with the shared protocol."""

    learning_rate, lr_scores = select_learning_rate(
        factory, split, settings=settings, seed=settings.seeds[0]
    )
    return run_seeded_experiment(
        factory,
        split,
        spec=RunSpec(
            track=TRACK,
            architecture=architecture,
            metric_name="test_accuracy",
            higher_is_better=True,
            variant=variant,
            architecture_version=ARCHITECTURE_VERSION,
            extra_config={
                **extra_config,
                **split.metadata,
                "learning_rate_grid": list(settings.learning_rate_grid),
                "learning_rate_validation_accuracy": lr_scores,
                "selected_learning_rate": learning_rate,
            },
            notes=notes,
        ),
        training_config=_training_config(settings, learning_rate),
        loss_fn=masked_cross_entropy,
        metric_fn=masked_accuracy,
        output_dir=output_dir,
        seeds=settings.seeds,
        profile_inference_cost=True,
        eval_batch_size=128,
    )


def build_model_zoo(
    split: SequenceSplit, settings: TitansExperimentConfig
) -> tuple[dict[str, tuple[str, str | None, ModelFactory, str | None]], dict[str, object]]:
    """Construct every architecture for one task, matched to the Titans parameter budget.

    Args:
        split: Task split, which fixes the vocabulary size.
        settings: Track settings.

    Returns:
        ``(zoo, shared_config)`` mapping a label to
        ``(architecture, variant, factory, notes)``.
    """

    vocab = split.vocab_size
    base = TitansConfig(d_model=settings.d_model, window=settings.window)

    def titans_factory(config: TitansConfig) -> ModelFactory:
        def factory(seed: int) -> nn.Module:
            seed_everything(seed)
            return TitansMAG(vocab, config)

        return factory

    budget = count_parameters(TitansMAG(vocab, base))
    lstm_width = match_width_to_budget(
        budget, lambda w: RecurrentBaseline(vocab, d_model=w, n_layers=1, kind="lstm")
    )
    transformer_width = match_width_to_budget(
        budget,
        lambda w: CausalTransformer(
            vocab, d_model=w, n_layers=1, n_heads=2, max_len=max(512, split.seq_len)
        ),
    )

    def lstm_factory(seed: int) -> nn.Module:
        seed_everything(seed)
        return RecurrentBaseline(vocab, d_model=lstm_width, n_layers=1, kind="lstm")

    def transformer_factory(seed: int) -> nn.Module:
        seed_everything(seed)
        return CausalTransformer(
            vocab,
            d_model=transformer_width,
            n_layers=1,
            n_heads=2,
            max_len=max(512, split.seq_len),
        )

    zoo: dict[str, tuple[str, str | None, ModelFactory, str | None]] = {
        "titans": ("titans", None, titans_factory(base), None),
        "short_only": (
            "titans",
            "short-term-only",
            titans_factory(replace(base, use_long_term=False)),
            "Ablation: no neural memory. Sliding-window attention alone, so the model can "
            "only answer when the fact lies inside its window.",
        ),
        "frozen": (
            "titans",
            "frozen-memory",
            titans_factory(replace(base, memory_updates=False)),
            "Ablation: the memory is read but never written.",
        ),
        "no_momentum": (
            "titans",
            "no-momentum",
            titans_factory(replace(base, use_momentum=False)),
            "Ablation: equation 14 reduced to a bare gradient step (the source's "
            "equation 8), removing past surprise.",
        ),
        "lstm": ("lstm", "matched-parameters", lstm_factory, None),
        "transformer": ("transformer", "matched-parameters", transformer_factory, None),
    }

    shared: dict[str, object] = {
        "parameter_budget": budget,
        **base.as_dict(),
        "lstm_d_model": lstm_width,
        "transformer_d_model": transformer_width,
    }
    return zoo, shared


def run_distance(
    split: SequenceSplit,
    settings: TitansExperimentConfig,
    output_dir: Path,
    *,
    labels: tuple[str, ...] | None = None,
) -> dict[str, RunGroup]:
    """Run architectures on one needle distance.

    Args:
        split: The task split.
        settings: Track settings.
        output_dir: Directory receiving records.
        labels: Restrict to these zoo labels.

    Returns:
        Mapping from label to :class:`RunGroup`.
    """

    zoo, shared = build_model_zoo(split, settings)
    selected = zoo if labels is None else {label: zoo[label] for label in labels}
    return {
        label: _run_model(
            factory,
            split,
            architecture=architecture,
            variant=variant,
            settings=settings,
            output_dir=output_dir,
            extra_config=shared,
            notes=notes,
        )
        for label, (architecture, variant, factory, notes) in selected.items()
    }


def write_memory_diagnostics(
    settings: TitansExperimentConfig, output_dir: Path, *, quick: bool = False
) -> Path:
    """Train one model and serialize what its memory actually did.

    Records three things task accuracy cannot show:

    1. **Per-token traces** — surprise, write magnitude, and the three gates, so a write
       can be located in the sequence rather than inferred from the answer.
    2. **A forgetting curve** — accuracy against write-to-query distance, for the full
       model and for the short-term-only ablation. The gap between the two curves *is*
       the long-term memory's contribution.
    3. **Repeated versus one-off facts** — whether writing the same association twice
       changes the surprise the second time, which is what the source's surprise metric
       implies it should.

    Args:
        settings: Track settings.
        output_dir: Directory receiving the artefact.
        quick: Use a reduced sweep.

    Returns:
        Path of the written file.
    """

    pairs = 5
    distances = (0, 2, 4) if quick else tuple(range(pairs))
    reference = make_needle_task(
        n_sequences=settings.n_sequences,
        n_pairs=pairs,
        needle_index=0,
        n_keys=10,
        n_values=6,
        seed=1729,
    )
    base = TitansConfig(d_model=settings.d_model, window=settings.window)

    def train_on(split: SequenceSplit, config: TitansConfig) -> tuple[TitansMAG, float]:
        seed_everything(0)
        model = TitansMAG(split.vocab_size, config)
        train_supervised(
            model,
            split.train_inputs,
            split.train_targets,
            config=_training_config(settings, settings.learning_rate_grid[-1]),
            loss_fn=masked_cross_entropy,
        )
        from modern_nn_lab.experiments.training import evaluate

        return model, evaluate(model, split.test_inputs, split.test_targets, masked_accuracy)

    curve: list[dict[str, object]] = []
    for index in distances:
        split = make_needle_task(
            n_sequences=settings.n_sequences,
            n_pairs=pairs,
            needle_index=index,
            n_keys=10,
            n_values=6,
            seed=1729,
        )
        _, full = train_on(split, base)
        _, short = train_on(split, replace(base, use_long_term=False))
        curve.append(
            {
                "needle_index": index,
                "distance": split.metadata["distance"],
                "accuracy_with_memory": full,
                "accuracy_short_term_only": short,
                "chance": split.metadata["chance_accuracy"],
            }
        )

    model, _ = train_on(reference, base)
    sample = reference.test_inputs[: settings.diagnostic_sequences]
    trace = model.memory_trace(sample)

    repeated = make_needle_task(
        n_sequences=settings.n_sequences,
        n_pairs=pairs,
        needle_index=0,
        n_keys=10,
        n_values=6,
        repeats=2,
        seed=1729,
        name="needle-repeated",
    )
    repeated_trace = model.memory_trace(repeated.test_inputs[: settings.diagnostic_sequences])

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "track": TRACK,
        "measurement": "memory_write_read_diagnostics",
        "window": settings.window,
        "persistent_tokens": base.persistent_tokens,
        "note": (
            "Traces are averaged over the batch at each token position. Positions 0..N_p-1 "
            "are the persistent-memory prefix, which is prepended before the real tokens."
        ),
        "forgetting_curve": curve,
        "trace_one_off": trace.as_dict(),
        "trace_repeated": repeated_trace.as_dict(),
    }
    artefacts = output_dir / ARTEFACT_DIRNAME
    artefacts.mkdir(parents=True, exist_ok=True)
    path = artefacts / "memory_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run(output_dir: Path | str | None = None, *, quick: bool = False) -> None:
    """Run the whole Titans suite and write every record under ``output_dir``.

    Args:
        output_dir: Destination directory. Defaults to ``results/titans``.
        quick: Reduced seeds, epochs, and task sizes for smoke-testing the protocol.
    """

    destination = Path(output_dir) if output_dir is not None else Path("results") / TRACK
    settings = (
        TitansExperimentConfig(
            seeds=(0, 1),
            epochs=3,
            n_sequences=120,
            d_model=16,
            learning_rate_grid=(3e-3,),
            diagnostic_sequences=8,
        )
        if quick
        else TitansExperimentConfig()
    )

    pairs = 5
    # needle_index 0 puts the fact outside the short-term window; the last index puts it
    # inside. The contrast between the two is the experiment.
    for needle_index in (0, pairs - 1):
        split = make_needle_task(
            n_sequences=settings.n_sequences,
            n_pairs=pairs,
            needle_index=needle_index,
            n_keys=10,
            n_values=6,
            seed=1729,
        )
        labels = (
            ("titans", "short_only", "frozen", "no_momentum", "lstm", "transformer")
            if needle_index == 0
            else ("titans", "short_only", "frozen", "lstm", "transformer")
        )
        run_distance(split, settings, destination, labels=labels)

    # The update-rate ablation, run only where distance makes memory necessary.
    far = make_needle_task(
        n_sequences=settings.n_sequences,
        n_pairs=pairs,
        needle_index=0,
        n_keys=10,
        n_values=6,
        seed=1729,
    )
    _, shared = build_model_zoo(far, settings)
    base = TitansConfig(d_model=settings.d_model, window=settings.window)

    def slow_factory(seed: int) -> nn.Module:
        seed_everything(seed)
        return TitansMAG(far.vocab_size, replace(base, learning_rate_scale=0.1))

    _run_model(
        slow_factory,
        far,
        architecture="titans",
        variant="slow-updates",
        settings=settings,
        output_dir=destination,
        extra_config={**shared, "learning_rate_scale": 0.1},
        notes="Ablation: theta_t scaled by 0.1, i.e. a ten-fold slower write rate.",
    )

    write_memory_diagnostics(settings, destination, quick=quick)
