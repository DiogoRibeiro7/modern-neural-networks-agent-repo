"""Experiment suite for Track 02 — xLSTM.

Protocol:

1. **Three diagnostics, each isolating a different demand on memory** — copy (retain),
   selective recall (retrieve one of many), state tracking (update a carried state at
   every step).
2. **Matched-parameter baselines** — LSTM, GRU, and a causal Transformer, each with its
   width searched so its parameter count matches the xLSTM's. Recurrent and attentive
   bodies differ by several times at equal width, so equal width would compare capacity.
3. **The gating ablation** — the identical model with ``input_gate="sigmoid"``. Nothing
   else changes: same normalizer, same stabilizer, same matrix memory, same blocks. Any
   difference is attributable to the gate's ceiling alone.
4. **A second architectural variant** — sLSTM blocks (scalar memory, memory mixing)
   against mLSTM blocks (matrix memory, no mixing).
5. **Context scaling** — state tracking at increasing sequence length, to separate
   "solves the task" from "solves the task at short range".
6. **Per-architecture learning-rate selection on validation**, so no model is reported at
   a rate that suits a different architecture. The test split is scored once.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from torch import nn

from modern_nn_lab.experiments.evaluation import aggregate_values
from modern_nn_lab.experiments.runner import RunGroup, RunSpec, run_seeded_experiment
from modern_nn_lab.experiments.tasks.sequence import (
    SequenceSplit,
    make_copy_task,
    make_selective_recall_task,
    make_state_tracking_task,
    masked_accuracy,
    masked_cross_entropy,
)
from modern_nn_lab.experiments.training import TrainingConfig, train_supervised
from modern_nn_lab.reproducibility import seed_everything
from modern_nn_lab.tracks.xlstm import (
    XLSTM,
    CausalTransformer,
    RecurrentBaseline,
    XLSTMExperimentConfig,
    count_parameters,
    match_width_to_budget,
)

TRACK = "xlstm"
ARCHITECTURE_VERSION = "0.1.0"

ModelFactory = Callable[[int], nn.Module]


def _training_config(settings: XLSTMExperimentConfig, learning_rate: float) -> TrainingConfig:
    """Translate the track settings into the shared optimizer configuration."""

    return TrainingConfig(
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        learning_rate=learning_rate,
        eval_every=max(1, settings.epochs // 6),
        cosine_schedule=True,
    )


def select_learning_rate(
    factory: ModelFactory,
    split: SequenceSplit,
    *,
    settings: XLSTMExperimentConfig,
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Pick a learning rate on the **validation** split, using the same grid for everyone.

    Args:
        factory: Builds a fresh model for a given seed.
        split: The task split; only train and validation are used.
        settings: Track settings supplying the grid and epoch budget.
        seed: Seed used for every trial, so trials differ only in learning rate.

    Returns:
        ``(best_learning_rate, {rate: validation_accuracy})``.
    """

    scores: dict[str, float] = {}
    best_rate = settings.learning_rate_grid[0]
    best_score = -1.0

    for rate in settings.learning_rate_grid:
        model = factory(seed)
        outcome = train_supervised(
            model,
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
    settings: XLSTMExperimentConfig,
    output_dir: Path,
    extra_config: dict[str, object],
    notes: str | None = None,
    seeds: tuple[int, ...] | None = None,
    learning_rate: float | None = None,
) -> RunGroup:
    """Run one architecture on one task with the shared protocol."""

    run_seeds = seeds if seeds is not None else settings.seeds

    if learning_rate is None:
        learning_rate, lr_scores = select_learning_rate(
            factory, split, settings=settings, seed=run_seeds[0]
        )
        selection = "validation grid search"
    else:
        lr_scores = {}
        selection = "inherited"

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
                "learning_rate_selection": selection,
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
        seeds=run_seeds,
        profile_inference_cost=True,
        eval_batch_size=128,
    )


def build_model_zoo(
    split: SequenceSplit, settings: XLSTMExperimentConfig
) -> tuple[dict[str, tuple[str, str | None, ModelFactory, str | None]], dict[str, object]]:
    """Construct every architecture for one task, matched to the xLSTM parameter budget.

    Args:
        split: The task split, which fixes the vocabulary size.
        settings: Track settings.

    Returns:
        ``(zoo, shared_config)`` where ``zoo`` maps a label to
        ``(architecture, variant, factory, notes)``.
    """

    vocab = split.vocab_size
    width, blocks, heads = settings.d_model, settings.n_blocks, 2

    def make_xlstm(seed: int, *, gate: str = "exponential", kinds: object = None) -> nn.Module:
        seed_everything(seed)
        return XLSTM(
            vocab,
            d_model=width,
            n_blocks=blocks,
            block_kinds=kinds,  # type: ignore[arg-type]
            heads=heads,
            input_gate=gate,  # type: ignore[arg-type]
        )

    budget = count_parameters(make_xlstm(0))

    lstm_width = match_width_to_budget(
        budget, lambda w: RecurrentBaseline(vocab, d_model=w, n_layers=blocks, kind="lstm")
    )
    gru_width = match_width_to_budget(
        budget, lambda w: RecurrentBaseline(vocab, d_model=w, n_layers=blocks, kind="gru")
    )
    transformer_width = match_width_to_budget(
        budget,
        lambda w: CausalTransformer(
            vocab, d_model=w, n_layers=blocks, n_heads=2, max_len=max(512, split.seq_len)
        ),
    )

    def make_recurrent(kind: str, model_width: int) -> ModelFactory:
        def factory(seed: int) -> nn.Module:
            seed_everything(seed)
            return RecurrentBaseline(
                vocab,
                d_model=model_width,
                n_layers=blocks,
                kind=kind,  # type: ignore[arg-type]
            )

        return factory

    def make_transformer(seed: int) -> nn.Module:
        seed_everything(seed)
        return CausalTransformer(
            vocab,
            d_model=transformer_width,
            n_layers=blocks,
            n_heads=2,
            max_len=max(512, split.seq_len),
        )

    zoo: dict[str, tuple[str, str | None, ModelFactory, str | None]] = {
        "xlstm": ("xlstm", None, make_xlstm, None),
        "xlstm_sigmoid": (
            "xlstm",
            "sigmoid-input-gate",
            lambda seed: make_xlstm(seed, gate="sigmoid"),
            "Ablation: the exponential input gate replaced by a sigmoid; nothing else changes.",
        ),
        "xlstm_slstm": (
            "xlstm",
            "slstm-blocks",
            lambda seed: make_xlstm(seed, kinds=tuple(["slstm"] * blocks)),
            "Variant: scalar memory with memory mixing instead of matrix memory.",
        ),
        "lstm": ("lstm", "matched-parameters", make_recurrent("lstm", lstm_width), None),
        "gru": ("gru", "matched-parameters", make_recurrent("gru", gru_width), None),
        "transformer": ("transformer", "matched-parameters", make_transformer, None),
    }

    shared: dict[str, object] = {
        "parameter_budget": budget,
        "xlstm_d_model": width,
        "n_blocks": blocks,
        "lstm_d_model": lstm_width,
        "gru_d_model": gru_width,
        "transformer_d_model": transformer_width,
    }
    return zoo, shared


def run_task(
    split: SequenceSplit, settings: XLSTMExperimentConfig, output_dir: Path
) -> dict[str, RunGroup]:
    """Run every architecture on one diagnostic task.

    Args:
        split: The task split.
        settings: Track settings.
        output_dir: Directory receiving records.

    Returns:
        Mapping from label to :class:`RunGroup`.
    """

    zoo, shared = build_model_zoo(split, settings)
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
        for label, (architecture, variant, factory, notes) in zoo.items()
    }


def run_context_scaling(
    settings: XLSTMExperimentConfig, output_dir: Path
) -> dict[int, dict[str, RunGroup]]:
    """Measure state tracking as the sequence length grows.

    Accuracy at one length cannot distinguish a model that carries state from one that
    exploits a short-range shortcut. The scaling curve can.

    Args:
        settings: Track settings.
        output_dir: Directory receiving records.

    Returns:
        Mapping from sequence length to the per-architecture run groups.
    """

    results: dict[int, dict[str, RunGroup]] = {}
    for seq_len in settings.context_lengths:
        split = make_state_tracking_task(
            n_sequences=settings.n_sequences, seq_len=seq_len, n_states=2, seed=1729
        )
        zoo, shared = build_model_zoo(split, settings)
        selected = {key: zoo[key] for key in ("xlstm", "xlstm_sigmoid", "lstm")}
        results[seq_len] = {
            label: _run_model(
                factory,
                split,
                architecture=architecture,
                variant=f"{variant or 'default'}@T{seq_len}",
                settings=settings,
                output_dir=output_dir,
                extra_config={**shared, "study": "context_scaling", "context_length": seq_len},
                notes=notes,
                seeds=settings.scaling_seeds,
            )
            for label, (architecture, variant, factory, notes) in selected.items()
        }
    return results


def run(output_dir: Path | str | None = None, *, quick: bool = False) -> None:
    """Run the whole xLSTM suite and write every record under ``output_dir``.

    Args:
        output_dir: Destination directory. Defaults to ``results/xlstm``.
        quick: Reduced seeds, epochs, and task sizes for smoke-testing the protocol.
            Records carry the reduced settings and must not be reported.
    """

    destination = Path(output_dir) if output_dir is not None else Path("results") / TRACK
    settings = (
        XLSTMExperimentConfig(
            seeds=(0, 1),
            scaling_seeds=(0,),
            epochs=3,
            n_sequences=120,
            d_model=16,
            learning_rate_grid=(3e-3,),
            context_lengths=(8, 16),
        )
        if quick
        else XLSTMExperimentConfig()
    )

    copy_split = make_copy_task(
        n_sequences=settings.n_sequences,
        payload_len=3 if quick else 4,
        delay=4 if quick else 8,
        n_symbols=8,
        seed=1729,
    )
    recall_split = make_selective_recall_task(
        n_sequences=settings.n_sequences,
        n_pairs=2 if quick else 4,
        n_keys=8,
        n_values=8,
        seed=1729,
    )
    tracking_split = make_state_tracking_task(
        n_sequences=settings.n_sequences,
        seq_len=8 if quick else 16,
        n_states=2,
        seed=1729,
    )

    for split in (copy_split, recall_split, tracking_split):
        run_task(split, settings, destination)

    run_context_scaling(settings, destination)


def summarize(groups: dict[str, RunGroup]) -> str:
    """Render a compact human-readable summary of one task's run groups.

    Args:
        groups: Mapping from label to run group.

    Returns:
        One line per label.
    """

    lines = []
    for label, group in groups.items():
        if not group.successful:
            lines.append(f"{label}: all seeds failed")
            continue
        aggregate = aggregate_values(
            "test_accuracy",
            [record.primary_metric.value for record in group.successful],
            higher_is_better=True,
        )
        lines.append(f"{label}: {aggregate.format()}")
    return "\n".join(lines)
