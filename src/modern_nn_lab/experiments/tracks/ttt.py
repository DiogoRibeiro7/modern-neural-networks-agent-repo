"""Experiment suite for Track 04 — Test-Time Training.

Protocol:

1. **The diagnostic the mechanism is for.** A TTT layer's selling point is that its state
   keeps *learning* while it reads. The rebinding task overwrites a key-value binding
   halfway through a sequence, so answering the second query requires revising something
   already stored. The headline metric is accuracy on the **post-shift answer only**; the
   pre-shift answer is ordinary associative recall, which the selective-recall task
   already measures.
2. **The required ablation.** The identical architecture with ``learner_updates=False``.
   Nothing else changes — same parameter count, same views, same ``W_0`` — so any
   difference is attributable to the inner loop alone.
3. **A second ablation taken from the source.** Batch gradient descent instead of online:
   the source proves this instance *is* linear attention, so it separates "the state is a
   learner" from "the state is updated online".
4. **Matched-parameter baselines** — LSTM, GRU, and a causal Transformer.
5. **Context scaling** on the same task, by increasing the number of bindings that must be
   revised.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor, nn

from modern_nn_lab.experiments.runner import RunGroup, RunSpec, run_seeded_experiment
from modern_nn_lab.experiments.tasks.sequence import (
    IGNORE_INDEX,
    SequenceSplit,
    make_rebinding_task,
    make_selective_recall_task,
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
from modern_nn_lab.tracks.ttt import TTT, TTTConfig, TTTExperimentConfig

TRACK = "ttt"
ARCHITECTURE_VERSION = "0.1.0"

ModelFactory = Callable[[int], nn.Module]
MetricFn = Callable[[Tensor, Tensor], float]


def _metadata_index(split: SequenceSplit, key: str) -> int:
    """Read an integer position out of a split's metadata.

    Args:
        split: The task split.
        key: Metadata key naming a sequence position.

    Returns:
        The position as an integer.

    Raises:
        KeyError: If the split does not carry that key.
        TypeError: If the value is not an integer.
    """

    value = split.metadata[key]
    if not isinstance(value, int):
        raise TypeError(f"metadata {key!r} must be an int, got {type(value).__name__}")
    return value


def positional_accuracy(index: int) -> MetricFn:
    """Return a metric scoring one sequence position only.

    Args:
        index: Position to score.

    Returns:
        A metric callable over ``(logits, targets)``.
    """

    def metric(logits: Tensor, targets: Tensor) -> float:
        selected = torch.full_like(targets, IGNORE_INDEX)
        selected[:, index] = targets[:, index]
        return masked_accuracy(logits, selected)

    return metric


def _training_config(settings: TTTExperimentConfig, learning_rate: float) -> TrainingConfig:
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
    settings: TTTExperimentConfig,
    metric_fn: MetricFn,
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Pick a learning rate on the validation split, using the same grid for everyone.

    Args:
        factory: Builds a fresh model for a given seed.
        split: Task split; only train and validation are used.
        settings: Track settings supplying the grid and epoch budget.
        metric_fn: Metric used to rank the trials.
        seed: Seed used for every trial.

    Returns:
        ``(best_learning_rate, {rate: validation_score})``.
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
            metric_fn=metric_fn,
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
    settings: TTTExperimentConfig,
    output_dir: Path,
    extra_config: dict[str, object],
    metric_fn: MetricFn,
    metric_name: str,
    notes: str | None = None,
    seeds: tuple[int, ...] | None = None,
) -> RunGroup:
    """Run one architecture on one task with the shared protocol."""

    run_seeds = seeds if seeds is not None else settings.seeds
    learning_rate, lr_scores = select_learning_rate(
        factory, split, settings=settings, metric_fn=metric_fn, seed=run_seeds[0]
    )

    return run_seeded_experiment(
        factory,
        split,
        spec=RunSpec(
            track=TRACK,
            architecture=architecture,
            metric_name=metric_name,
            higher_is_better=True,
            variant=variant,
            architecture_version=ARCHITECTURE_VERSION,
            extra_config={
                **extra_config,
                **split.metadata,
                "learning_rate_grid": list(settings.learning_rate_grid),
                "learning_rate_validation_score": lr_scores,
                "selected_learning_rate": learning_rate,
            },
            notes=notes,
        ),
        training_config=_training_config(settings, learning_rate),
        loss_fn=masked_cross_entropy,
        metric_fn=metric_fn,
        output_dir=output_dir,
        seeds=run_seeds,
        profile_inference_cost=True,
        eval_batch_size=128,
    )


def build_model_zoo(
    split: SequenceSplit, settings: TTTExperimentConfig
) -> tuple[dict[str, tuple[str, str | None, ModelFactory, str | None]], dict[str, object]]:
    """Construct every architecture for one task, matched to the TTT parameter budget.

    Args:
        split: Task split, which fixes the vocabulary size.
        settings: Track settings.

    Returns:
        ``(zoo, shared_config)`` mapping a label to
        ``(architecture, variant, factory, notes)``.
    """

    vocab = split.vocab_size
    blocks = settings.n_blocks
    base = TTTConfig(d_model=settings.d_model, n_blocks=blocks)

    def ttt_factory(config: TTTConfig) -> ModelFactory:
        def factory(seed: int) -> nn.Module:
            seed_everything(seed)
            return TTT(vocab, config)

        return factory

    budget = count_parameters(TTT(vocab, base))

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

    def recurrent_factory(kind: str, width: int) -> ModelFactory:
        def factory(seed: int) -> nn.Module:
            seed_everything(seed)
            return RecurrentBaseline(vocab, d_model=width, n_layers=blocks, kind=kind)  # type: ignore[arg-type]

        return factory

    def transformer_factory(seed: int) -> nn.Module:
        seed_everything(seed)
        return CausalTransformer(
            vocab,
            d_model=transformer_width,
            n_layers=blocks,
            n_heads=2,
            max_len=max(512, split.seq_len),
        )

    zoo: dict[str, tuple[str, str | None, ModelFactory, str | None]] = {
        "ttt_linear": ("ttt", None, ttt_factory(base), None),
        "ttt_mlp": (
            "ttt",
            "ttt-mlp",
            ttt_factory(replace(base, inner_model="mlp")),
            "Inner model is a two-layer GELU MLP with 4x hidden width.",
        ),
        "ttt_frozen": (
            "ttt",
            "frozen-learner",
            ttt_factory(replace(base, learner_updates=False)),
            "Required ablation: the inner loop is disabled, so W_t = W_0 for every t. "
            "Same parameter count and same views as the default.",
        ),
        "ttt_batch": (
            "ttt",
            "batch-gradient-descent",
            ttt_factory(replace(base, update_rule="batch")),
            "Ablation: every inner gradient is taken at W_0 rather than W_{t-1}. The "
            "source proves this instance is linear attention.",
        ),
        "lstm": ("lstm", "matched-parameters", recurrent_factory("lstm", lstm_width), None),
        "gru": ("gru", "matched-parameters", recurrent_factory("gru", gru_width), None),
        "transformer": ("transformer", "matched-parameters", transformer_factory, None),
    }

    shared: dict[str, object] = {
        "parameter_budget": budget,
        **base.as_dict(),
        "lstm_d_model": lstm_width,
        "gru_d_model": gru_width,
        "transformer_d_model": transformer_width,
    }
    return zoo, shared


def run_task(
    split: SequenceSplit,
    settings: TTTExperimentConfig,
    output_dir: Path,
    *,
    metric_fn: MetricFn,
    metric_name: str,
    labels: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, RunGroup]:
    """Run architectures on one task.

    Args:
        split: The task split.
        settings: Track settings.
        output_dir: Directory receiving records.
        metric_fn: Primary metric.
        metric_name: Name recorded for the primary metric.
        labels: Restrict to these zoo labels. ``None`` runs all of them.
        seeds: Override the seed budget.
        extra: Extra configuration merged into every record.

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
            extra_config={**shared, **(extra or {})},
            metric_fn=metric_fn,
            metric_name=metric_name,
            notes=notes,
            seeds=seeds,
        )
        for label, (architecture, variant, factory, notes) in selected.items()
    }


def run(output_dir: Path | str | None = None, *, quick: bool = False) -> None:
    """Run the whole TTT suite and write every record under ``output_dir``.

    Args:
        output_dir: Destination directory. Defaults to ``results/ttt``.
        quick: Reduced seeds, epochs, and task sizes for smoke-testing the protocol.
    """

    destination = Path(output_dir) if output_dir is not None else Path("results") / TRACK
    settings = (
        TTTExperimentConfig(
            seeds=(0, 1), epochs=3, n_sequences=120, d_model=16, learning_rate_grid=(3e-3,)
        )
        if quick
        else TTTExperimentConfig()
    )

    # 1. The mechanism's own diagnostic: adaptation to a mid-sequence rule change.
    rebinding = make_rebinding_task(
        n_sequences=settings.n_sequences,
        n_pairs=settings.rebinding_pairs,
        n_keys=6,
        n_values=6,
        seed=1729,
    )
    post_shift = _metadata_index(rebinding, "second_answer_index")
    run_task(
        rebinding,
        settings,
        destination,
        metric_fn=positional_accuracy(post_shift),
        metric_name="test_accuracy_post_shift",
    )

    # 2. Ordinary associative recall, identical to Tracks 02 and 03 for comparability.
    recall = make_selective_recall_task(
        n_sequences=settings.n_sequences,
        n_pairs=2 if quick else 4,
        n_keys=8,
        n_values=8,
        seed=1729,
    )
    run_task(
        recall,
        settings,
        destination,
        metric_fn=masked_accuracy,
        metric_name="test_accuracy",
    )

    # 3. Context scaling: more bindings to revise means a longer sequence and more
    #    interference between the two phases.
    for pairs in (2, 4) if quick else (5,):
        scaled = make_rebinding_task(
            n_sequences=settings.n_sequences,
            n_pairs=pairs,
            n_keys=8,
            n_values=6,
            seed=1729,
            name=f"rebinding-{pairs}pairs",
        )
        index = _metadata_index(scaled, "second_answer_index")
        run_task(
            scaled,
            settings,
            destination,
            metric_fn=positional_accuracy(index),
            metric_name="test_accuracy_post_shift",
            labels=("ttt_linear", "ttt_frozen", "lstm"),
            seeds=settings.seeds[:3],
            extra={"study": "context_scaling", "n_pairs": pairs},
        )
