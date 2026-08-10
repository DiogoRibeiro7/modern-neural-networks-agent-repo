"""Experiment suite for Track 03 — Mamba-3 / modern state-space models.

Protocol:

1. **The same three diagnostics as Track 02, on identical data.** The task splits are
   generated with the same parameters and the same seed, so this track's records and the
   xLSTM track's records are directly comparable without rerunning anything.
2. **Ablations that recover prior methods exactly.** Each of the source's three
   contributions has a flag, and turning it off returns the model to what came before:
   ``trapezoidal=False`` is Mamba-1/2's exponential-Euler rule, ``rotary=False`` is a real
   non-negative transition, ``rank=1`` is SISO.
3. **A pre-registered prediction.** The source motivates complex-valued states by noting
   that real transitions cannot represent parity. State tracking *is* parity, so the
   rotary ablation should fail it while the full model should not. This is the sharpest
   test in the suite precisely because it can come out wrong.
4. **Matched-parameter baselines** — LSTM, GRU, and a causal Transformer, widths searched
   to match the SSM's parameter count.
5. **Cost scaling with sequence length**, measured rather than asserted, and reported
   with an explicit warning that a Python scan is not a kernel.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from modern_nn_lab.experiments.profiling import profile_inference, profile_parameters
from modern_nn_lab.experiments.records import ARTEFACT_DIRNAME, RESULT_SCHEMA_VERSION
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
from modern_nn_lab.models.sequence import (
    CausalTransformer,
    RecurrentBaseline,
    count_parameters,
    match_width_to_budget,
)
from modern_nn_lab.reproducibility import seed_everything
from modern_nn_lab.tracks.mamba3 import Mamba3, Mamba3Config, Mamba3ExperimentConfig

TRACK = "mamba3"
ARCHITECTURE_VERSION = "0.1.0"

ModelFactory = Callable[[int], nn.Module]


def _training_config(settings: Mamba3ExperimentConfig, learning_rate: float) -> TrainingConfig:
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
    settings: Mamba3ExperimentConfig,
    seed: int,
) -> tuple[float, dict[str, float]]:
    """Pick a learning rate on the validation split, using the same grid for everyone.

    Args:
        factory: Builds a fresh model for a given seed.
        split: Task split; only train and validation are used.
        settings: Track settings supplying the grid and epoch budget.
        seed: Seed used for every trial, so trials differ only in learning rate.

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
    settings: Mamba3ExperimentConfig,
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
    split: SequenceSplit, settings: Mamba3ExperimentConfig
) -> tuple[dict[str, tuple[str, str | None, ModelFactory, str | None]], dict[str, object]]:
    """Construct every architecture for one task, matched to the SSM parameter budget.

    Args:
        split: Task split, which fixes the vocabulary size.
        settings: Track settings.

    Returns:
        ``(zoo, shared_config)`` mapping a label to
        ``(architecture, variant, factory, notes)``.
    """

    vocab = split.vocab_size
    blocks = settings.n_blocks
    base = Mamba3Config(
        d_model=settings.d_model,
        n_blocks=blocks,
        heads=settings.heads,
        state_size=settings.state_size,
        rank=settings.rank,
    )

    def ssm_factory(config: Mamba3Config) -> ModelFactory:
        def factory(seed: int) -> nn.Module:
            seed_everything(seed)
            return Mamba3(vocab, config)

        return factory

    budget = count_parameters(Mamba3(vocab, base))

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
        "mamba3": ("mamba3", None, ssm_factory(base), None),
        "euler": (
            "mamba3",
            "euler-discretization",
            ssm_factory(replace(base, trapezoidal=False)),
            "Ablation: exponential-Euler instead of exponential-trapezoidal, i.e. the "
            "Mamba-1/2 discretization.",
        ),
        "real": (
            "mamba3",
            "no-rotation",
            ssm_factory(replace(base, rotary=False)),
            "Ablation: real non-negative transition instead of data-dependent rotations. "
            "The source predicts this cannot represent parity.",
        ),
        "siso": (
            "mamba3",
            "siso-rank-1",
            ssm_factory(replace(base, rank=1)),
            "Ablation: rank-1 (SISO) state update instead of MIMO.",
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
    split: SequenceSplit, settings: Mamba3ExperimentConfig, output_dir: Path
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


def measure_cost_scaling(
    settings: Mamba3ExperimentConfig, output_dir: Path, *, vocab_size: int = 8
) -> Path:
    """Measure forward-pass latency against sequence length, and serialize it.

    Asymptotically the SSM is linear in ``T`` and attention is quadratic. Whether that
    shows up in wall clock at these lengths is an empirical question, and one this
    implementation is badly placed to answer, so the measurement is recorded as an
    artefact with the caveat attached rather than presented as an architecture result.

    Args:
        settings: Track settings.
        output_dir: Directory receiving the artefact.
        vocab_size: Vocabulary used for the synthetic timing inputs.

    Returns:
        Path of the written file.
    """

    seed_everything(0)
    config = Mamba3Config(
        d_model=settings.d_model,
        n_blocks=settings.n_blocks,
        heads=settings.heads,
        state_size=settings.state_size,
        rank=settings.rank,
    )
    models: dict[str, nn.Module] = {
        "mamba3": Mamba3(vocab_size, config),
        "lstm": RecurrentBaseline(vocab_size, d_model=settings.d_model, n_layers=settings.n_blocks),
        "transformer": CausalTransformer(
            vocab_size,
            d_model=settings.d_model,
            n_layers=settings.n_blocks,
            n_heads=2,
            max_len=max(512, *settings.throughput_lengths),
        ),
    }

    measurements: list[dict[str, object]] = []
    for name, model in models.items():
        for length in settings.throughput_lengths:
            tokens = torch.randint(0, vocab_size, (16, length))
            profile = profile_inference(model, tokens, repeats=5, warmup=2)
            measurements.append(
                {
                    "architecture": name,
                    "sequence_length": length,
                    "latency_ms": profile.latency_ms_mean,
                    "latency_ms_std": profile.latency_ms_std,
                    "throughput_per_s": profile.throughput_per_s,
                    "parameters": profile_parameters(model).total,
                }
            )

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "track": TRACK,
        "measurement": "forward_latency_vs_sequence_length",
        "batch_size": 16,
        "caveat": (
            "The SSM and the recurrent baseline are stepped in Python; the Transformer "
            "uses batched attention kernels. These numbers compare implementations, not "
            "architectures, and must not be read as throughput claims."
        ),
        "measurements": measurements,
    }
    artefacts = output_dir / ARTEFACT_DIRNAME
    artefacts.mkdir(parents=True, exist_ok=True)
    path = artefacts / "cost_scaling.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run(output_dir: Path | str | None = None, *, quick: bool = False) -> None:
    """Run the whole Mamba-3 suite and write every record under ``output_dir``.

    Args:
        output_dir: Destination directory. Defaults to ``results/mamba3``.
        quick: Reduced seeds, epochs, and task sizes for smoke-testing the protocol.
    """

    destination = Path(output_dir) if output_dir is not None else Path("results") / TRACK
    settings = (
        Mamba3ExperimentConfig(
            seeds=(0, 1),
            epochs=3,
            n_sequences=120,
            d_model=16,
            learning_rate_grid=(3e-3,),
            throughput_lengths=(8, 16),
        )
        if quick
        else Mamba3ExperimentConfig()
    )

    # Identical to Track 02's tasks, so the two tracks' records line up.
    splits = (
        make_copy_task(
            n_sequences=settings.n_sequences,
            payload_len=3 if quick else 4,
            delay=4 if quick else 8,
            n_symbols=8,
            seed=1729,
        ),
        make_selective_recall_task(
            n_sequences=settings.n_sequences,
            n_pairs=2 if quick else 4,
            n_keys=8,
            n_values=8,
            seed=1729,
        ),
        make_state_tracking_task(
            n_sequences=settings.n_sequences,
            seq_len=8 if quick else 16,
            n_states=2,
            seed=1729,
        ),
    )

    for split in splits:
        run_task(split, settings, destination)

    measure_cost_scaling(settings, destination)
