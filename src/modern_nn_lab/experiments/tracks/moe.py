"""Experiment suite for Track 09 — sparse mixture of experts.

The comparison the prompt asks for is not a single number. A sparse layer is interesting
only if the *relationship* between five quantities is favourable — total parameters,
activated parameters, FLOPs, measured throughput, and the task metric — so all five are
recorded for every run and the report shows them in one table rather than quoting whichever
flatters the method.

Two things this suite is careful about:

**The dense ensemble is the reference, not the enemy.** ``dense-moe`` runs every expert on
every token. It is what a sparse layer approximates, so the gap between them is the price of
sparsity, measured directly rather than argued about. ``dense-ffn`` is the conventional
baseline whose cost sparsity is trying to beat.

**Throughput is measured on CPU and reported as such.** Sparse dispatch replaces one large
matmul with a gather, several small matmuls, and a scatter. On this hardware, at this scale,
that is very likely to be *slower* than dense despite costing fewer FLOPs. That is a real
measurement of this implementation on this machine, not a claim about what an optimized
kernel would do, and the report says so.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from modern_nn_lab.experiments.data import SupervisedSplit
from modern_nn_lab.experiments.records import ARTEFACT_DIRNAME, RESULT_SCHEMA_VERSION
from modern_nn_lab.experiments.runner import RunGroup, RunSpec, run_seeded_experiment
from modern_nn_lab.experiments.training import TrainingConfig, train_supervised
from modern_nn_lab.tracks.moe import (
    MixtureModel,
    MixtureTask,
    MoEConfig,
    MoEExperimentConfig,
    SparseMoELayer,
    generate,
    specialization_matrix,
    specialization_purity,
)

TRACK = "moe"
ARCHITECTURE_VERSION = "0.1.0"

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15


def build_task(settings: MoEExperimentConfig) -> MixtureTask:
    """Generate the mixture-of-functions dataset.

    Args:
        settings: Track settings.

    Returns:
        The task.
    """

    return generate(
        n_sequences=settings.n_sequences,
        seq_len=settings.seq_len,
        n_functions=settings.n_functions,
        d_selector=settings.d_selector,
        d_value=settings.d_value,
        seed=settings.data_seed,
    )


def build_split(task: MixtureTask, settings: MoEExperimentConfig) -> SupervisedSplit:
    """Split the task into train, validation, and test portions.

    Args:
        task: The generated task.
        settings: Track settings.

    Returns:
        The split.
    """

    total = int(task.inputs.shape[0])
    n_train = int(total * TRAIN_FRACTION)
    n_val = int(total * VAL_FRACTION)

    return SupervisedSplit(
        name="mixture_of_functions",
        train_inputs=task.inputs[:n_train],
        train_targets=task.targets[:n_train],
        val_inputs=task.inputs[n_train : n_train + n_val],
        val_targets=task.targets[n_train : n_train + n_val],
        test_inputs=task.inputs[n_train + n_val :],
        test_targets=task.targets[n_train + n_val :],
        strategy=(
            f"{total} sequences of {settings.seq_len} tokens drawn from a fixed bank of "
            f"{settings.n_functions} functions, split {n_train}/{n_val}/"
            f"{total - n_train - n_val} by sequence; the function bank is shared across "
            "splits, so this measures specialization rather than extrapolation"
        ),
        metadata={
            "n_functions": settings.n_functions,
            "seq_len": settings.seq_len,
            "d_selector": settings.d_selector,
            "d_value": settings.d_value,
        },
    )


def mse(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean squared error over tokens.

    Args:
        predictions: Shape ``(B, T, 1)``.
        targets: Shape ``(B, T, 1)``.

    Returns:
        Scalar loss.
    """

    return nn.functional.mse_loss(predictions, targets)


def mse_metric(predictions: Tensor, targets: Tensor) -> float:
    """Mean squared error as a float, for reporting.

    Args:
        predictions: Shape ``(B, T, 1)``.
        targets: Shape ``(B, T, 1)``.

    Returns:
        The error.
    """

    return float(nn.functional.mse_loss(predictions, targets))


class ModelCell:
    """Holds the model currently being trained, so the loss can reach its router.

    The shared runner builds a fresh model per seed and the shared training loop expects a
    loss of ``(predictions, targets)``. The auxiliary balancing term lives on neither: it
    is a property of the routing decision made during the forward pass. Binding the loss
    to a model built up front would silently penalize a *different* model's routing —
    one that never trains — so the factory publishes each seed's model here and the loss
    reads whichever is current.

    Attributes:
        model: The most recently constructed model, or ``None`` before the first build.
    """

    def __init__(self) -> None:
        """Create an empty cell."""

        self.model: MixtureModel | None = None


def make_loss(cell: ModelCell, weight: float) -> Callable[[Tensor, Tensor], Tensor]:
    """Build a loss that adds the current model's balancing term to the task loss.

    Args:
        cell: Holder for the model being trained.
        weight: Multiplier on the balancing loss; ``0`` disables it.

    Returns:
        A loss function.
    """

    def loss_fn(predictions: Tensor, targets: Tensor) -> Tensor:
        loss = mse(predictions, targets)
        model = cell.model
        info = None if model is None else model.last_routing
        if weight > 0 and info is not None:
            loss = loss + weight * info.load_balancing_loss
        return loss

    return loss_fn


def _factory(config: MoEConfig, cell: ModelCell) -> Callable[[int], nn.Module]:
    """Return a model factory that builds a freshly seeded model and publishes it.

    Args:
        config: Architecture configuration.
        cell: Holder updated with each newly built model.

    Returns:
        A callable taking a seed and returning a new model.
    """

    def build(seed: int) -> nn.Module:
        torch.manual_seed(seed)
        model = MixtureModel(config)
        cell.model = model
        return model

    return build


def select_learning_rate(
    config: MoEConfig, split: SupervisedSplit, settings: MoEExperimentConfig
) -> float:
    """Choose a learning rate per architecture on the validation split.

    Args:
        config: Architecture configuration.
        split: The split whose validation portion drives the choice.
        settings: Track settings.

    Returns:
        The best rate from the grid.
    """

    best_rate = settings.learning_rate_grid[0]
    best_score = float("inf")

    for rate in settings.learning_rate_grid:
        torch.manual_seed(0)
        cell = ModelCell()
        model = MixtureModel(config)
        cell.model = model
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
            loss_fn=make_loss(cell, config.aux_loss_weight),
            eval_inputs=split.val_inputs,
            eval_targets=split.val_targets,
            metric_fn=mse_metric,
            higher_is_better=False,
        )
        if outcome.best_metric < best_score:
            best_score = outcome.best_metric
            best_rate = rate

    return best_rate


def measure_throughput(model: MixtureModel, inputs: Tensor, repeats: int = 5) -> float:
    """Measure forward-pass throughput in tokens per second.

    Args:
        model: The model to time.
        inputs: Shape ``(B, T, d_in)`` batch to run.
        repeats: Timed passes to average over.

    Returns:
        Tokens per second.
    """

    model.eval()
    n_tokens = int(inputs.shape[0] * inputs.shape[1])
    with torch.no_grad():
        model(inputs)  # warm up allocator and any lazy initialization
        start = time.perf_counter()
        for _ in range(repeats):
            model(inputs)
        elapsed = time.perf_counter() - start
    model.train()
    return n_tokens * repeats / max(elapsed, 1e-9)


def run_variant(
    name: str,
    config: MoEConfig,
    split: SupervisedSplit,
    settings: MoEExperimentConfig,
    output_dir: Path,
) -> RunGroup:
    """Train one architecture across seeds and record the five required quantities.

    Args:
        name: Architecture label for the record.
        config: Architecture configuration.
        split: The shared split.
        settings: Track settings.
        output_dir: Directory receiving records.

    Returns:
        A :class:`RunGroup`.
    """

    rate = select_learning_rate(config, split, settings)
    cell = ModelCell()
    probe = MixtureModel(config)
    total_parameters = sum(parameter.numel() for parameter in probe.parameters())

    extra: dict[str, Any] = {
        **config.as_dict(),
        "selected_learning_rate": rate,
        "flops_per_token": probe.flops_per_token(),
        "total_parameters": total_parameters,
        "activated_parameters": probe.activated_parameters(),
        "activated_fraction": probe.activated_parameters() / max(total_parameters, 1),
        "measured_tokens_per_s": measure_throughput(probe, split.test_inputs),
    }

    return run_seeded_experiment(
        _factory(config, cell),
        split,
        spec=RunSpec(
            track=TRACK,
            architecture=name,
            metric_name="test_mse",
            higher_is_better=False,
            variant=config.layer,
            architecture_version=ARCHITECTURE_VERSION,
            activated_parameters=probe.activated_parameters(),
            extra_config=extra,
            notes=(
                "Activated parameters and FLOPs are analytic; throughput is measured on "
                "CPU and is a property of this implementation on this machine, not of "
                "sparse routing in general."
            ),
        ),
        training_config=TrainingConfig(
            epochs=settings.epochs, batch_size=settings.batch_size, learning_rate=rate
        ),
        loss_fn=make_loss(cell, config.aux_loss_weight),
        metric_fn=mse_metric,
        output_dir=output_dir,
        seeds=settings.seeds,
    )


def variants(settings: MoEExperimentConfig) -> dict[str, MoEConfig]:
    """Enumerate every configuration in the comparison.

    Args:
        settings: Track settings.

    Returns:
        Mapping from architecture label to configuration.
    """

    d_in = settings.d_selector + settings.d_value
    base: dict[str, Any] = {"d_in": d_in, "n_experts": settings.n_experts}
    configs: dict[str, MoEConfig] = {
        "dense-ffn": MoEConfig(layer="dense-ffn", aux_loss_weight=0.0, **base),
        "dense-moe": MoEConfig(layer="dense-moe", aux_loss_weight=0.0, **base),
    }

    for top_k in settings.top_k_grid:
        configs[f"sparse-top{top_k}"] = MoEConfig(layer="sparse-moe", top_k=top_k, **base)

    # Top-1 without renormalization. Under top-1 the renormalized gate is identically 1,
    # so it is constant in the router's parameters and the task loss contributes exactly
    # zero gradient to routing. This variant keeps the raw gate, which restores it.
    configs["sparse-top1-raw"] = MoEConfig(layer="sparse-moe", top_k=1, renormalize=False, **base)

    # Capacity below the even share forces overflow, which is the point of including it.
    for factor in settings.capacity_grid:
        if factor == settings.capacity_grid[0]:
            continue
        configs[f"sparse-capacity{factor:g}"] = MoEConfig(
            layer="sparse-moe", top_k=1, capacity_factor=factor, **base
        )

    # Zero weight is the ablation that shows what the balancing loss is doing.
    for weight in settings.aux_weight_grid:
        if weight == settings.aux_weight_grid[0]:
            continue
        configs[f"sparse-aux{weight:g}"] = MoEConfig(
            layer="sparse-moe", top_k=1, aux_loss_weight=weight, **base
        )

    return configs


def write_routing_diagnostics(
    task: MixtureTask, split: SupervisedSplit, settings: MoEExperimentConfig, output_dir: Path
) -> Path:
    """Train one model per sparse variant and record what its router actually did.

    The acceptance criterion: expert utilization and routing entropy must be reported. This
    also records the specialization matrix against the *known* generating function, which
    is what makes "the experts specialized" a checkable claim rather than a story told
    about a metric.

    Args:
        task: The generated task, whose latent ids are the ground truth.
        split: The shared split.
        settings: Track settings.
        output_dir: Directory receiving the artefact.

    Returns:
        Path of the written file.
    """

    total = int(task.inputs.shape[0])
    n_train = int(total * TRAIN_FRACTION)
    n_val = int(total * VAL_FRACTION)
    test_latent = task.latent[n_train + n_val :].reshape(-1)

    rows: list[dict[str, Any]] = []
    for name, config in variants(settings).items():
        if config.layer != "sparse-moe":
            continue

        torch.manual_seed(0)
        cell = ModelCell()
        model = MixtureModel(config)
        cell.model = model
        train_supervised(
            model,
            split.train_inputs,
            split.train_targets,
            config=TrainingConfig(
                epochs=settings.epochs,
                batch_size=settings.batch_size,
                learning_rate=settings.learning_rate_grid[0],
                seed=0,
            ),
            loss_fn=make_loss(cell, config.aux_loss_weight),
            eval_inputs=split.val_inputs,
            eval_targets=split.val_targets,
            metric_fn=mse_metric,
            higher_is_better=False,
        )

        layer = model.layer
        assert isinstance(layer, SparseMoELayer)
        model.eval()
        with torch.no_grad():
            hidden = model.norm(model.input_projection(split.test_inputs))
            expert_index, _, keep, info = layer.router(hidden.reshape(-1, config.d_model))

        chosen = expert_index[:, 0]
        matrix = specialization_matrix(test_latent, chosen, task.n_functions, settings.n_experts)
        rows.append(
            {
                "variant": name,
                "top_k": config.top_k,
                "capacity_factor": config.capacity_factor,
                "aux_loss_weight": config.aux_loss_weight,
                **info.as_metrics(),
                "capacity": info.capacity,
                "utilization": info.utilization.tolist(),
                "specialization_matrix": matrix.tolist(),
                "specialization_purity": specialization_purity(matrix),
                "kept_fraction": float(keep.float().mean()),
            }
        )

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "track": TRACK,
        "measurement": "routing_diagnostics",
        "note": (
            "Single seed per variant. `specialization_matrix` counts tokens by (true "
            "generating function, chosen expert): a router that recovered the mixture "
            "produces one dominant entry per row. `specialization_purity` summarizes that "
            "permutation-invariantly, and must be read beside the utilization spread — "
            "two functions sharing one expert still score well on purity alone."
        ),
        "n_functions": task.n_functions,
        "n_experts": settings.n_experts,
        "chance_purity": 1.0 / settings.n_experts,
        "variants": rows,
    }

    artefacts = output_dir / ARTEFACT_DIRNAME
    artefacts.mkdir(parents=True, exist_ok=True)
    path = artefacts / "routing_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return path


def run(output_dir: Path | str | None = None, *, quick: bool = False) -> None:
    """Run the whole mixture-of-experts suite.

    Args:
        output_dir: Destination directory. Defaults to ``results/moe``.
        quick: Reduced seeds, sequences, and epochs.
    """

    destination = Path(output_dir) if output_dir is not None else Path("results") / TRACK
    settings = (
        MoEExperimentConfig(seeds=(0, 1), n_sequences=300, epochs=12, learning_rate_grid=(3e-3,))
        if quick
        else MoEExperimentConfig()
    )

    task = build_task(settings)
    split = build_split(task, settings)

    for name, config in variants(settings).items():
        run_variant(name, config, split, settings, destination)

    write_routing_diagnostics(task, split, settings, destination)
