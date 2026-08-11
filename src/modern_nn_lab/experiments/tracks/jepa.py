"""Experiment suite for Track 11 — JEPA / predictive representation learning.

Every number here is a property of the *representation*, measured on held-out data, and
none of them appears in any training objective. That matters more in this track than in
most: a latent-prediction loss is minimized exactly by a constant encoder, so training loss
is not merely a weak signal of representation quality — it can move in the opposite
direction.

The five required analyses map onto the runs as follows:

- **collapse metrics** — representation standard deviation and effective rank, on every run;
- **linear probing** — closed-form ridge regression recovering the content factors;
- **invariance to nuisance** — the same probe pointed at the nuisance factors, where a
  *low* score is the good outcome;
- **amount of target masking** — a sweep over how many patches are hidden;
- **predictor capacity** — a sweep over predictor depth, including the identity.

The headline metric is the content probe, but it is never reported without the collapse
metrics beside it, because a collapsed representation scores zero on the probe and that
coincidence should be legible rather than confusing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from modern_nn_lab.experiments.external import run_meta_evaluation
from modern_nn_lab.experiments.records import ARTEFACT_DIRNAME, RESULT_SCHEMA_VERSION, fingerprint
from modern_nn_lab.experiments.runner import RunGroup, RunSpec
from modern_nn_lab.reproducibility import seed_everything
from modern_nn_lab.tracks.jepa import (
    JEPA,
    Autoencoder,
    ContrastiveLearner,
    JEPAConfig,
    JEPAExperimentConfig,
    LatentDataset,
    RawFeatures,
    collapse_report,
    generate,
    jepa_loss,
    linear_probe,
    sample_masks,
)

TRACK = "jepa"
ARCHITECTURE_VERSION = "0.1.0"

Model = JEPA | Autoencoder | ContrastiveLearner | RawFeatures


def build_dataset(settings: JEPAExperimentConfig) -> tuple[LatentDataset, LatentDataset]:
    """Generate the dataset and split it into train and test parts.

    Args:
        settings: Track settings.

    Returns:
        ``(train, test)``.
    """

    dataset = generate(
        n_samples=settings.n_samples,
        n_patches=settings.n_patches,
        d_patch=settings.d_patch,
        d_content=settings.d_content,
        d_nuisance=settings.d_nuisance,
        noise=settings.noise,
        seed=settings.data_seed,
    )
    return dataset.split(0.7)


def train_model(
    name: str,
    config: JEPAConfig,
    train: LatentDataset,
    settings: JEPAExperimentConfig,
    *,
    seed: int,
    n_targets: int,
) -> Model:
    """Train one representation learner.

    Args:
        name: Which model to build.
        config: Architecture configuration.
        train: Training split.
        settings: Track settings.
        seed: Seed for initialization and batching.
        n_targets: Patches masked per sample, for the JEPA variants.

    Returns:
        The trained model.

    Raises:
        ValueError: If the model name is unknown.
    """

    seed_everything(seed)
    model = build_model(name, config)
    if isinstance(model, RawFeatures):
        return model  # nothing to learn

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)
    n_train = train.n_samples

    model.train()
    for _ in range(settings.steps):
        index = torch.randint(0, n_train, (settings.batch_size,), generator=generator)
        patches = train.patches[index]

        if isinstance(model, JEPA):
            context_mask, target_mask = sample_masks(
                patches.shape[0], patches.shape[1], n_targets=n_targets, generator=generator
            )
            predicted, targets = model(patches, context_mask)
            loss = jepa_loss(
                predicted,
                targets,
                target_mask,
                variance_weight=config.effective_variance_weight,
                variance_floor=config.variance_floor,
            )
        elif isinstance(model, Autoencoder):
            loss = nn.functional.mse_loss(model(patches), patches)
        else:
            loss = model(patches)

        optimizer.zero_grad(set_to_none=True)
        # Torch ships `Tensor.backward` unannotated; the call is what strict mypy flags.
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        if isinstance(model, JEPA):
            model.update_target()

    model.eval()
    return model


def build_model(name: str, config: JEPAConfig) -> Model:
    """Construct a model by name.

    Args:
        name: Model name.
        config: Architecture configuration.

    Returns:
        The model.

    Raises:
        ValueError: If the name is unknown.
    """

    if name == "autoencoder":
        return Autoencoder(config)
    if name == "contrastive":
        return ContrastiveLearner(config)
    if name == "raw-features":
        return RawFeatures(config)
    if name.startswith("jepa"):
        return JEPA(config)
    raise ValueError(f"unknown model {name!r}")


def evaluate_representation(
    model: Model, train: LatentDataset, test: LatentDataset
) -> dict[str, float]:
    """Probe a trained representation for content, nuisance, and collapse.

    Args:
        model: The trained model.
        train: Split used to fit the probes.
        test: Held-out split the probes are scored on.

    Returns:
        Probe scores and collapse statistics.
    """

    with torch.no_grad():
        train_features = model.represent(train.patches)
        test_features = model.represent(test.patches)

    content = linear_probe(train_features, train.content, test_features, test.content)
    # Nuisance is per patch; the sample-level representation is asked to recover the mean,
    # which is the part of it a pooled representation could plausibly carry.
    nuisance = linear_probe(
        train_features,
        train.nuisance.mean(dim=1),
        test_features,
        test.nuisance.mean(dim=1),
    )

    return {
        "content_r2": content,
        "nuisance_r2": nuisance,
        # High content and low nuisance is the goal; this states the trade in one number
        # without hiding either component, both of which are reported above.
        "content_minus_nuisance": content - nuisance,
        **collapse_report(test_features),
    }


def run_variant(
    name: str,
    config: JEPAConfig,
    settings: JEPAExperimentConfig,
    output_dir: Path,
    *,
    n_targets: int | None = None,
    label: str | None = None,
) -> RunGroup:
    """Train and probe one configuration across seeds.

    Args:
        name: Model name.
        config: Architecture configuration.
        settings: Track settings.
        output_dir: Directory receiving records.
        n_targets: Patches masked; defaults to the configured value.
        label: Record architecture label; defaults to ``name``.

    Returns:
        A :class:`RunGroup`.
    """

    train, test = build_dataset(settings)
    masked = n_targets if n_targets is not None else settings.n_targets
    architecture = label or name

    shared: dict[str, Any] = {
        "model": name,
        "n_targets": masked,
        "n_patches": settings.n_patches,
        "d_content": settings.d_content,
        "d_nuisance": settings.d_nuisance,
        "training_steps": settings.steps,
        **config.as_dict(),
    }
    identity = fingerprint(
        {
            "n_samples": settings.n_samples,
            "n_patches": settings.n_patches,
            "d_patch": settings.d_patch,
            "d_content": settings.d_content,
            "d_nuisance": settings.d_nuisance,
            "noise": settings.noise,
            "data_seed": settings.data_seed,
        }
    )

    def evaluate(seed: int) -> tuple[float, dict[str, float], dict[str, Any]]:
        model = train_model(name, config, train, settings, seed=seed, n_targets=masked)
        scores = evaluate_representation(model, train, test)
        return scores["content_r2"], scores, dict(shared)

    return run_meta_evaluation(
        evaluate,
        spec=RunSpec(
            track=TRACK,
            architecture=architecture,
            metric_name="content_probe_r2",
            higher_is_better=True,
            variant=config.anti_collapse if name.startswith("jepa") else name,
            architecture_version=ARCHITECTURE_VERSION,
            extra_config=shared,
            notes=(
                "Every reported quantity is a property of the representation on held-out "
                "data; none appears in any training objective."
            ),
        ),
        dataset="latent_patches",
        split_strategy=(
            f"{settings.n_samples} samples of {settings.n_patches} patches, split 70/30; "
            "content factors are shared within a sample and nuisance factors are drawn per "
            "patch, so content is exactly the predictable part"
        ),
        output_dir=output_dir,
        seeds=settings.seeds,
        dataset_fingerprint=identity,
        parameter_count=sum(
            p.numel() for p in build_model(name, config).parameters() if p.requires_grad
        ),
    )


def write_diagnostics(settings: JEPAExperimentConfig, output_dir: Path) -> Path:
    """Serialize the masking sweep, the predictor sweep, and a collapse demonstration.

    Args:
        settings: Track settings.
        output_dir: Directory receiving the artefact.

    Returns:
        Path of the written file.
    """

    train, test = build_dataset(settings)
    base = JEPAConfig(d_patch=settings.d_patch)

    masking: list[dict[str, Any]] = []
    for n_targets in settings.mask_sweep:
        model = train_model("jepa", base, train, settings, seed=0, n_targets=n_targets)
        masking.append(
            {
                "n_targets": n_targets,
                "context_patches": settings.n_patches - n_targets,
                "mask_fraction": n_targets / settings.n_patches,
                **evaluate_representation(model, train, test),
            }
        )

    predictor: list[dict[str, Any]] = []
    for depth in settings.predictor_sweep:
        config = JEPAConfig(d_patch=settings.d_patch, n_predictor_layers=depth)
        model = train_model("jepa", config, train, settings, seed=0, n_targets=settings.n_targets)
        predictor.append(
            {
                "predictor_layers": depth,
                "identity_predictor": depth == 0,
                **evaluate_representation(model, train, test),
            }
        )

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "track": TRACK,
        "measurement": "representation_diagnostics",
        "note": (
            "`content_r2` should be high and `nuisance_r2` low: the first says the "
            "representation kept what is shared across a sample's patches, the second that "
            "it discarded what varies between them. `normalized_effective_rank` near zero "
            "means collapse, and a collapsed representation also scores zero on both "
            "probes — so the probes alone cannot distinguish collapse from a merely "
            "uninformative representation, which is why the rank is reported beside them."
        ),
        "representation_width": base.d_representation,
        "masking_sweep": masking,
        "predictor_sweep": predictor,
    }

    artefacts = output_dir / ARTEFACT_DIRNAME
    artefacts.mkdir(parents=True, exist_ok=True)
    destination = artefacts / "representation_diagnostics.json"
    destination.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    return destination


def variants(settings: JEPAExperimentConfig) -> dict[str, tuple[str, JEPAConfig]]:
    """Enumerate the headline comparison.

    Args:
        settings: Track settings.

    Returns:
        Mapping from record label to ``(model name, configuration)``.
    """

    patch = settings.d_patch
    return {
        "jepa-ema": ("jepa", JEPAConfig(d_patch=patch, anti_collapse="ema")),
        "jepa-variance": ("jepa", JEPAConfig(d_patch=patch, anti_collapse="variance")),
        "jepa-none": ("jepa", JEPAConfig(d_patch=patch, anti_collapse="none")),
        "autoencoder": ("autoencoder", JEPAConfig(d_patch=patch)),
        "contrastive": ("contrastive", JEPAConfig(d_patch=patch)),
        "raw-features": ("raw-features", JEPAConfig(d_patch=patch)),
    }


def run(output_dir: Path | str | None = None, *, quick: bool = False) -> None:
    """Run the whole JEPA suite.

    Args:
        output_dir: Destination directory. Defaults to ``results/jepa``.
        quick: Reduced seeds, samples, and training steps.
    """

    destination = Path(output_dir) if output_dir is not None else Path("results") / TRACK
    settings = (
        JEPAExperimentConfig(
            seeds=(0, 1),
            n_samples=800,
            steps=200,
            mask_sweep=(1, 4, 7),
            predictor_sweep=(0, 2),
        )
        if quick
        else JEPAExperimentConfig()
    )

    for label, (name, config) in variants(settings).items():
        run_variant(name, config, settings, destination, label=label)

    write_diagnostics(settings, destination)
