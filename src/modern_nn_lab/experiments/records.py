"""Versioned, machine-readable experiment records.

Every reported number in this repository must originate from a serialized
:class:`ExperimentRecord`. Reports and figures are regenerated from these files, so a
record is the unit of scientific evidence, not a convenience log.

The schema implements ``docs/experiment_contract.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

RESULT_SCHEMA_VERSION = 1
"""Increment on any breaking change to :class:`ExperimentRecord`."""

RunStatus = Literal["success", "failed", "diverged", "interrupted"]

RESULTS_ROOT = Path("results")
"""Repository-relative directory holding committed raw records."""


class HardwareDescriptor(BaseModel):
    """Description of the machine that produced a record.

    Throughput and wall-clock numbers are meaningless without it, so it is mandatory.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    device: str = Field(description="Torch device string, for example 'cpu' or 'cuda:0'.")
    device_name: str = Field(description="Accelerator or CPU model name.")
    platform: str = Field(description="Operating system and release identifier.")
    python_version: str
    torch_version: str
    cpu_count: int | None = None


def describe_hardware(device: torch.device | str = "cpu") -> HardwareDescriptor:
    """Collect a hardware descriptor for the current process.

    Args:
        device: Device the measurements were taken on.

    Returns:
        A populated :class:`HardwareDescriptor`.
    """

    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(resolved)
    else:
        device_name = platform.processor() or platform.machine() or "unknown"

    return HardwareDescriptor(
        device=str(resolved),
        device_name=device_name,
        platform=platform.platform(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cpu_count=os.cpu_count(),
    )


def current_git_commit() -> str | None:
    """Return the current ``HEAD`` commit hash, or ``None`` outside a Git checkout.

    Returns:
        Full 40-character hash, or ``None`` when Git is unavailable or the working
        directory is not a repository.
    """

    try:
        # Fixed argument vector, no shell, no user-controlled input.
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def fingerprint(payload: object) -> str:
    """Return a stable 16-character fingerprint of a JSON-serializable payload.

    Used for dataset and configuration identity. Keys are sorted so the fingerprint is
    independent of dictionary ordering.

    Args:
        payload: Any JSON-serializable object.

    Returns:
        Truncated hexadecimal SHA-256 digest.
    """

    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def tensor_fingerprint(*tensors: torch.Tensor) -> str:
    """Return a stable fingerprint of one or more tensors.

    Args:
        *tensors: Tensors whose contents identify a dataset split.

    Returns:
        Truncated hexadecimal SHA-256 digest over shapes, dtypes, and raw bytes.
    """

    digest = hashlib.sha256()
    for tensor in tensors:
        detached = tensor.detach().to("cpu").contiguous()
        digest.update(str(tuple(detached.shape)).encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()[:16]


class MetricValue(BaseModel):
    """A named metric with its orientation.

    ``higher_is_better`` is stored explicitly so aggregation code never has to guess
    from the metric name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float
    higher_is_better: bool


class ExperimentRecord(BaseModel):
    """One completed run of one model under one configuration and one seed.

    A record describes a *single* seed. Uncertainty is computed across records by
    :func:`modern_nn_lab.experiments.evaluation.aggregate_runs`, never stored as a
    pre-aggregated number, so the individual seeds always remain inspectable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = RESULT_SCHEMA_VERSION

    # Identity
    track: str
    architecture: str
    architecture_version: str = "0.1.0"
    variant: str | None = Field(
        default=None,
        description="Ablation or configuration label, for example 'frozen-memory'.",
    )
    git_commit: str | None = None

    # Data
    dataset: str
    dataset_fingerprint: str | None = None
    split_strategy: str
    train_samples: int | None = None
    eval_samples: int | None = None

    # Optimization
    seed: int
    optimizer: str
    scheduler: str | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    epochs: int | None = None
    steps: int | None = None
    effective_samples: int | None = Field(
        default=None, description="Samples (or tokens) actually processed during training."
    )

    # Capacity and cost
    parameter_count: int
    activated_parameter_count: int | None = Field(
        default=None,
        description="Parameters used per token/example. Differs from parameter_count for MoE.",
    )
    flops_per_sample: float | None = None
    train_wall_clock_s: float
    inference_latency_ms: float | None = None
    inference_throughput_per_s: float | None = None
    peak_memory_bytes: int | None = None

    # Results
    primary_metric: MetricValue
    secondary_metrics: dict[str, float] = Field(default_factory=dict)
    train_loss_trajectory: list[float] = Field(default_factory=list)
    eval_metric_trajectory: list[float] = Field(default_factory=list)

    # Environment and provenance
    hardware: HardwareDescriptor
    precision: str = "float32"
    config: dict[str, Any] = Field(default_factory=dict)
    status: RunStatus = "success"
    notes: str | None = None

    @field_validator("seed")
    @classmethod
    def _seed_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be non-negative")
        return value

    @field_validator("parameter_count")
    @classmethod
    def _parameter_count_is_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("parameter_count must be non-negative")
        return value

    @field_validator("train_wall_clock_s")
    @classmethod
    def _wall_clock_is_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("train_wall_clock_s must be non-negative")
        return value

    @property
    def config_fingerprint(self) -> str:
        """Immutable fingerprint of the configuration snapshot."""

        return fingerprint(self.config)

    def filename(self) -> str:
        """Return a deterministic, collision-resistant file name for this record.

        The dataset *fingerprint* is part of the name, not just the dataset *label*. Two
        splits can legitimately share a label while differing in their parameters — a
        task at two sequence lengths, say — and without the fingerprint the second run
        would silently overwrite the first, leaving a group with missing seeds and
        results from two different datasets merged under one name.
        """

        variant = self.variant or "default"
        parts = [self.architecture, variant, self.dataset]
        if self.dataset_fingerprint is not None:
            parts.append(self.dataset_fingerprint[:8])
        parts.append(f"seed{self.seed}")
        slug = "__".join(part.replace("/", "-").replace(" ", "-") for part in parts)
        return f"{slug}.json"


def save_record(record: ExperimentRecord, directory: Path | str) -> Path:
    """Write ``record`` to ``directory`` as pretty-printed JSON.

    Args:
        record: Record to serialize.
        directory: Destination directory. Created if absent.

    Returns:
        Path of the written file.
    """

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / record.filename()
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_record(path: Path | str) -> ExperimentRecord:
    """Load and validate a single record.

    Args:
        path: Path to a JSON record.

    Returns:
        The validated record.

    Raises:
        ValueError: If the record was written by an incompatible schema version.
    """

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema_version {version!r} is not supported by this checkout "
            f"(expected {RESULT_SCHEMA_VERSION})."
        )
    return ExperimentRecord.model_validate(payload)


ARTEFACT_DIRNAME = "artefacts"
"""Subdirectory holding derived data that is *not* an experiment record.

Serialized edge functions, routing traces, and similar diagnostics belong under
``results/<track>/artefacts/``. They are committed so figures stay reproducible, but they
do not satisfy the record schema and must not be validated against it.
"""


def iter_records(root: Path | str = RESULTS_ROOT) -> Iterator[ExperimentRecord]:
    """Yield every record under ``root`` in sorted path order.

    Files under an :data:`ARTEFACT_DIRNAME` directory are skipped: they are derived
    diagnostics, not records.

    Args:
        root: Directory searched recursively for ``*.json`` files.

    Yields:
        Validated records.
    """

    for path in sorted(Path(root).rglob("*.json")):
        if ARTEFACT_DIRNAME in path.parts:
            continue
        yield load_record(path)


def records_to_rows(records: Iterable[ExperimentRecord]) -> list[dict[str, Any]]:
    """Flatten records into tabular rows suitable for a report table or CSV.

    Args:
        records: Records to flatten.

    Returns:
        One dictionary per record with the comparison-relevant columns.
    """

    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {
            "track": record.track,
            "architecture": record.architecture,
            "variant": record.variant or "default",
            "dataset": record.dataset,
            "seed": record.seed,
            "params": record.parameter_count,
            "activated_params": record.activated_parameter_count,
            "metric": record.primary_metric.name,
            "value": record.primary_metric.value,
            "train_s": record.train_wall_clock_s,
            "status": record.status,
        }
        row.update({f"secondary.{k}": v for k, v in record.secondary_metrics.items()})
        rows.append(row)
    return rows


def format_markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    """Render rows as a GitHub-flavoured Markdown table.

    Args:
        rows: Row dictionaries, typically from :func:`records_to_rows`.
        columns: Column keys to render, in order.

    Returns:
        Markdown table text, or a placeholder when ``rows`` is empty.
    """

    if not rows:
        return "_No records._"

    def cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        if value is None:
            return "-"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])
