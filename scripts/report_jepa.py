"""Regenerate the generated tables inside ``reports/jepa.md`` from saved records."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import aggregate_table, inject  # noqa: E402

RESULTS = ROOT / "results" / "jepa"
REPORT = ROOT / "reports" / "jepa.md"
NEWLINE = chr(10)

MODEL_ORDER = (
    "jepa-ema",
    "jepa-variance",
    "jepa-none",
    "autoencoder",
    "contrastive",
    "raw-features",
)


def main() -> int:
    """Regenerate every marked block in the JEPA report.

    Returns:
        ``0`` on success, ``1`` when no records are available.
    """

    if not RESULTS.exists():
        print(f"No records under {RESULTS}. Run `modern-nn run-track jepa` first.")
        return 1

    records = list(iter_records(RESULTS))
    if not records:
        print("No records found.")
        return 1

    blocks = {
        "headline": _headline_table(records),
        "masking": _sweep_table("masking_sweep", "n_targets", "patches masked"),
        "predictor": _sweep_table("predictor_sweep", "predictor_layers", "predictor layers"),
        "all_runs": aggregate_table(
            records,
            key=lambda record: (record.architecture,),
            key_columns=["model"],
        ),
    }

    replaced = inject(REPORT, blocks)
    print(f"Regenerated {len(replaced)} block(s) in {REPORT.relative_to(ROOT)}: {replaced}")
    print(f"Records read: {len(records)}")
    return 0


def _mean(values: list[float]) -> float:
    """Return the mean of a non-empty list, or NaN."""

    return sum(values) / len(values) if values else float("nan")


def _headline_table(records: list[ExperimentRecord]) -> str:
    """Render probes and collapse metrics side by side, which is the only honest way."""

    grouped: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault(record.architecture, []).append(record)

    headers = [
        "model",
        "content R2",
        "nuisance R2",
        "content - nuisance",
        "repr. std",
        "norm. eff. rank",
        "collapsed",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]

    known = [name for name in MODEL_ORDER if name in grouped]
    for name in known + sorted(set(grouped) - set(MODEL_ORDER)):
        group = grouped[name]

        def metric(key: str, source: list[ExperimentRecord] = group) -> float:
            return _mean([r.secondary_metrics[key] for r in source if key in r.secondary_metrics])

        std = metric("representation_std")
        # Standard deviation is the collapse detector; the rank cannot see total collapse.
        collapsed = "yes" if std < 0.05 else "no"
        lines.append(
            f"| {name} | {metric('content_r2'):.3f} | {metric('nuisance_r2'):.3f} | "
            f"{metric('content_minus_nuisance'):+.3f} | {std:.4f} | "
            f"{metric('normalized_effective_rank'):.3f} | {collapsed} |"
        )
    return NEWLINE.join(lines)


def _diagnostics() -> dict[str, Any] | None:
    """Load the diagnostics artefact, or ``None`` when it is absent."""

    source = RESULTS / "artefacts" / "representation_diagnostics.json"
    if not source.exists():
        return None
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    return payload


def _sweep_table(key: str, axis: str, axis_label: str) -> str:
    """Render one sweep from the diagnostics artefact.

    Args:
        key: Which sweep to render.
        axis: Field holding the swept value.
        axis_label: Column heading for that field.

    Returns:
        A markdown table.
    """

    payload = _diagnostics()
    if payload is None:
        return "_No diagnostics recorded._"

    headers = [axis_label, "content R2", "nuisance R2", "repr. std", "norm. eff. rank"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in payload[key]:
        lines.append(
            f"| {row[axis]} | {row['content_r2']:.3f} | {row['nuisance_r2']:.3f} | "
            f"{row['representation_std']:.4f} | {row['normalized_effective_rank']:.3f} |"
        )

    lines.append("")
    lines.append("_Single seed. A high content score with a low nuisance score is the goal._")
    return NEWLINE.join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
