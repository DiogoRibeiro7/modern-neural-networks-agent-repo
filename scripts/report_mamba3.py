"""Regenerate the generated tables inside ``reports/mamba3.md`` from saved records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import aggregate_table, inject  # noqa: E402

RESULTS = ROOT / "results" / "mamba3"
NEWLINE = chr(10)
REPORT = ROOT / "reports" / "mamba3.md"


def main() -> int:
    """Regenerate every marked block in the Mamba-3 report.

    Returns:
        ``0`` on success, ``1`` when no records are available.
    """

    if not RESULTS.exists():
        print(f"No records under {RESULTS}. Run `modern-nn run-track mamba3` first.")
        return 1

    records = list(iter_records(RESULTS))
    if not records:
        print("No records found.")
        return 1

    def task_key(record: ExperimentRecord) -> tuple[str, ...]:
        return (record.dataset, record.architecture, record.variant or "default")

    blocks = {
        "diagnostics": aggregate_table(
            records, key=task_key, key_columns=["task", "architecture", "variant"]
        ),
        "scaling": _scaling_table(),
        "cost": _cost_table(records),
    }

    replaced = inject(REPORT, blocks)
    print(f"Regenerated {len(replaced)} block(s) in {REPORT.relative_to(ROOT)}: {replaced}")
    print(f"Records read: {len(records)}")
    return 0


def _scaling_table() -> str:
    """Render the latency-versus-sequence-length artefact."""

    source = RESULTS / "artefacts" / "cost_scaling.json"
    if not source.exists():
        return "_No cost-scaling measurement recorded._"

    payload = json.loads(source.read_text(encoding="utf-8"))
    lengths = sorted({row["sequence_length"] for row in payload["measurements"]})
    architectures = sorted({row["architecture"] for row in payload["measurements"]})
    lookup = {
        (row["architecture"], row["sequence_length"]): row["latency_ms"]
        for row in payload["measurements"]
    }

    headers = ["architecture", *[f"T={length}" for length in lengths], "T scaling"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for architecture in architectures:
        values = [lookup[(architecture, length)] for length in lengths]
        ratio = values[-1] / values[0] if values[0] > 0 else float("nan")
        span = lengths[-1] / lengths[0]
        lines.append(
            "| "
            + " | ".join(
                [
                    architecture,
                    *[f"{value:.2f}" for value in values],
                    f"{ratio:.1f}x over {span:.0f}x",
                ]
            )
            + " |"
        )
    return NEWLINE.join(lines)


def _cost_table(records: list[ExperimentRecord]) -> str:
    """Render measured training and inference cost per architecture."""

    grouped: dict[tuple[str, str], list[ExperimentRecord]] = {}
    for record in records:
        if record.inference_latency_ms is None:
            continue
        grouped.setdefault((record.architecture, record.variant or "default"), []).append(record)

    if not grouped:
        return "_No latency measurements recorded._"

    headers = ["architecture", "variant", "params", "latency ms", "throughput /s", "train s"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for (architecture, variant), group in sorted(grouped.items()):
        latency = [r.inference_latency_ms for r in group if r.inference_latency_ms is not None]
        throughput = [
            r.inference_throughput_per_s for r in group if r.inference_throughput_per_s is not None
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    architecture,
                    variant,
                    str(group[0].parameter_count),
                    f"{sum(latency) / len(latency):.2f}",
                    f"{sum(throughput) / len(throughput):.0f}",
                    f"{sum(r.train_wall_clock_s for r in group) / len(group):.1f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
