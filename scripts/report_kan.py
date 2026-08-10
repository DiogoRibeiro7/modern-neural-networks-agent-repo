"""Regenerate the generated tables inside ``reports/kan.md`` from saved records.

Prose in the report is written by hand. Every number is produced here, so the report
cannot cite a figure that the committed records do not support.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import aggregate_table, inject  # noqa: E402

RESULTS = ROOT / "results" / "kan"
REPORT = ROOT / "reports" / "kan.md"

SYNTHETIC_DATASETS = ("sin-2pix", "exp-sin-plus-square")


def main() -> int:
    """Regenerate every marked block in the KAN report.

    Returns:
        ``0`` on success, ``1`` when no records are available.
    """

    if not RESULTS.exists():
        print(f"No records under {RESULTS}. Run `modern-nn run-track kan` first.")
        return 1

    records = list(iter_records(RESULTS))
    if not records:
        print("No records found.")
        return 1

    def is_sweep(record: ExperimentRecord) -> bool:
        return record.config.get("sweep") is not None

    def model_key(record: ExperimentRecord) -> tuple[str, ...]:
        return (record.dataset, record.architecture, record.variant or "default")

    def sweep_key(record: ExperimentRecord) -> tuple[str, ...]:
        return (str(record.config.get("sweep")), record.variant or "default")

    diagnostics = [
        record
        for record in records
        if record.dataset in SYNTHETIC_DATASETS and not is_sweep(record)
    ]
    sweeps = [record for record in records if is_sweep(record)]
    benchmark = [
        record
        for record in records
        if record.dataset not in SYNTHETIC_DATASETS and not is_sweep(record)
    ]

    blocks = {
        "diagnostics": aggregate_table(
            diagnostics, key=model_key, key_columns=["dataset", "architecture", "variant"]
        ),
        "sensitivity": aggregate_table(sweeps, key=sweep_key, key_columns=["sweep", "setting"]),
        "benchmark": aggregate_table(
            benchmark, key=model_key, key_columns=["dataset", "model", "variant"]
        ),
        "cost": _cost_table(records),
    }

    replaced = inject(REPORT, blocks)
    print(f"Regenerated {len(replaced)} block(s) in {REPORT.relative_to(ROOT)}: {replaced}")
    print(f"Records read: {len(records)}")
    return 0


def _cost_table(records: list[ExperimentRecord]) -> str:
    """Render measured inference cost for the main architectures."""

    rows: dict[tuple[str, str], list[ExperimentRecord]] = {}
    for record in records:
        if record.config.get("sweep") is not None or record.inference_latency_ms is None:
            continue
        rows.setdefault((record.architecture, record.variant or "default"), []).append(record)

    if not rows:
        return "_No latency measurements recorded._"

    headers = [
        "architecture",
        "variant",
        "params",
        "latency ms (batch)",
        "throughput /s",
        "train s",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for (architecture, variant), group in sorted(rows.items()):
        latencies = [r.inference_latency_ms for r in group if r.inference_latency_ms is not None]
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
                    f"{sum(latencies) / len(latencies):.3f}",
                    f"{sum(throughput) / len(throughput):.0f}",
                    f"{sum(r.train_wall_clock_s for r in group) / len(group):.2f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
