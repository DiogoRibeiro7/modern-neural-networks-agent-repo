"""Regenerate the generated tables inside ``reports/ttt.md`` from saved records."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import aggregate_table, inject  # noqa: E402

RESULTS = ROOT / "results" / "ttt"
REPORT = ROOT / "reports" / "ttt.md"


def main() -> int:
    """Regenerate every marked block in the TTT report.

    Returns:
        ``0`` on success, ``1`` when no records are available.
    """

    if not RESULTS.exists():
        print(f"No records under {RESULTS}. Run `modern-nn run-track ttt` first.")
        return 1

    records = list(iter_records(RESULTS))
    if not records:
        print("No records found.")
        return 1

    def task_key(record: ExperimentRecord) -> tuple[str, ...]:
        return (record.dataset, record.architecture, record.variant or "default")

    blocks = {
        "diagnostics": aggregate_table(
            [r for r in records if r.config.get("study") != "context_scaling"],
            key=task_key,
            key_columns=["task", "architecture", "variant"],
        ),
        "scaling": aggregate_table(
            [r for r in records if r.config.get("study") == "context_scaling"],
            key=lambda r: (
                f"pairs={r.config.get('n_pairs')}",
                r.architecture,
                r.variant or "default",
            ),
            key_columns=["bindings", "architecture", "variant"],
        ),
        "cost": _cost_table(records),
    }

    replaced = inject(REPORT, blocks)
    print(f"Regenerated {len(replaced)} block(s) in {REPORT.relative_to(ROOT)}: {replaced}")
    print(f"Records read: {len(records)}")
    return 0


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
