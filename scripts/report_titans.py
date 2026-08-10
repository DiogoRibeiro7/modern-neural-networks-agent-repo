"""Regenerate the generated tables inside ``reports/titans.md`` from saved records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import aggregate_table, inject  # noqa: E402

RESULTS = ROOT / "results" / "titans"
NEWLINE = chr(10)
REPORT = ROOT / "reports" / "titans.md"


def main() -> int:
    """Regenerate every marked block in the Titans report.

    Returns:
        ``0`` on success, ``1`` when no records are available.
    """

    if not RESULTS.exists():
        print(f"No records under {RESULTS}. Run `modern-nn run-track titans` first.")
        return 1

    records = list(iter_records(RESULTS))
    if not records:
        print("No records found.")
        return 1

    def task_key(record: ExperimentRecord) -> tuple[str, ...]:
        distance = record.config.get("distance", "?")
        return (f"distance {distance}", record.architecture, record.variant or "default")

    blocks = {
        "diagnostics": aggregate_table(
            records, key=task_key, key_columns=["task", "architecture", "variant"]
        ),
        "forgetting": _forgetting_table(),
        "cost": _cost_table(records),
    }

    replaced = inject(REPORT, blocks)
    print(f"Regenerated {len(replaced)} block(s) in {REPORT.relative_to(ROOT)}: {replaced}")
    print(f"Records read: {len(records)}")
    return 0


def _forgetting_table() -> str:
    """Render the forgetting curve and the memory-trace summary from the artefact."""

    source = RESULTS / "artefacts" / "memory_diagnostics.json"
    if not source.exists():
        return "_No memory diagnostics recorded._"

    payload = json.loads(source.read_text(encoding="utf-8"))
    headers = ["write-to-query distance", "with memory", "short-term only", "gap", "chance"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in payload["forgetting_curve"]:
        gap = row["accuracy_with_memory"] - row["accuracy_short_term_only"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["distance"]),
                    f"{row['accuracy_with_memory']:.3f}",
                    f"{row['accuracy_short_term_only']:.3f}",
                    f"{gap:+.3f}",
                    f"{row['chance']:.3f}",
                ]
            )
            + " |"
        )

    trace = payload["trace_one_off"]
    repeated = payload["trace_repeated"]
    lines.append("")
    lines.append("Memory-trace summary (mean over tokens, averaged across the batch):")
    lines.append("")
    trace_headers = ["series", "one-off fact", "repeated fact"]
    lines.append("| " + " | ".join(trace_headers) + " |")
    lines.append("|" + "|".join("---" for _ in trace_headers) + "|")
    for key in ("loss", "surprise_norm", "write_norm", "forget_gate", "learning_rate"):
        one = sum(trace[key]) / max(len(trace[key]), 1)
        rep = sum(repeated[key]) / max(len(repeated[key]), 1)
        lines.append(f"| {key} | {one:.4g} | {rep:.4g} |")
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
