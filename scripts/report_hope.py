"""Regenerate the generated tables inside ``reports/hope.md`` from saved records."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import aggregate_table, inject  # noqa: E402

RESULTS = ROOT / "results" / "hope"
NEWLINE = chr(10)
REPORT = ROOT / "reports" / "hope.md"


def main() -> int:
    """Regenerate every marked block in the Hope report.

    Returns:
        ``0`` on success, ``1`` when no records are available.
    """

    if not RESULTS.exists():
        print(f"No records under {RESULTS}. Run `modern-nn run-track hope` first.")
        return 1

    records = list(iter_records(RESULTS))
    if not records:
        print("No records found.")
        return 1

    def task_key(record: ExperimentRecord) -> tuple[str, ...]:
        return (record.dataset, record.variant or "default")

    blocks = {
        "diagnostics": aggregate_table(records, key=task_key, key_columns=["stream", "learner"]),
        "forgetting": _forgetting_table(),
        "cost": _cost_table(records),
    }

    replaced = inject(REPORT, blocks)
    print(f"Regenerated {len(replaced)} block(s) in {REPORT.relative_to(ROOT)}: {replaced}")
    print(f"Records read: {len(records)}")
    return 0


def _forgetting_table() -> str:
    """Render the forgetting curve and the memory-trace summary from the artefact."""

    source = RESULTS / "artefacts" / "continual_diagnostics.json"
    if not source.exists():
        return "_No memory diagnostics recorded._"

    payload = json.loads(source.read_text(encoding="utf-8"))
    tasks = sorted(payload["forgetting_curve"][0]["per_task_mse"])
    headers = ["learner", *[f"task {t} MSE" for t in tasks], "lr"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in payload["forgetting_curve"]:
        cells = [row["learner"]]
        cells += [f"{row['per_task_mse'][t]:.3f}" for t in tasks]
        cells.append(f"{row['selected_learning_rate']:g}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Level update frequency (the timescale claim, audited):")
    lines.append("")
    freq_headers = ["learner", "L0 updates", "L1 updates", "samples seen"]
    lines.append("| " + " | ".join(freq_headers) + " |")
    lines.append("|" + "|".join("---" for _ in freq_headers) + "|")
    for learner, row in payload["level_update_frequency"].items():
        lines.append(
            f"| {learner} | {row['data_memory_steps']} | "
            f"{row['gradient_memory_steps']} | {row['samples_seen']} |"
        )

    lines.append("")
    lines.append("Few-shot adaptation after a long history, and conflicting tasks:")
    lines.append("")
    probe_headers = ["learner", "few-shot MSE", "conflicting-stream MSE"]
    lines.append("| " + " | ".join(probe_headers) + " |")
    lines.append("|" + "|".join("---" for _ in probe_headers) + "|")
    for learner, value in payload["few_shot_after_long_history_mse"].items():
        conflict = payload["conflicting_tasks_mse"].get(learner, float("nan"))
        lines.append(f"| {learner} | {value:.3f} | {conflict:.3f} |")
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
