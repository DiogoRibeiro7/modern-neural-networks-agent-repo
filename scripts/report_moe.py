"""Regenerate the generated tables inside ``reports/moe.md`` from saved records."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import aggregate_table, inject  # noqa: E402

RESULTS = ROOT / "results" / "moe"
REPORT = ROOT / "reports" / "moe.md"
NEWLINE = chr(10)

MODEL_ORDER = (
    "dense-ffn",
    "dense-moe",
    "sparse-top1",
    "sparse-top1-raw",
    "sparse-top2",
    "sparse-capacity0.5",
    "sparse-aux0",
)


def main() -> int:
    """Regenerate every marked block in the mixture-of-experts report.

    Returns:
        ``0`` on success, ``1`` when no records are available.
    """

    if not RESULTS.exists():
        print(f"No records under {RESULTS}. Run `modern-nn run-track moe` first.")
        return 1

    records = list(iter_records(RESULTS))
    if not records:
        print("No records found.")
        return 1

    blocks = {
        "cost": _cost_table(records),
        "routing": _routing_table(),
        "specialization": _specialization_table(),
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


def _ordered(names: set[str]) -> list[str]:
    """Order model names by the reporting sequence, appending unexpected ones."""

    known = [name for name in MODEL_ORDER if name in names]
    return known + sorted(names - set(MODEL_ORDER))


def _cost_table(records: list[ExperimentRecord]) -> str:
    """Render the five quantities the prompt requires, in one table."""

    grouped: dict[str, list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault(record.architecture, []).append(record)

    headers = [
        "model",
        "test MSE",
        "±",
        "total params",
        "activated",
        "activated %",
        "FLOPs/token",
        "tokens/s",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]

    for name in _ordered(set(grouped)):
        group = grouped[name]
        scores = [record.primary_metric.value for record in group]
        mean = sum(scores) / len(scores)
        spread = (max(scores) - min(scores)) / 2
        config = group[0].config
        lines.append(
            f"| {name} | {mean:.4f} | {spread:.4f} | "
            f"{int(config['total_parameters'])} | {int(config['activated_parameters'])} | "
            f"{float(config['activated_fraction']) * 100:.0f}% | "
            f"{int(config['flops_per_token'])} | "
            f"{float(config['measured_tokens_per_s']):,.0f} |"
        )
    return NEWLINE.join(lines)


def _diagnostics() -> dict[str, Any] | None:
    """Load the routing diagnostics artefact, or ``None`` when it is absent."""

    source = RESULTS / "artefacts" / "routing_diagnostics.json"
    if not source.exists():
        return None
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    return payload


def _routing_table() -> str:
    """Render utilization, entropy, and drops — the acceptance criterion."""

    payload = _diagnostics()
    if payload is None:
        return "_No routing diagnostics recorded._"

    headers = [
        "variant",
        "balancing loss",
        "entropy (norm.)",
        "utilization min",
        "utilization max",
        "dropped",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in payload["variants"]:
        lines.append(
            f"| {row['variant']} | {row['load_balancing_loss']:.3f} | "
            f"{row['routing_entropy_normalized']:.3f} | "
            f"{row['expert_utilization_min']:.3f} | {row['expert_utilization_max']:.3f} | "
            f"{row['dropped_fraction']:.3f} |"
        )
    lines.append("")
    lines.append(
        "_Normalized entropy is 1.0 for a uniform router and 0.0 for a fully committed "
        "one. The balancing loss is 1.0 under uniform routing and "
        f"{payload['n_experts']}.0 under total collapse._"
    )
    return NEWLINE.join(lines)


def _specialization_table() -> str:
    """Render specialization purity and the confusion matrix of the best variant."""

    payload = _diagnostics()
    if payload is None:
        return "_No specialization diagnostics recorded._"

    chance = float(payload["chance_purity"])
    headers = ["variant", "purity", "vs chance", "verdict"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]

    best = None
    for row in payload["variants"]:
        purity = float(row["specialization_purity"])
        if best is None or purity > float(best["specialization_purity"]):
            best = row
        margin = purity - chance
        if purity > 0.75:
            verdict = "specialized"
        elif margin > 0.15:
            verdict = "partial"
        else:
            verdict = "near chance"
        lines.append(f"| {row['variant']} | {purity:.3f} | {margin:+.3f} | {verdict} |")

    lines.append("")
    lines.append(f"_Chance purity is {chance:.2f} with {payload['n_experts']} experts._")

    if best is not None:
        lines.append("")
        lines.append(
            f"Routing counts for `{best['variant']}`, "
            "rows are the true generating function and columns the chosen expert:"
        )
        lines.append("")
        n_experts = payload["n_experts"]
        head = ["function", *[f"expert {index}" for index in range(n_experts)]]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join("---" for _ in head) + "|")
        for index, row_counts in enumerate(best["specialization_matrix"]):
            cells = [f"f{index}", *[str(int(value)) for value in row_counts]]
            lines.append("| " + " | ".join(cells) + " |")

    return NEWLINE.join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
