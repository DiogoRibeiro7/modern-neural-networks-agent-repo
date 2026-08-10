"""Regenerate the generated tables inside ``reports/relational.md`` from saved records."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import aggregate_table, inject  # noqa: E402

RESULTS = ROOT / "results" / "relational"
REPORT = ROOT / "reports" / "relational.md"
NEWLINE = chr(10)

REGIME_ORDER = ("one_hop", "multi_hop", "temporal", "irrelevant", "cold_start")
MODEL_ORDER = ("relational", "gnn-flat", "gbdt-flat", "target-only", "no-types", "no-time")


def main() -> int:
    """Regenerate every marked block in the relational report.

    Returns:
        ``0`` on success, ``1`` when no records are available.
    """

    if not RESULTS.exists():
        print(f"No records under {RESULTS}. Run `modern-nn run-track relational` first.")
        return 1

    records = list(iter_records(RESULTS))
    if not records:
        print("No records found.")
        return 1

    blocks = {
        "regimes": _regime_matrix(records),
        "ablations": _ablation_table(records),
        "capacity": _capacity_table(records),
        "leakage": _leakage_table(),
        "paths": _path_table(),
        "all_runs": aggregate_table(
            records,
            key=lambda record: (record.variant or "default", record.architecture),
            key_columns=["regime", "model"],
        ),
    }

    replaced = inject(REPORT, blocks)
    print(f"Regenerated {len(replaced)} block(s) in {REPORT.relative_to(ROOT)}: {replaced}")
    print(f"Records read: {len(records)}")
    return 0


def _mean_by(records: list[ExperimentRecord]) -> dict[tuple[str, str], list[float]]:
    """Group primary metrics by ``(regime, model)``."""

    grouped: dict[tuple[str, str], list[float]] = {}
    for record in records:
        key = (record.variant or "default", record.architecture)
        grouped.setdefault(key, []).append(record.primary_metric.value)
    return grouped


def _ordered(names: set[str], preferred: tuple[str, ...]) -> list[str]:
    """Order names by a preferred sequence, appending any unexpected ones."""

    known = [name for name in preferred if name in names]
    return known + sorted(names - set(preferred))


def _regime_matrix(records: list[ExperimentRecord]) -> str:
    """Render test accuracy per regime for the headline comparisons."""

    grouped = _mean_by(records)
    headline = ("relational", "gnn-flat", "gbdt-flat", "target-only")
    regimes = _ordered({regime for regime, _ in grouped}, REGIME_ORDER)

    headers = ["regime", *headline]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for regime in regimes:
        cells = [regime]
        for model in headline:
            scores = grouped.get((regime, model))
            cells.append(f"{sum(scores) / len(scores):.3f}" if scores else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return NEWLINE.join(lines)


def _ablation_table(records: list[ExperimentRecord]) -> str:
    """Render each single-flag ablation against the full prototype."""

    grouped = _mean_by(records)
    ablations = ("relational", "no-types", "no-time", "gnn-flat")
    regimes = _ordered({regime for regime, _ in grouped}, REGIME_ORDER)

    headers = ["regime", *ablations, "largest drop"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for regime in regimes:
        cells = [regime]
        full = grouped.get((regime, "relational"))
        baseline = sum(full) / len(full) if full else None
        drops: dict[str, float] = {}
        for model in ablations:
            scores = grouped.get((regime, model))
            if not scores:
                cells.append("—")
                continue
            mean = sum(scores) / len(scores)
            cells.append(f"{mean:.3f}")
            if baseline is not None and model != "relational":
                drops[model] = baseline - mean
        if drops:
            worst = max(drops.items(), key=lambda item: item[1])
            cells.append(f"{worst[0]} ({worst[1]:+.3f})")
        else:
            cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    return NEWLINE.join(lines)


def _capacity_table(records: list[ExperimentRecord]) -> str:
    """Render parameter counts, so the width matching is visible rather than asserted."""

    counts: dict[str, set[int]] = {}
    rates: dict[str, set[float]] = {}
    for record in records:
        counts.setdefault(record.architecture, set()).add(record.parameter_count)
        rate = record.config.get("selected_learning_rate")
        if rate is not None:
            rates.setdefault(record.architecture, set()).add(float(rate))

    headers = ["model", "params", "selected learning rate(s)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for model in _ordered(set(counts), MODEL_ORDER):
        sizes = sorted(counts[model])
        shown = str(sizes[0]) if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]}"
        rate = (
            ", ".join(f"{value:g}" for value in sorted(rates.get(model, set())))
            if rates.get(model)
            else "n/a"
        )
        lines.append(f"| {model} | {shown} | {rate} |")
    return NEWLINE.join(lines)


def _audit() -> dict[str, Any] | None:
    """Load the leakage audit artefact, or ``None`` when it is absent."""

    source = RESULTS / "artefacts" / "leakage_audit.json"
    if not source.exists():
        return None
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    return payload


def _leakage_table() -> str:
    """Render the temporal-leakage audit."""

    payload = _audit()
    if payload is None:
        return "_No leakage audit recorded._"

    headers = [
        "regime",
        "canary accuracy",
        "rows seen (gated)",
        "rows seen (ungated)",
        "min event age",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for entry in payload["regimes"]:
        age = entry["min_event_age_observed"]
        lines.append(
            f"| {entry['regime']} | {entry['leakage_canary_accuracy']:.3f} | "
            f"{entry['rows_visible_gated']:.1f} | {entry['rows_visible_ungated']:.1f} | "
            + (f"{age:.4f} |" if age is not None else "n/a |")
        )
    lines.append("")
    lines.append(f"_{payload['note']}_")
    return NEWLINE.join(lines)


def _path_table() -> str:
    """Render the reachable-path summary and one worked example."""

    payload = _audit()
    if payload is None:
        return "_No path trace recorded._"

    headers = ["regime", "mean paths per prediction", "distinct path shapes"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for entry in payload["regimes"]:
        shapes = "; ".join(f"`{path}`" for path in entry["distinct_paths"])
        lines.append(f"| {entry['regime']} | {entry['mean_paths_per_prediction']:.1f} | {shapes} |")

    example = payload.get("explainability_example") or {}
    if example:
        lines.append("")
        lines.append(f"One prediction from `{example['regime']}`, every path it can reach:")
        lines.append("")
        lines.append("```")
        lines.extend(example["paths"])
        lines.append("```")
        lines.append("")
        lines.append(f"_{example['note']}_")
    return NEWLINE.join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
