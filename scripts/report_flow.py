"""Regenerate the generated tables inside ``reports/flow.md`` from saved records."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import aggregate_table, inject  # noqa: E402

RESULTS = ROOT / "results" / "flow"
REPORT = ROOT / "reports" / "flow.md"
NEWLINE = chr(10)

DATASET_ORDER = ("gaussian", "moons", "mixture")
PATH_ORDER = ("linear", "trigonometric")


def main() -> int:
    """Regenerate every marked block in the flow-matching report.

    Returns:
        ``0`` on success, ``1`` when no records are available.
    """

    if not RESULTS.exists():
        print(f"No records under {RESULTS}. Run `modern-nn run-track flow` first.")
        return 1

    records = list(iter_records(RESULTS))
    if not records:
        print("No records found.")
        return 1

    blocks = {
        "quality": _quality_table(records),
        "separation": _separation_table(),
        "solver": _solver_table(),
        "paths": _path_check_table(),
        "all_runs": aggregate_table(
            records,
            key=lambda record: (record.variant or "default", record.architecture),
            key_columns=["target", "path"],
        ),
    }

    replaced = inject(REPORT, blocks)
    print(f"Regenerated {len(replaced)} block(s) in {REPORT.relative_to(ROOT)}: {replaced}")
    print(f"Records read: {len(records)}")
    return 0


def _quality_table(records: list[ExperimentRecord]) -> str:
    """Render energy distance per target and path, beside the sampling-noise floor."""

    grouped: dict[tuple[str, str], list[ExperimentRecord]] = {}
    for record in records:
        grouped.setdefault((record.variant or "default", record.architecture), []).append(record)

    headers = ["target", "path", "energy distance", "±", "floor", "ratio to floor", "coverage"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]

    for dataset in DATASET_ORDER:
        for path in PATH_ORDER:
            group = grouped.get((dataset, path))
            if not group:
                continue
            scores = [record.primary_metric.value for record in group]
            mean = sum(scores) / len(scores)
            spread = (max(scores) - min(scores)) / 2
            floors = [
                record.secondary_metrics["energy_distance_floor"]
                for record in group
                if "energy_distance_floor" in record.secondary_metrics
            ]
            floor = sum(floors) / len(floors) if floors else float("nan")
            coverage = [
                record.secondary_metrics["mode_coverage"]
                for record in group
                if "mode_coverage" in record.secondary_metrics
            ]
            shown = f"{sum(coverage) / len(coverage):.2f}" if coverage else "—"
            lines.append(
                f"| {dataset} | {path} | {mean:.4f} | {spread:.4f} | {floor:.4f} | "
                f"{mean / max(floor, 1e-9):.1f}x | {shown} |"
            )
    return NEWLINE.join(lines)


def _diagnostics() -> dict[str, Any] | None:
    """Load the diagnostics artefact, or ``None`` when it is absent."""

    source = RESULTS / "artefacts" / "flow_diagnostics.json"
    if not source.exists():
        return None
    payload: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    return payload


def _separation_table() -> str:
    """Render the acceptance criterion: the two error sources, measured apart."""

    payload = _diagnostics()
    if payload is None:
        return "_No diagnostics recorded._"

    separation = payload["error_separation"]
    rows = [row for row in separation["rows"] if row["method"] == "euler"]

    headers = [
        "path",
        "solver steps",
        "discretization only",
        "combined",
        "field RMSE (no solver)",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append(
            f"| {row['path']} | {row['steps']} | {row['discretization_only']:.4f} | "
            f"{row['combined']:.4f} | {row['field_rmse']:.4f} |"
        )

    lines.append("")
    lines.append(
        f"_Sampling-noise floor: {separation['energy_distance_floor']:.4f}. {separation['note']}_"
    )
    return NEWLINE.join(lines)


def _solver_table() -> str:
    """Render Euler against midpoint at equal field-evaluation cost."""

    payload = _diagnostics()
    if payload is None:
        return "_No diagnostics recorded._"

    rows = payload["error_separation"]["rows"]
    by_cost: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        key = (row["path"], row["n_evaluations"])
        by_cost.setdefault(key, {})[row["method"]] = row["discretization_only"]

    headers = ["path", "field evaluations", "euler", "midpoint", "better"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for (path, evaluations), methods in sorted(by_cost.items()):
        if len(methods) < 2:
            continue
        euler = methods["euler"]
        midpoint = methods["midpoint"]
        better = "midpoint" if midpoint < euler else "euler"
        lines.append(f"| {path} | {evaluations} | {euler:.4f} | {midpoint:.4f} | {better} |")

    lines.append("")
    lines.append(
        "_Discretization error only, so the network contributes nothing. Compared at equal "
        "field evaluations rather than equal steps, since a midpoint step costs two._"
    )
    return NEWLINE.join(lines)


def _path_check_table() -> str:
    """Render the endpoint and orthogonality checks on each probability path."""

    payload = _diagnostics()
    if payload is None:
        return "_No diagnostics recorded._"

    headers = ["path", "(alpha, sigma) at t=0", "(alpha, sigma) at t=1", "max residual"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for check in payload["path_checks"]:
        start = ", ".join(f"{value:.3f}" for value in check["alpha_sigma_at_0"])
        end = ", ".join(f"{value:.3f}" for value in check["alpha_sigma_at_1"])
        worst = max(abs(value) for value in check["projection_residual"].values())
        lines.append(f"| {check['path']} | ({start}) | ({end}) | {worst:.4f} |")

    lines.append("")
    lines.append(
        "_A valid path has (alpha, sigma) = (0, 1) at t=0 and (1, 0) at t=1. The projection "
        "residual "
        "tests the closed-form marginal field against the defining property of a "
        "conditional expectation; it is a Monte-Carlo estimate of zero._"
    )
    return NEWLINE.join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
