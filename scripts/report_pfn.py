"""Regenerate the generated tables inside ``reports/pfn.md`` from saved records."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import aggregate_table, inject  # noqa: E402

RESULTS = ROOT / "results" / "pfn"
REPORT = ROOT / "reports" / "pfn.md"
NEWLINE = chr(10)


def main() -> int:
    """Regenerate every marked block in the PFN report.

    Returns:
        ``0`` on success, ``1`` when no records are available.
    """

    if not RESULTS.exists():
        print(f"No records under {RESULTS}. Run `modern-nn run-track pfn` first.")
        return 1

    records = list(iter_records(RESULTS))
    if not records:
        print("No records found.")
        return 1

    blocks = {
        "in_prior": _comparison_table(records, _in_prior_datasets(records)),
        "out_of_prior": _comparison_table(records, _out_of_prior_datasets(records)),
        "context_sweep": _context_table(records),
        "noise": _comparison_table(records, _noise_datasets(records)),
        "imbalance": _comparison_table(records, _prior_datasets(records, "imbalanced")),
        "missingness": _comparison_table(records, _missing_datasets(records)),
        "features": _feature_table(records),
        "transfer": _transfer_table(),
        "cost": _cost_table(),
        "reference": _reference_status(),
        "all_runs": aggregate_table(
            records,
            key=lambda record: (record.dataset, record.architecture),
            key_columns=["evaluation", "model"],
        ),
    }

    replaced = inject(REPORT, blocks)
    print(f"Regenerated {len(replaced)} block(s) in {REPORT.relative_to(ROOT)}: {replaced}")
    print(f"Records read: {len(records)}")
    return 0


def _config(record: ExperimentRecord, key: str, default: Any = None) -> Any:
    """Read one entry from a record's configuration snapshot."""

    return record.config.get(key, default)


BASE_FEATURES = 4
BASE_CONTEXT = 20


def _is_baseline_condition(record: ExperimentRecord) -> bool:
    """Whether a record sits at the reference point of every sweep."""

    return (
        not _config(record, "label_noise")
        and not _config(record, "missing_rate")
        and _config(record, "n_features") == BASE_FEATURES
        and _config(record, "n_context") == BASE_CONTEXT
    )


def _in_prior_datasets(records: list[ExperimentRecord]) -> list[str]:
    """Datasets where the evaluation prior is the fitting prior, at the reference point."""

    return sorted(
        {
            record.dataset
            for record in records
            if _config(record, "in_prior") and _is_baseline_condition(record)
        }
    )


OUT_OF_PRIOR_FAMILIES = ("mlp", "xor")


def _out_of_prior_datasets(records: list[ExperimentRecord]) -> list[str]:
    """Datasets whose labelling family the model was never fitted to.

    The imbalanced prior is excluded on purpose. Its decision boundary is still a linear
    separator — only the label marginal moves — so grouping it with genuinely different
    boundary families would blur what "out of prior" means. It has its own section.
    """

    return sorted(
        {
            record.dataset
            for record in records
            if _config(record, "eval_prior") in OUT_OF_PRIOR_FAMILIES
        }
    )


def _noise_datasets(records: list[ExperimentRecord]) -> list[str]:
    """Datasets carrying label noise."""

    return sorted({record.dataset for record in records if _config(record, "label_noise")})


def _prior_datasets(records: list[ExperimentRecord], prior: str) -> list[str]:
    """Datasets evaluated on a named prior."""

    return sorted({record.dataset for record in records if _config(record, "eval_prior") == prior})


def _missing_datasets(records: list[ExperimentRecord]) -> list[str]:
    """Datasets with blanked feature entries."""

    return sorted({record.dataset for record in records if _config(record, "missing_rate")})


def _feature_table(records: list[ExperimentRecord]) -> str:
    """Render accuracy against input width — one prior-fitted model per width."""

    grouped: dict[tuple[int, str], list[float]] = {}
    for record in records:
        if not _config(record, "in_prior") or _config(record, "label_noise"):
            continue
        if _config(record, "missing_rate") or _config(record, "n_context") != BASE_CONTEXT:
            continue
        width = int(_config(record, "n_features", 0) or 0)
        grouped.setdefault((width, record.architecture), []).append(record.primary_metric.value)

    if not grouped:
        return "_No feature-count records._"

    widths = sorted({width for width, _ in grouped})
    models = sorted({model for _, model in grouped})

    headers = ["model", *[f"d={width}" for width in widths]]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for model in models:
        cells = [model]
        for width in widths:
            scores = grouped.get((width, model))
            cells.append(f"{sum(scores) / len(scores):.3f}" if scores else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return NEWLINE.join(lines)


def _cost_table() -> str:
    """Render measured prediction cost and the PFN's break-even dataset count."""

    source = RESULTS / "artefacts" / "prior_diagnostics.json"
    if not source.exists():
        return "_No cost measurements recorded._"

    cost = json.loads(source.read_text(encoding="utf-8"))["cost"]
    break_even = cost["break_even_tasks"]

    headers = ["model", "ms per dataset", "up-front cost", "PFN break-even (datasets)"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.append(
        f"| pfn | {cost['per_task_ms']['pfn']:.2f} | "
        f"{cost['prior_fitting_s']:.1f} s prior fitting "
        f"({cost['prior_fitting_steps']} steps) | — |"
    )
    for model, milliseconds in cost["per_task_ms"].items():
        if model == "pfn":
            continue
        crossover = break_even.get(model)
        lines.append(
            f"| {model} | {milliseconds:.2f} | none | "
            + (f"{crossover:.0f} |" if crossover is not None else "never |")
        )

    lines.append("")
    lines.append(f"_{cost['note']} Timed over {cost['tasks_timed']} tasks._")
    return NEWLINE.join(lines)


def _comparison_table(records: list[ExperimentRecord], datasets: list[str]) -> str:
    """Render mean accuracy and calibration per model for the given evaluations."""

    selected = [record for record in records if record.dataset in datasets]
    if not selected:
        return "_No matching records._"

    grouped: dict[tuple[str, str], list[ExperimentRecord]] = {}
    for record in selected:
        grouped.setdefault((record.dataset, record.architecture), []).append(record)

    headers = ["evaluation", "model", "accuracy", "±", "ECE", "seeds"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for (dataset, architecture), group in sorted(grouped.items()):
        scores = [record.primary_metric.value for record in group]
        mean = sum(scores) / len(scores)
        spread = (max(scores) - min(scores)) / 2
        calibration = [
            record.secondary_metrics["expected_calibration_error"]
            for record in group
            if "expected_calibration_error" in record.secondary_metrics
        ]
        ece = sum(calibration) / len(calibration) if calibration else float("nan")
        lines.append(
            f"| {dataset} | {architecture} | {mean:.3f} | {spread:.3f} | {ece:.3f} | {len(group)} |"
        )
    return NEWLINE.join(lines)


def _context_table(records: list[ExperimentRecord]) -> str:
    """Render accuracy against context size — the small-n regime."""

    grouped: dict[tuple[int, str], list[float]] = {}
    for record in records:
        if not _config(record, "in_prior") or _config(record, "label_noise"):
            continue
        if _config(record, "missing_rate") or _config(record, "n_features") != BASE_FEATURES:
            continue
        context = int(_config(record, "n_context", 0) or 0)
        grouped.setdefault((context, record.architecture), []).append(record.primary_metric.value)

    if not grouped:
        return "_No context-sweep records._"

    contexts = sorted({context for context, _ in grouped})
    models = sorted({model for _, model in grouped})

    headers = ["model", *[f"n={context}" for context in contexts]]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for model in models:
        cells = [model]
        for context in contexts:
            scores = grouped.get((context, model))
            cells.append(f"{sum(scores) / len(scores):.3f}" if scores else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return NEWLINE.join(lines)


def _transfer_table() -> str:
    """Render the prior-transfer matrix from the diagnostics artefact."""

    source = RESULTS / "artefacts" / "prior_diagnostics.json"
    if not source.exists():
        return "_No prior diagnostics recorded._"

    payload = json.loads(source.read_text(encoding="utf-8"))
    headers = ["fitted on", "evaluated on", "in prior", "accuracy", "ECE"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in payload["transfer"]:
        lines.append(
            f"| {row['fit_prior']} | {row['eval_prior']} | "
            f"{'yes' if row['in_prior'] else 'no'} | {row['accuracy']:.3f} | "
            f"{row['expected_calibration_error']:.3f} |"
        )
    lines.append("")
    lines.append(f"_{payload['note']}_")
    return NEWLINE.join(lines)


def _reference_status() -> str:
    """Render the recorded status of the official checkpoint (deliverable B)."""

    source = RESULTS / "artefacts" / "prior_diagnostics.json"
    if not source.exists():
        return "_No reference status recorded._"

    reference = json.loads(source.read_text(encoding="utf-8"))["tabpfn_reference"]
    return NEWLINE.join(
        [
            f"- **Executed:** {'yes' if reference['executed'] else 'no'}",
            f"- **Package importable:** {'yes' if reference['package_importable'] else 'no'}",
            f"- **Reason:** {reference['reason']}",
            f"- **Adapter:** `{reference['adapter']}`",
            "",
            f"> {reference['pretraining_advantage_note']}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
