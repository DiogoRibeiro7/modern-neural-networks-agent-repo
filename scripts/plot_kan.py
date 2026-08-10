"""Regenerate every KAN figure from saved records.

The script reads only files under ``results/kan/`` and writes PNGs under
``artifacts/kan/``. It never trains anything: if a figure cannot be produced from saved
records, that is a reproducibility defect to fix in the experiment, not here.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402  (path bootstrap must run first)

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from modern_nn_lab.experiments.records import ExperimentRecord, iter_records  # noqa: E402

RESULTS = ROOT / "results" / "kan"
ARTIFACTS = ROOT / "artifacts" / "kan"


def load() -> list[ExperimentRecord]:
    """Return every committed KAN record, or an empty list when none exist."""

    if not RESULTS.exists():
        return []
    return [record for record in iter_records(RESULTS) if record.status == "success"]


def plot_edge_functions(path: Path) -> Path | None:
    """Plot the learned first-layer edge functions from the exported JSON."""

    source = RESULTS / "artefacts" / "edge_functions__compositional.json"
    if not source.exists():
        return None

    payload = json.loads(source.read_text(encoding="utf-8"))
    samples = payload["samples"]
    edges = payload["edges"]  # [out][in][sample]

    n_out, n_in = len(edges), len(edges[0])
    figure, axes = plt.subplots(
        n_out, n_in, figsize=(3.0 * n_in, 1.9 * n_out), squeeze=False, sharex=True
    )
    for out_index in range(n_out):
        for in_index in range(n_in):
            axis = axes[out_index][in_index]
            axis.plot(samples, edges[out_index][in_index], linewidth=1.6)
            axis.axhline(0.0, color="0.8", linewidth=0.7, zorder=0)
            axis.set_title(f"phi[{out_index},{in_index}]", fontsize=8)
            axis.tick_params(labelsize=7)
    figure.suptitle(
        f"Learned first-layer edge functions — {payload['dataset']} (seed {payload['seed']})"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_comparison(records: list[ExperimentRecord], path: Path) -> Path | None:
    """Plot per-seed test error for each architecture and variant, grouped by dataset."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.config.get("sweep") is not None:
            continue
        label = f"{record.architecture}/{record.variant or 'default'}"
        grouped[record.dataset][label].append(record.primary_metric.value)

    if not grouped:
        return None

    datasets = sorted(grouped)
    figure, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.0), squeeze=False)
    for index, dataset in enumerate(datasets):
        axis = axes[0][index]
        labels = sorted(grouped[dataset])
        for position, label in enumerate(labels):
            values = grouped[dataset][label]
            axis.scatter([position] * len(values), values, s=22, zorder=3)
            axis.scatter([position], [sum(values) / len(values)], marker="_", s=400, zorder=4)
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        axis.set_yscale("log")
        axis.set_ylabel("test MSE (log scale, lower is better)")
        axis.set_title(dataset, fontsize=9)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("KAN vs baselines and ablations — individual seeds and mean")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_sensitivity(records: list[ExperimentRecord], path: Path) -> Path | None:
    """Plot the grid-size, spline-order, and regularization sweeps."""

    sweeps: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        sweep = record.config.get("sweep")
        if sweep == "grid_size":
            sweeps["grid size G"][float(record.config["grid_size"])].append(
                record.primary_metric.value
            )
        elif sweep == "spline_order":
            sweeps["spline order k"][float(record.config["spline_order"])].append(
                record.primary_metric.value
            )
        elif sweep == "regularization":
            sweeps["L1 weight"][float(record.config["regularization_weight"])].append(
                record.primary_metric.value
            )

    if not sweeps:
        return None

    names = list(sweeps)
    figure, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.6), squeeze=False)
    for index, name in enumerate(names):
        axis = axes[0][index]
        points = sorted(sweeps[name])
        means = [sum(sweeps[name][p]) / len(sweeps[name][p]) for p in points]
        for value in points:
            seed_values = sweeps[name][value]
            axis.scatter([value] * len(seed_values), seed_values, s=18, alpha=0.6)
        axis.plot(points, means, marker="o", linewidth=1.5)
        axis.set_xlabel(name)
        axis.set_ylabel("test MSE")
        axis.set_yscale("log")
        axis.grid(alpha=0.3)
    figure.suptitle("KAN hyperparameter sensitivity (compositional task, all seeds)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def main() -> int:
    """Regenerate all figures.

    Returns:
        ``0`` on success, ``1`` when no records were found.
    """

    records = load()
    if not records:
        print(f"No successful records under {RESULTS}. Run `modern-nn run-track kan` first.")
        return 1

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    produced = [
        plot_comparison(records, ARTIFACTS / "comparison.png"),
        plot_sensitivity(records, ARTIFACTS / "sensitivity.png"),
        plot_edge_functions(ARTIFACTS / "edge_functions.png"),
    ]
    for figure_path in produced:
        if figure_path is not None:
            print(f"wrote {figure_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
