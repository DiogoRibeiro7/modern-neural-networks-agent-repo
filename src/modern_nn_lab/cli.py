"""Command-line interface for repository inspection and experiments."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from modern_nn_lab.experiments.evaluation import aggregate_runs
from modern_nn_lab.experiments.records import RESULTS_ROOT, ExperimentRecord, iter_records
from modern_nn_lab.registry import TRACKS

app = typer.Typer(no_args_is_help=True, help="Modern Neural Networks Lab")
console = Console()


@app.command("list-tracks")
def list_tracks() -> None:
    """List architecture tracks and their current scaffold status."""

    table = Table(title="Modern Neural Networks Lab")
    table.add_column("Key")
    table.add_column("Track")
    table.add_column("Domain")
    table.add_column("Target claim")
    table.add_column("Status")

    for track in TRACKS:
        table.add_row(track.key, track.name, track.domain, track.target_claim, track.status)

    console.print(table)


@app.command("summarize")
def summarize(
    results: Path = typer.Option(RESULTS_ROOT, help="Directory holding committed records."),
    track: str | None = typer.Option(None, help="Restrict the summary to one track key."),
) -> None:
    """Aggregate committed experiment records across seeds.

    Runs are grouped by track, architecture, variant, and dataset. Groups containing a
    non-successful run are reported as such instead of being silently averaged.
    """

    if not results.exists():
        console.print(f"[yellow]No results directory at {results}[/yellow]")
        raise typer.Exit(code=0)

    groups: dict[tuple[str, str, str, str], list[ExperimentRecord]] = defaultdict(list)
    for record in iter_records(results):
        if track is not None and record.track != track:
            continue
        key = (record.track, record.architecture, record.variant or "default", record.dataset)
        groups[key].append(record)

    if not groups:
        console.print("[yellow]No records matched.[/yellow]")
        raise typer.Exit(code=0)

    table = Table(title=f"Aggregated results ({results})")
    for column in ("Track", "Architecture", "Variant", "Dataset", "Metric", "Value", "Seeds"):
        table.add_column(column)

    for (track_key, architecture, variant, dataset), records in sorted(groups.items()):
        failures = [record for record in records if record.status != "success"]
        if failures:
            statuses = ", ".join(sorted({record.status for record in failures}))
            table.add_row(
                track_key,
                architecture,
                variant,
                dataset,
                records[0].primary_metric.name,
                f"[red]{statuses}[/red]",
                str(len(records)),
            )
            continue

        aggregate = aggregate_runs(records)
        arrow = "↑" if aggregate.higher_is_better else "↓"
        table.add_row(
            track_key,
            architecture,
            variant,
            dataset,
            f"{aggregate.name} {arrow}",
            f"{aggregate.mean:.4g} ± {aggregate.std:.2g}",
            str(aggregate.n_seeds),
        )

    console.print(table)


if __name__ == "__main__":
    app()
