"""Command-line interface for repository inspection and experiments."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from modern_nn_lab.registry import TRACKS

app = typer.Typer(no_args_is_help=True)
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


if __name__ == "__main__":
    app()
