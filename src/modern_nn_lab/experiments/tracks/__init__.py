"""Runnable experiment suites, one module per architecture track.

Each module exposes ``run(output_dir, *, quick)`` and is dispatched by
``modern-nn run-track <key>``. Track modules own *protocol* — which models, which
diagnostics, which ablations — and delegate all training, profiling, and serialization to
:mod:`modern_nn_lab.experiments`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

RunTrack = Callable[..., None]
"""Signature of a track suite: ``run(output_dir: Path, *, quick: bool) -> None``."""


def _module(key: str) -> RunTrack:
    """Import one track suite and return its ``run`` callable.

    Args:
        key: Registry key.

    Returns:
        The suite's ``run`` function.
    """

    import importlib

    module = importlib.import_module(f"modern_nn_lab.experiments.tracks.{key}")
    runner: RunTrack = module.run
    return runner


def get_track_suite(key: str) -> RunTrack:
    """Return the experiment suite for a track key.

    Imports are deferred so that listing tracks never pays for importing every model.

    Args:
        key: Registry key such as ``"kan"``.

    Returns:
        The track's ``run`` callable.

    Raises:
        KeyError: If the track has no runnable suite yet.
    """

    # A lazy lookup table: importing every track's models to answer "which suites exist"
    # would make `modern-nn list-tracks` pay for all of them.
    loaders: dict[str, Callable[[], RunTrack]] = {
        "kan": lambda: _module("kan"),
        "xlstm": lambda: _module("xlstm"),
        "mamba3": lambda: _module("mamba3"),
        "ttt": lambda: _module("ttt"),
        "titans": lambda: _module("titans"),
        "hope": lambda: _module("hope"),
        "pfn": lambda: _module("pfn"),
        "relational": lambda: _module("relational"),
        "moe": lambda: _module("moe"),
        "flow": lambda: _module("flow"),
    }
    if key in loaders:
        return loaders[key]()

    raise KeyError(f"track {key!r} has no runnable experiment suite yet")


def default_output_dir(key: str) -> Path:
    """Return the conventional results directory for a track.

    Args:
        key: Registry key.

    Returns:
        ``results/<key>``.
    """

    return Path("results") / key
