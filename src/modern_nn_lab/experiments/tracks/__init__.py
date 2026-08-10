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

    if key == "kan":
        from modern_nn_lab.experiments.tracks import kan

        return kan.run

    if key == "xlstm":
        from modern_nn_lab.experiments.tracks import xlstm

        return xlstm.run

    if key == "mamba3":
        from modern_nn_lab.experiments.tracks import mamba3

        return mamba3.run

    if key == "ttt":
        from modern_nn_lab.experiments.tracks import ttt

        return ttt.run

    if key == "titans":
        from modern_nn_lab.experiments.tracks import titans

        return titans.run

    if key == "hope":
        from modern_nn_lab.experiments.tracks import hope

        return hope.run

    raise KeyError(f"track {key!r} has no runnable experiment suite yet")


def default_output_dir(key: str) -> Path:
    """Return the conventional results directory for a track.

    Args:
        key: Registry key.

    Returns:
        ``results/<key>``.
    """

    return Path("results") / key
