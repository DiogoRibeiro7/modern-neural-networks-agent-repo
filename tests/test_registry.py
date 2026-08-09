"""Tests for lightweight repository metadata."""

from modern_nn_lab.registry import TRACKS, get_track


def test_track_keys_are_unique() -> None:
    """Registry keys must remain unique."""

    keys = [track.key for track in TRACKS]
    assert len(keys) == len(set(keys))


def test_expected_number_of_initial_tracks() -> None:
    """The initial research program contains eleven tracks."""

    assert len(TRACKS) == 11


def test_get_track_returns_requested_track() -> None:
    """Track lookup should be exact and deterministic."""

    assert get_track("kan").name == "Kolmogorov-Arnold Networks"
