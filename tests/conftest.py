"""Shared test helpers.

``conftest.py`` is importable by name from every test module, so helpers defined here
do not require the test directory to be a package.
"""

from __future__ import annotations

from typing import Any

from modern_nn_lab.experiments.records import ExperimentRecord, MetricValue, describe_hardware


def make_record(**overrides: Any) -> ExperimentRecord:
    """Build a minimal valid :class:`ExperimentRecord` for schema tests.

    Args:
        **overrides: Fields replacing the defaults below.

    Returns:
        A validated record.
    """

    payload: dict[str, Any] = {
        "track": "kan",
        "architecture": "kan",
        "dataset": "synthetic",
        "split_strategy": "iid",
        "seed": 0,
        "optimizer": "adam",
        "parameter_count": 100,
        "train_wall_clock_s": 1.0,
        "primary_metric": MetricValue(name="test_mse", value=0.25, higher_is_better=False),
        "hardware": describe_hardware("cpu"),
    }
    payload.update(overrides)
    return ExperimentRecord(**payload)
