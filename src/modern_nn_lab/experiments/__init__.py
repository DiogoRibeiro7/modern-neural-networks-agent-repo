"""Experiment orchestration, profiling, and result serialization.

Track packages must not import from each other, and must not perform I/O. Everything
that turns a model into evidence lives here:

- :mod:`~modern_nn_lab.experiments.records` — the versioned result schema;
- :mod:`~modern_nn_lab.experiments.data` — leakage-safe splits;
- :mod:`~modern_nn_lab.experiments.training` — the single shared training loop;
- :mod:`~modern_nn_lab.experiments.evaluation` — metrics and across-seed aggregation;
- :mod:`~modern_nn_lab.experiments.profiling` — capacity and measured cost;
- :mod:`~modern_nn_lab.experiments.runner` — multi-seed orchestration.
"""

from modern_nn_lab.experiments.data import SupervisedSplit, make_function_split, make_tabular_split
from modern_nn_lab.experiments.evaluation import Aggregate, aggregate_runs, aggregate_values
from modern_nn_lab.experiments.profiling import (
    ParameterProfile,
    profile_inference,
    profile_parameters,
)
from modern_nn_lab.experiments.records import (
    RESULT_SCHEMA_VERSION,
    ExperimentRecord,
    MetricValue,
    iter_records,
    load_record,
    save_record,
)
from modern_nn_lab.experiments.runner import (
    DEFAULT_SEEDS,
    RunGroup,
    RunSpec,
    run_seeded_experiment,
)
from modern_nn_lab.experiments.training import TrainingConfig, TrainingOutcome, train_supervised

__all__ = [
    "DEFAULT_SEEDS",
    "RESULT_SCHEMA_VERSION",
    "Aggregate",
    "ExperimentRecord",
    "MetricValue",
    "ParameterProfile",
    "RunGroup",
    "RunSpec",
    "SupervisedSplit",
    "TrainingConfig",
    "TrainingOutcome",
    "aggregate_runs",
    "aggregate_values",
    "iter_records",
    "load_record",
    "make_function_split",
    "make_tabular_split",
    "profile_inference",
    "profile_parameters",
    "run_seeded_experiment",
    "save_record",
    "train_supervised",
]
