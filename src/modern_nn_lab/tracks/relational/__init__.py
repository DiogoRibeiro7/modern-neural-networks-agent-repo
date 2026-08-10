"""Relational foundation-model prototype: learning over linked tables without flattening.

The track's claim is narrow and its scope is deliberately small. This is **not** a
reproduction of any relational foundation model, and no pre-trained relational checkpoint is
invoked anywhere in it. What is built here is a prototype that represents entities, rows,
typed columns, foreign keys and timestamps directly, together with the baselines needed to
tell whether representing them directly buys anything.

The two acceptance criteria set by the track prompt are met by construction rather than by
prose:

- **Temporal leakage** is prevented in exactly one place,
  :mod:`~modern_nn_lab.tracks.relational.sampler`, which every model and every baseline
  reads its inputs from. The generator plants a post-timestamp shortcut that any leaking
  pipeline would find, and the tests assert that none does.
- **Explainability** is provided by :mod:`~modern_nn_lab.tracks.relational.trace`, which
  enumerates the relational paths that can reach a prediction exactly, and separately
  estimates which of them did.
"""

from modern_nn_lab.tracks.relational.config import RelationalConfig, RelationalExperimentConfig
from modern_nn_lab.tracks.relational.features import FEATURE_NAMES, flatten
from modern_nn_lab.tracks.relational.generator import (
    REGIMES,
    PredictionTask,
    Regime,
    RelationalProblem,
    generate,
    leakage_canary_strength,
)
from modern_nn_lab.tracks.relational.model import RelationalEncoder, TargetOnlyModel
from modern_nn_lab.tracks.relational.sampler import (
    ROW_WIDTH,
    Normalizer,
    SamplingConfig,
    build_row_sets,
)
from modern_nn_lab.tracks.relational.schema import Column, Database, ForeignKey, Table
from modern_nn_lab.tracks.relational.trace import PathTrace, attribute, reachable_paths, summarize

__all__ = [
    "FEATURE_NAMES",
    "REGIMES",
    "ROW_WIDTH",
    "Column",
    "Database",
    "ForeignKey",
    "Normalizer",
    "PathTrace",
    "PredictionTask",
    "Regime",
    "RelationalConfig",
    "RelationalEncoder",
    "RelationalExperimentConfig",
    "RelationalProblem",
    "SamplingConfig",
    "Table",
    "TargetOnlyModel",
    "attribute",
    "build_row_sets",
    "flatten",
    "generate",
    "leakage_canary_strength",
    "reachable_paths",
    "summarize",
]
