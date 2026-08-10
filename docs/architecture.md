# Repository Architecture

The repository uses a strict separation between **mechanism**, **task**, **experiment**, and **reporting**.

```text
Dataset adapter ──> Task contract ──> Model ──> Trainer/Evaluator ──> Raw result
                                          │
                                          └────────> Complexity/profiling

Raw result ──> Aggregation ──> Statistical comparison ──> Plot/table/report
```

## Shared layer

`modern_nn_lab.contracts` defines typed data structures and model/experiment protocols.

`modern_nn_lab.registry` exposes architecture tracks without importing heavyweight optional dependencies.

`modern_nn_lab.reproducibility` centralizes seed behavior and deterministic settings.

## Track layer

Each track should eventually contain:

```text
tracks/<track>/
├── __init__.py
├── model.py
├── layers.py               # when needed
├── math.py                 # explicit mathematical/numerical helpers
├── config.py               # typed track-specific config
├── baselines.py
└── README.md               # theory-to-code mapping
```

Track code must not know where results are stored. Experiment code handles I/O.

## Experiment layer

```text
src/modern_nn_lab/experiments/
├── records.py      # versioned ExperimentRecord schema, fingerprints, provenance
├── data.py         # leakage-safe splits and training-only standardization
├── training.py     # the single shared supervised loop
├── evaluation.py   # metrics, bootstrap intervals, across-seed aggregation
├── profiling.py    # parameter/activated-parameter counts, latency, peak memory
└── runner.py       # multi-seed orchestration; the only writer of records
```

Deliberately small. It exists to guarantee three properties that cannot be enforced by
convention alone:

1. **A target and its baseline are optimized by identical code.** `train_supervised` is
   the only training loop, so an apparent quality difference cannot come from a
   different loop, schedule, or clipping rule.
2. **No record can silently omit a contract field.** `runner.run_seeded_experiment` is
   the only place that constructs an `ExperimentRecord`.
3. **Failures survive.** A non-finite loss produces a record with `status="diverged"`;
   `aggregate_runs` refuses to average over it.

Split responsibilities:

| Concern | Owner | Never does |
|---|---|---|
| Mechanism | `tracks/<track>/` | I/O, result storage, seeding policy |
| Data and splits | `experiments/data.py` | model-specific preprocessing |
| Optimization | `experiments/training.py` | metric selection, storage |
| Evidence | `experiments/records.py`, `runner.py` | model construction |

Add further shared abstractions only after two tracks demonstrate the same need.

## Result storage

```text
results/<track>/<architecture>__<variant>__<dataset>__seed<k>.json
```

Records are committed. Figures under `artifacts/` are ignored by Git and must be
regenerable from `results/`. `scripts/validate_results.py` re-validates every committed
record against the current schema in CI, so a schema change cannot silently orphan past
evidence.
