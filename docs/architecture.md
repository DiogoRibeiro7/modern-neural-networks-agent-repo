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

## Experiment layer to add during development

```text
src/modern_nn_lab/experiments/
├── runner.py
├── training.py
├── evaluation.py
├── profiling.py
└── tasks/
```

The agent should not introduce a large framework prematurely. Add shared abstractions only after two tracks demonstrate the same need.
