# Development Status

## Repository state

Scaffold created and professionalized. No architecture track is claimed complete.

Repository-level Zenodo metadata was added in `.zenodo.json` for software archiving. The
initial scaffold was lint-cleaned and pytest was configured for the `src/` layout. The
repository now carries an MIT license, contribution guide, code of conduct, security policy,
citation metadata, issue/pull-request templates, Dependabot, CodeQL analysis, and a split CI
pipeline with a Python 3.11-3.13 test matrix. No architecture track status changed.

| Track | Specification | Core implementation | Tests | Benchmarks | Report | Status |
|---|---:|---:|---:|---:|---:|---|
| KAN | 100% | 100% | 100% | 100% | 100% | complete |
| xLSTM | 100% | 100% | 100% | 100% | 100% | complete |
| Mamba-3 | 0% | 0% | 0% | 0% | 0% | queued |
| TTT | 0% | 0% | 0% | 0% | 0% | queued |
| Titans | 0% | 0% | 0% | 0% | 0% | queued |
| Nested Learning / Hope | 0% | 0% | 0% | 0% | 0% | queued |
| PFN / TabPFN | 0% | 0% | 0% | 0% | 0% | queued |
| Relational FM | 0% | 0% | 0% | 0% | 0% | queued |
| Sparse MoE | 0% | 0% | 0% | 0% | 0% | queued |
| Flow Matching | 0% | 0% | 0% | 0% | 0% | queued |
| JEPA | 0% | 0% | 0% | 0% | 0% | queued |

Milestone **M0 (repository contracts)** is complete. Tracks 01 (KAN) and 02 (xLSTM) are
complete end to end and establish the conventions every later track follows.

## Next atomic milestone

Track 03 (Mamba-3 / modern SSMs): verify the primary source, implement a transparent
selective state-space recurrence, and reuse the shared sequence tasks from Track 02 so the
comparison against xLSTM, LSTM, GRU, and the Transformer is on identical data.

## Last verification

**Track 01 — KAN** (claim level: `educational implementation`)

- Added `tracks/kan/` (`spline`, `layers`, `model`, `config`, `README`), the experiment
  suite, the report generator, and the figure script.
- 37 invariant tests: partition of unity, non-negativity, local support, the degree-1 hat
  function, degree-1 coefficient interpolation, least-squares refit accuracy, adaptive-grid
  monotonicity under a constant feature, hand-computed edge evaluation, forward-pass
  decomposition, gradient finiteness, deterministic initialization, ablation semantics,
  function-preserving grid update, and serialization round-trip.
- 90 records committed under `results/kan/`, all `status="success"`.
- Ran `python -m ruff check .` — clean.
- Ran `python -m ruff format --check .` — 47 files formatted.
- Ran `python -m mypy src` — no issues in 32 source files.
- Ran `python -m pytest` — 102 passed.
- Ran `python scripts/validate_scaffold.py`, `python scripts/validate_results.py`,
  `python scripts/report_kan.py`, `python scripts/plot_kan.py`.

**Track 02 — xLSTM** (claim level: `educational implementation`)

- Added `tracks/xlstm/` (`cells`, `model`, `config`, `README`), the shared sequence-task
  module `experiments/tasks/sequence.py`, the experiment suite, and the report generator.
- Added a `Split` protocol so one runner serves both tabular and sequence splits.
- 41 invariant tests: task construction and scoring masks, causality under future-token
  modification for **all five** architectures, two-step hand-computed sLSTM and mLSTM
  recurrences against unstabilized references, 400-step finiteness, exact reset
  semantics, gate-ablation isolation, determinism, serialization, and width matching.
- 108 records committed under `results/xlstm/`, all `status="success"`.
- Ran the full gate: ruff, ruff format, mypy, pytest, both validators, report generator.

### Corrected defects worth carrying forward

The first version of the KAN suite trained every model at one shared learning rate. The
MLP baseline was badly under-trained at that rate, producing an apparent 1000x advantage
for the KAN that collapsed to roughly 5x once each architecture selected its learning rate
from the same grid on validation data. **Every later track must use per-architecture
learning-rate selection**, and any margin of several orders of magnitude should be treated
as a suspected baseline failure until checked. Details in `reports/kan.md`, section 9.

**Stabilized recurrences must be tested against an unstabilized reference.** The mLSTM
covering rule `max(|n^T q|, 1)` is not scale-invariant, so applying it to the stabilized
state silently computed the wrong function. Shapes, finiteness, and training behaviour
were all unremarkable; only a hand-computed two-step comparison caught it. Every later
track that stabilizes a recurrence must carry the same kind of test. Details in
`reports/xlstm.md`, section 8.

### Unresolved

- `Tensor.backward` is unannotated in the Torch distribution, so one narrowly scoped
  `# type: ignore[no-untyped-call]` remains in `experiments/training.py` with an inline
  justification.
- Adaptive grid updates are implemented and tested but excluded from the reported KAN
  training loop; they are listed as deviation 4 and as a next experiment.
- Nine architecture tracks remain queued (Mamba-3, TTT, Titans, Nested Learning/Hope,
  PFN/TabPFN, Relational FM, Sparse MoE, Flow Matching, JEPA) plus the final integration
  prompt.
- The xLSTM selective-recall diagnostic did not discriminate between any two
  architectures at this budget; it needs more data or a larger memory to be informative.
- The xLSTM sLSTM variant is not width-matched to the mLSTM variant (8578 vs 4614
  parameters). The LSTM comparison resolves the confound for state tracking only.

### Known environment issue

`poetry install` fails on this Windows host because of path-length limits while installing
Torch metadata. The quality gate is therefore executed with the interpreter's installed
toolchain (`python -m ruff`, `python -m mypy`, `python -m pytest`), which runs the identical
commands CI runs through Poetry.
