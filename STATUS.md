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
| xLSTM | 0% | 0% | 0% | 0% | 0% | queued |
| Mamba-3 | 0% | 0% | 0% | 0% | 0% | queued |
| TTT | 0% | 0% | 0% | 0% | 0% | queued |
| Titans | 0% | 0% | 0% | 0% | 0% | queued |
| Nested Learning / Hope | 0% | 0% | 0% | 0% | 0% | queued |
| PFN / TabPFN | 0% | 0% | 0% | 0% | 0% | queued |
| Relational FM | 0% | 0% | 0% | 0% | 0% | queued |
| Sparse MoE | 0% | 0% | 0% | 0% | 0% | queued |
| Flow Matching | 0% | 0% | 0% | 0% | 0% | queued |
| JEPA | 0% | 0% | 0% | 0% | 0% | queued |

Milestone **M0 (repository contracts)** is complete. Track 01 (KAN) is the first track
completed end to end and establishes the conventions every later track follows.

## Next atomic milestone

Execute `prompts/02_xlstm.md`: implement compact sLSTM and mLSTM cells with exponential
gating and the normalizer/stabilizer states, add the shared synthetic sequence tasks
(copy, selective recall, state tracking), and compare against LSTM, GRU, and a
matched-parameter causal Transformer.

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

### Corrected defect worth carrying forward

The first version of the KAN suite trained every model at one shared learning rate. The
MLP baseline was badly under-trained at that rate, producing an apparent 1000x advantage
for the KAN that collapsed to roughly 5x once each architecture selected its learning rate
from the same grid on validation data. **Every later track must use per-architecture
learning-rate selection**, and any margin of several orders of magnitude should be treated
as a suspected baseline failure until checked. Details in `reports/kan.md`, section 9.

### Unresolved

- `Tensor.backward` is unannotated in the Torch distribution, so one narrowly scoped
  `# type: ignore[no-untyped-call]` remains in `experiments/training.py` with an inline
  justification.
- Adaptive grid updates are implemented and tested but excluded from the reported KAN
  training loop; they are listed as deviation 4 and as a next experiment.
- Ten architecture tracks remain queued.

### Known environment issue

`poetry install` fails on this Windows host because of path-length limits while installing
Torch metadata. The quality gate is therefore executed with the interpreter's installed
toolchain (`python -m ruff`, `python -m mypy`, `python -m pytest`), which runs the identical
commands CI runs through Poetry.
