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
| KAN | 0% | 0% | 0% | 0% | 0% | queued |
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

Milestone **M0 (repository contracts)** is complete: the package installs, the registry
works, the scaffold validator passes, CI runs, and the shared experiment schema is
implemented and enforced.

## Next atomic milestone

Execute `prompts/01_kan.md`: verify primary sources, write the track mathematical
specification, add failing invariant tests, then implement KAN layers on top of the shared
harness.

## Last verification

- Added `modern_nn_lab.experiments` with `records`, `data`, `training`, `evaluation`,
  `profiling`, and `runner` modules.
- Added `results/`, `scripts/validate_results.py`, a CI record-audit job, and the
  `modern-nn summarize` command.
- Added 59 harness tests covering schema immutability and versioning, fingerprint
  sensitivity, split disjointness, train-only standardization, batching alignment,
  training determinism, divergence reporting, refusal to aggregate diverged runs, and
  parameter/latency accounting.
- Ran `python -m ruff check .` — clean.
- Ran `python -m ruff format --check .` — 34 files formatted.
- Ran `python -m mypy src` — no issues in 24 source files.
- Ran `python -m pytest` — 65 passed.
- Ran `python scripts/validate_scaffold.py` and `python scripts/validate_results.py`.

### Unresolved

- `Tensor.backward` is unannotated in the Torch distribution, so one narrowly scoped
  `# type: ignore[no-untyped-call]` remains in `experiments/training.py` with an inline
  justification.

### Known environment issue

`poetry install` fails on this Windows host because of path-length limits while installing
Torch metadata. The quality gate is therefore executed with the interpreter's installed
toolchain (`python -m ruff`, `python -m mypy`, `python -m pytest`), which runs the identical
commands CI runs through Poetry.
