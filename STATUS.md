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

## Next atomic milestone

Implement the shared experiment harness required by `docs/experiment_contract.md`
(experiment records, seeded training loop, profiling, and shared synthetic tasks) so that
track results are comparable from the first track onward. Then execute `prompts/01_kan.md`.

## Last verification

- Added `LICENSE` (MIT), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`,
  `CHANGELOG.md`, `CITATION.cff`, `.editorconfig`, and `.gitattributes`.
- Added issue forms, a pull-request template, `CODEOWNERS`, Dependabot, and CodeQL analysis.
- Split CI into `static` and `test` jobs, added a 3.11/3.12/3.13 matrix, CPU-only Torch
  installation, coverage reporting, concurrency cancellation, and least-privilege permissions.
- Extended Ruff rule selection (`N`, `C4`, `PT`, `RET`, `ARG`, `PL`, `D`), documented each
  ignore, and added coverage configuration and pytest markers.
- Ran `python -m ruff check .`.
- Ran `python -m ruff format --check .`.
- Ran `python -m mypy src`.
- Ran `python -m pytest`.
- Ran `python scripts/validate_scaffold.py`.

### Known environment issue

`poetry install` fails on this Windows host because of path-length limits while installing
Torch metadata. The quality gate is therefore executed with the interpreter's installed
toolchain (`python -m ruff`, `python -m mypy`, `python -m pytest`), which runs the identical
commands CI runs through Poetry.
