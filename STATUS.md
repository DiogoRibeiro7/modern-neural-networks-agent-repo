# Development Status

## Repository state

Scaffold created. No architecture track is claimed complete.

Repository-level Zenodo metadata was added in `.zenodo.json` for software archiving. The
initial scaffold was lint-cleaned and pytest was configured for the `src/` layout. No
architecture track status changed.

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

Execute `prompts/01_kan.md`: verify primary sources, write the track mathematical specification, and add failing invariant tests before implementing KAN layers.

## Last verification

- Added `.zenodo.json` metadata.
- Fixed scaffold lint issues in shared contracts, registry metadata, and contract tests.
- Configured pytest to import from `src/`.
- Ran `python -m json.tool .zenodo.json`.
- Ran `poetry run ruff check .`.
- Ran `poetry run ruff format --check .`.
- Ran `poetry run mypy src`.
- Ran `poetry run pytest`.
- `poetry install` was attempted but failed on Windows path-length limits while installing
  Torch metadata; the quality gate was rerun successfully with the existing Poetry
  environment.
