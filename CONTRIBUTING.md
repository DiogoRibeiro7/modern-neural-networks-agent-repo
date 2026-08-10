# Contributing

Thank you for considering a contribution. This repository is a research laboratory, so
contributions are judged primarily on **scientific correctness and auditability**, not on
the number of architectures added.

## Ground rules

1. Primary sources win. Implement from the paper and the authors' official repository,
   never from a blog summary, and record what you used in [`docs/source_registry.md`](docs/source_registry.md).
2. Claims must satisfy [`docs/claim_policy.md`](docs/claim_policy.md). Never write "state of
   the art", "reproduces the paper", "faster", or "better" without the evidence that policy
   demands.
3. Every experiment must serialize a machine-readable record that satisfies
   [`docs/experiment_contract.md`](docs/experiment_contract.md).
4. Plots and tables must be regenerable from saved raw results. Hard-coded numbers in a
   report are a defect; use the generated-block markers described in
   `modern_nn_lab.experiments.reporting`.
5. Negative results, divergences, and instabilities are reported, not deleted.

## Development setup

```bash
poetry install
poetry run pre-commit install
```

If you do not use Poetry, an editable install works too:

```bash
python -m pip install -e ".[dev]"
```

## Quality gate

Run the full gate before opening a pull request. CI runs exactly the same commands.

```bash
make check
```

which is equivalent to:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy src
poetry run pytest
```

Fix causes, not symptoms. A `# type: ignore` or `# noqa` requires a written justification on
the same line or immediately above it.

## Adding or extending a track

Follow the phase order used by every completed track (see `reports/kan.md` and
`reports/xlstm.md` for worked examples):

1. **Specification** — `src/modern_nn_lab/tracks/<track>/README.md` with the mathematical
   formulation, tensor-shape table, equation-to-code mapping, complexity, stability notes,
   and explicit deviations from the source.
2. **Tests before implementation** — invariant tests, not forward-pass smoke tests.
3. **Smallest correct model** — transparency before speed. No custom kernels first.
4. **Baseline and diagnostic task** — isolate the claimed mechanism.
5. **Fair experiment** — matched parameter budget, and matched compute where meaningful.
6. **Ablation** — neutralize the new mechanism and measure the effect.
7. **Benchmark** — only after the diagnostics pass.
8. **Report** — `reports/<track>.md` following the required section order.

Public functions and classes need type annotations, docstrings, documented tensor shapes,
and input validation wherever misuse is plausible.

## Commit and pull-request style

- Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`,
  `ci:`, `build:`. Keep the subject line short and in the imperative mood.
- One track (or one piece of shared infrastructure) per pull request.
- Fill in the pull-request template, including the quality-gate results and the claim level
  for any new experimental result.
- Update `STATUS.md` in the same pull request: completed work, exact tests run, unresolved
  failures, files changed, and the next atomic milestone.

## Data

Raw experiment records are committed under `results/<track>/`; derived diagnostics that
do not satisfy the record schema go under `results/<track>/artefacts/`. Never commit large
raw datasets. Download through a documented adapter into `data/raw/`,
which is ignored by Git. Keep raw experiment records small enough to commit under
`results/`, so figures remain reproducible.

## Reporting problems

Open an issue using one of the templates. For anything with a security or supply-chain
dimension, follow [`SECURITY.md`](SECURITY.md) instead.
