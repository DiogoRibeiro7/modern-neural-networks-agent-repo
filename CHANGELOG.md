# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research repository, entries also record **claim levels** for new
experimental results, as defined in [`docs/claim_policy.md`](docs/claim_policy.md).

## [Unreleased]

### Added

- Shared experiment harness under `modern_nn_lab.experiments`: versioned
  `ExperimentRecord` schema, leakage-safe splits, one shared supervised training loop,
  metrics with bootstrap intervals, capacity/latency profiling, and multi-seed
  orchestration that is the sole writer of records.
- `results/` for committed raw records, `scripts/validate_results.py`, a CI job that
  re-validates every record against the current schema, and `modern-nn summarize`.
- **Track 01 — Kolmogorov-Arnold Networks** (claim level: `educational implementation`):
  B-spline machinery, KAN layer and network, matched-budget MLP baseline, two ablations,
  hyperparameter sensitivity sweeps, a tabular benchmark against tree ensembles, 37
  invariant tests, 90 committed records, and `reports/kan.md`.
- **Track 02 — xLSTM** (claim level: `educational implementation`): sLSTM and mLSTM cells
  with exponential gating, normalizer state, and a running-maximum stabilizer; LSTM, GRU,
  and causal-Transformer baselines matched by width to the xLSTM parameter budget; the
  gating ablation; shared copy / selective-recall / state-tracking diagnostics; a
  context-scaling study; 41 invariant tests; 108 committed records; and `reports/xlstm.md`.
- **Track 03 — Mamba-3** (claim level: `educational implementation`): exponential-trapezoidal
  discretization, complex-valued dynamics as data-dependent rotary embeddings, and a MIMO
  state update, each removable by a flag that recovers the prior method exactly; LSTM, GRU,
  and Transformer baselines matched by width; 22 invariant tests including four equality
  assertions against the source's own equations; 105 committed records; `reports/mamba3.md`.
- `modern_nn_lab.models.sequence`: shared token-sequence scaffolding and baselines,
  promoted out of the xLSTM track once a second track needed them.
- A `Split` protocol so one experiment runner serves tabular and sequence splits alike.
- `modern-nn run-track`, per-track experiment suites, non-Torch baseline records
  (`experiments/external.py`), and marker-based report generation
  (`experiments/reporting.py`).

- MIT license, contribution guide, code of conduct, security policy, and citation metadata.
- Issue and pull-request templates, `CODEOWNERS`, and Dependabot configuration.
- CodeQL analysis workflow and an extended CI matrix with coverage reporting.
- `.editorconfig` and `.gitattributes` for consistent cross-platform checkouts.

### Changed

- Pre-commit now runs the standard hygiene hooks and `mypy` alongside Ruff.
- `pyproject.toml` declares project URLs, classifiers, keywords, and coverage settings.

## [0.1.0] - 2026-08-08

### Added

- Initial research scaffold: package layout, track registry, typed contracts,
  reproducibility helpers, CLI, scaffold validator, and CI.
- Repository documentation: architecture, benchmark protocol, claim policy, experiment
  contract, mathematical notation, milestones, source registry, and track matrix.
- Track prompts for eleven architecture tracks plus the final integration prompt.

[Unreleased]: https://github.com/DiogoRibeiro7/modern-neural-networks-agent-repo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/modern-neural-networks-agent-repo/releases/tag/v0.1.0
