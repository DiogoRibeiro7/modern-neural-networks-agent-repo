# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research repository, entries also record **claim levels** for new
experimental results, as defined in [`docs/claim_policy.md`](docs/claim_policy.md).

## [Unreleased]

### Added

- **Track 09 — Sparse Mixture of Experts** (claim level: `educational implementation`): a
  top-k router with explicit capacity and dropping, a dense-ensemble reference, a dense
  feed-forward baseline, and ablations for top-k, capacity, and the balancing loss; every
  record carries total *and* activated parameters, an analytic FLOP estimate, and a measured
  throughput; a mixture-of-functions task whose generating function is recorded, so expert
  specialization is a confusion matrix rather than an inference; 34 tests including both
  exact endpoints of the balancing loss; and `reports/moe.md`. The required routing
  diagnostics exposed a real defect: renormalizing kept gates under top-1 severs the router
  from the task loss entirely, and fixing it cut error 39% at identical cost.

### Added

- Zenodo concept DOI `10.5281/zenodo.21879430` in `CITATION.cff` (as both `doi` and a
  described `identifiers` entry) and as a README badge. The README now distinguishes the
  concept DOI, for citing the software in general, from a version DOI, which is what a
  reproducibility claim should name.
- A note in the README's citation section stating what citing this repository does not
  assert: no track claims to reproduce a published number, and citing the software is not
  a citation of the papers it implements.

## [0.2.1] - 2026-08-10

**No code or result changes.** This release exists so that the repository's first archived
deposit is created: Zenodo's GitHub integration only captures releases published *after*
the repository is enabled, and it does not archive earlier ones retroactively. Every number
and every record is identical to `v0.2.0`.

### Changed

- `.zenodo.json` now references the primary sources for all six tracks whose papers were
  actually read, rather than the first three. Tracks 07 and 08 remain deliberately
  unreferenced, because no primary source was retrieved for either and citing one would
  imply it had informed the implementation.

## [0.2.0] - 2026-08-10

**Scope.** Eight of eleven architecture tracks are complete end to end (KAN, xLSTM,
Mamba-3, TTT, Titans, Nested Learning, Prior-Fitted Networks, Relational). Sparse MoE,
Flow Matching, JEPA, and the cross-track integration are **not** in this release. Every
committed result carries its claim level, and no track claims a reproduction of a
published number.

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
- **Track 04 — Test-Time Training** (claim level: `educational implementation`): TTT-Linear
  and TTT-MLP whose hidden state is an inner model trained by gradient descent during the
  forward pass; the required frozen-learner ablation; a batch-gradient-descent ablation the
  source proves is linear attention; a new rebinding task that overwrites a binding
  mid-sequence; 21 invariant tests; 79 committed records; `reports/ttt.md`.
- **Track 05 — Titans** (claim level: `educational implementation`): neural long-term
  memory with surprise, momentum, and adaptive forgetting, wired as Memory-as-Gate with
  sliding-window attention; short-term-only, frozen-memory, no-momentum, and slow-update
  ablations; a needle task with controlled write-to-query distance; a memory-diagnostics
  artefact carrying per-token write/read traces and a forgetting curve; 28 invariant tests;
  60 committed records; `reports/titans.md`.
- **Track 06 — Nested Learning** (claim level: `research prototype`): a formalization
  audit written before any code, optimization levels as separately testable objects with
  explicit update frequencies, one-level/two-level/slow/reset/self-referential learners
  over a continual stream, a continual-diagnostics artefact, 24 tests including an exact
  match against `torch.optim.SGD(momentum=0.9)`, 50 committed records, and `reports/hope.md`.
  The Continuum Memory System and the Hope architecture are deliberately not implemented.
- **Track 07 — Prior-Fitted Networks** (claim level: `educational implementation`): a PFN
  that predicts a whole dataset in one forward pass with no per-dataset gradient step, with
  the two properties that make that well posed — queries cannot attend to one another, and
  predictions are invariant to context order — asserted rather than assumed; four
  controllable task priors, of which two are structurally out of reach for the fitted model;
  in-prior, out-of-prior, small-n, label-noise, class-imbalance, missingness, and
  feature-count studies against four per-task baselines; a measured break-even count for the
  PFN's up-front cost; 20 tests; and `reports/pfn.md`. **The official TabPFN checkpoint is
  not executed** — it is license-gated behind an interactive flow — so the adapter ships
  behind an optional extra and no TabPFN number appears anywhere in the track.
- **Track 08 — Relational Foundation Models** (claim level: `research prototype`): a
  prototype that keeps rows as rows — typed columns, foreign-key pointers and timestamps —
  and passes messages along foreign keys instead of flattening the tables first; five
  diagnostic regimes (one-hop, multi-hop, temporal, irrelevant tables, cold start) against a
  homogeneous-GNN baseline, leakage-safe feature engineering with a GBDT, and a
  target-table-only floor, all at matched parameter counts; temporal gating concentrated in
  a single module every model reads from; a planted post-timestamp shortcut that makes the
  leakage tests positive controls rather than assertions; an exact reachable-path trace with
  separate gradient attribution; 45 tests; and `reports/relational.md`. No relational
  foundation model is reproduced or invoked.
- Record filenames now include the dataset fingerprint, and the result validator fails when
  one aggregation group mixes datasets — a label collision had silently overwritten records.
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

[Unreleased]: https://github.com/DiogoRibeiro7/modern-neural-networks-agent-repo/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/DiogoRibeiro7/modern-neural-networks-agent-repo/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/DiogoRibeiro7/modern-neural-networks-agent-repo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/modern-neural-networks-agent-repo/releases/tag/v0.1.0
