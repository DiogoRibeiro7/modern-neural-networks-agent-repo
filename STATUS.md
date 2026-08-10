# Development Status

## Repository state

Scaffold created and professionalized. No architecture track is claimed complete.

Repository-level Zenodo metadata was added in `.zenodo.json` for software archiving. The
initial scaffold was lint-cleaned and pytest was configured for the `src/` layout. The
repository now carries an MIT license, contribution guide, code of conduct, security policy,
citation metadata, issue/pull-request templates, Dependabot, CodeQL analysis, and a split CI
pipeline with a Python 3.11-3.13 test matrix. No architecture track status changed.

The Zenodo metadata was enriched for software archiving: it now includes publication date,
rights-holder metadata, a Zenodo license identifier, repository and documentation related
identifiers, completed-track source references, broader keywords, and a note preserving the
repository's claim boundaries.

| Track | Specification | Core implementation | Tests | Benchmarks | Report | Status |
|---|---:|---:|---:|---:|---:|---|
| KAN | 100% | 100% | 100% | 100% | 100% | complete |
| xLSTM | 100% | 100% | 100% | 100% | 100% | complete |
| Mamba-3 | 100% | 100% | 100% | 100% | 100% | complete |
| TTT | 100% | 100% | 100% | 100% | 100% | complete |
| Titans | 0% | 0% | 0% | 0% | 0% | queued |
| Nested Learning / Hope | 0% | 0% | 0% | 0% | 0% | queued |
| PFN / TabPFN | 0% | 0% | 0% | 0% | 0% | queued |
| Relational FM | 0% | 0% | 0% | 0% | 0% | queued |
| Sparse MoE | 0% | 0% | 0% | 0% | 0% | queued |
| Flow Matching | 0% | 0% | 0% | 0% | 0% | queued |
| JEPA | 0% | 0% | 0% | 0% | 0% | queued |

Milestone **M0 (repository contracts)** is complete. Tracks 01 (KAN), 02 (xLSTM),
03 (Mamba-3), and 04 (TTT) are complete end to end and establish the conventions every later track
follows. Tracks 02 and 03 share task generation, so their records are directly comparable.

## Next atomic milestone

Track 05 (Titans-style neural long-term memory): choose one of the source's three
architecture variants deliberately and document why, implement the surprise/update signal
as the source defines it, and produce explicit memory-write/read diagnostics rather than
task accuracy alone.

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

**Track 03 — Mamba-3** (claim level: `educational implementation`)

- First track whose primary source was retrieved and read in-environment: the recurrence,
  discretization coefficients, and both rotation formulations are transcribed from
  Propositions 1-4 and equations (1)-(14), not reconstructed.
- Added `tracks/mamba3/` (`ssm`, `model`, `config`, `README`), the experiment suite, and
  the report generator. Promoted the shared sequence scaffolding to
  `modern_nn_lab/models/sequence.py` so no track imports another track.
- 22 invariant tests, including four equality assertions against the source's own
  formulas: Table 1's exponential-Euler row, Proposition 1's coefficients, the
  Proposition 2 vs Proposition 3 identity (block rotation vs the RoPE trick), and the
  MIMO decomposition of equations (12)-(14).
- 105 records committed under `results/mamba3/`, all `status="success"`.
- **Pre-registered prediction confirmed**: removing rotation collapses parity accuracy
  from 0.847 to 0.578 with disjoint intervals, and resolves the mLSTM parity failure that
  Track 02 left open (0.589 on identical data).

**Track 04 — TTT** (claim level: `educational implementation`)

- Primary source retrieved and read in-environment; the inner update, the inner learning
  rate, and the batch-gradient-descent instantiation are transcribed from equations (4)-(6),
  Subsection 2.7, and Theorem 1.
- Added `tracks/ttt/` (`layer`, `model`, `config`, `README`), the rebinding task, the
  experiment suite, and the report generator.
- 21 invariant tests, including the source's Theorem 1 (batch GD with a linear inner model
  is exactly causal linear attention) and the track's acceptance criterion: a forward pass
  performs gradient descent while mutating **no** `nn.Parameter`.
- 79 records committed under `results/ttt/`, all `status="success"`.
- The required ablation is decisive: freezing the inner loop drops post-shift accuracy from
  0.371 to 0.221 and puts selective recall exactly at chance, because the layer becomes a
  position-independent function of each token.

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

**A dataset label is not a dataset identity.** The TTT context-scaling study reused the
label `rebinding` for a differently-sized split, and because record filenames keyed only on
`(architecture, variant, dataset, seed)` it silently overwrote three seeds of the main
task — leaving groups that were incomplete *and* mixed two datasets. Filenames now include
the dataset fingerprint, and `scripts/validate_results.py` fails when one aggregation group
contains more than one fingerprint. Details in `reports/ttt.md`, section 8.

### Unresolved

- `Tensor.backward` is unannotated in the Torch distribution, so one narrowly scoped
  `# type: ignore[no-untyped-call]` remains in `experiments/training.py` with an inline
  justification.
- Adaptive grid updates are implemented and tested but excluded from the reported KAN
  training loop; they are listed as deviation 4 and as a next experiment.
- Seven architecture tracks remain queued (Titans, Nested Learning/Hope, PFN/TabPFN,
  Relational FM, Sparse MoE, Flow Matching, JEPA) plus the final integration prompt.
- TTT runs pure online gradient descent (`b = 1`). The source reports mini-batch TTT with
  `b = 16` as its single largest quality gain, so this track's numbers are knowingly below
  the paper's configuration and its online-vs-batch comparison is not a test of the
  source's claim.
- Mamba-3's MIMO contribution **cannot be evaluated by this repository**: its benefit is
  decode-time arithmetic intensity on an accelerator, and a Python scan on CPU has no
  memory-bound decode to improve. Recorded as an untestable claim, not a negative result.
- Mamba-3 ablations are not parameter-matched to the full model (5-25 % gaps), because
  each removes the parameters belonging to its mechanism. The parity conclusion is
  protected by the LSTM comparison; other comparisons are weaker.
- The xLSTM selective-recall diagnostic did not discriminate between any two
  architectures at this budget; it needs more data or a larger memory to be informative.
- The xLSTM sLSTM variant is not width-matched to the mLSTM variant (8578 vs 4614
  parameters). The LSTM comparison resolves the confound for state tracking only.

### Known environment issue

`poetry install` fails on this Windows host because of path-length limits while installing
Torch metadata. The quality gate is therefore executed with the interpreter's installed
toolchain (`python -m ruff`, `python -m mypy`, `python -m pytest`), which runs the identical
commands CI runs through Poetry.

### Last metadata-only verification

- Read Zenodo's current GitHub `.zenodo.json` guidance and deposition metadata field
  documentation.
- Queried Zenodo's license vocabulary endpoint and confirmed the MIT license identifier is
  `mit`.
- Ran `python -m json.tool .zenodo.json`.
- Attempted validation against Zenodo's legacy deposition JSON schema. That schema rejects
  `version` and `language`, even though Zenodo's current GitHub `.zenodo.json` guidance
  includes both fields; the file keeps those software-specific metadata fields.
- Full source quality gate not rerun because only repository archive metadata and status
  documentation changed.
