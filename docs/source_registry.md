# Primary Source Registry

This file is a starting point, not permission to skip source verification. The coding agent must check the current version of each primary paper and the authors' official repository before implementing a track.

| Track | Primary source |
|---|---|
| KAN | Liu et al., *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756 |
| xLSTM | Beck et al., *xLSTM: Extended Long Short-Term Memory*, arXiv:2405.04517 |
| Mamba-3 | Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, arXiv:2603.15569v1 (ICLR 2026) — **verified** |
| TTT | Sun et al., *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*, arXiv:2407.04620 — **verified** |
| Titans | Behrouz et al., *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663 — **verified** |
| Nested Learning / Hope | Behrouz et al., *Nested Learning: The Illusion of Deep Learning Architectures*, arXiv:2512.24695 — **partially verified**; sections 7-8 not read, Hope not implemented |
| PFN / TabPFN | Use the current TabPFN primary paper/model report and official package; also implement the PFN concept independently on synthetic tasks |
| Relational FM | Hudovernik et al., *KumoRFM-2: Scaling Foundation Models for Relational Learning*, arXiv:2604.12596 |
| Sparse MoE | Select and record one canonical routing paper plus one recent primary source before implementation |
| Flow Matching | Lipman et al., *Flow Matching for Generative Modeling*, arXiv:2210.02747 |
| JEPA | Select the relevant primary JEPA/V-JEPA paper and current authors' implementation before implementation |

## Per-track verification records

### Track 01 — KAN

| Field | Value |
|---|---|
| Primary source | Liu et al., *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756 |
| Formulation used | Residual edge function `phi(x) = w_b·silu(x) + spline(x)` with B-spline `spline` on a `k`-extended uniform knot vector |
| Official repository | `KindXiaoming/pykan` (not vendored, not invoked; no reference-integration claim is made) |
| Reference comparison | **None run.** No number in `reports/kan.md` is compared against the paper's or the official implementation's. |
| Datasets | Two synthetic function families defined in-repo; `sklearn.datasets.load_diabetes` (BSD-3-Clause, ships with scikit-learn, no download) |
| Deviations | Six, enumerated in [`src/modern_nn_lab/tracks/kan/README.md`](../src/modern_nn_lab/tracks/kan/README.md#known-approximations-and-deviations-from-the-primary-source) |
| Claim level | `educational implementation` |

**Verification limitation, stated plainly.** The implementation follows the formulation as
recorded above, and the spline machinery is verified against hand-computable cases
(partition of unity, local support, the degree-1 hat function, degree-1 coefficient
interpolation) rather than against the authors' code. The paper was not re-downloaded and
diffed inside this environment, and no output was compared numerically against the
official implementation. That is exactly why the claim level is
`educational implementation` and not `compact reproduction`: this track claims a correct
implementation of the described mechanism, not a reproduction of the paper's results.

### Track 02 — xLSTM

| Field | Value |
|---|---|
| Primary source | Beck et al., *xLSTM: Extended Long Short-Term Memory*, arXiv:2405.04517 |
| Formulation used | sLSTM (scalar memory, memory mixing) and mLSTM (matrix memory, no mixing), both with exponential input gating, a normalizer state, and the running-maximum stabilizer |
| Official repository | `NX-AI/xlstm` (not vendored, not invoked; no reference-integration claim is made) |
| Reference comparison | **None run.** No number in `reports/xlstm.md` is compared against the paper's or the official implementation's. |
| Datasets | Three synthetic sequence tasks defined in-repo (copy, selective recall, state tracking). No external data. |
| Deviations | Five, enumerated in [`src/modern_nn_lab/tracks/xlstm/README.md`](../src/modern_nn_lab/tracks/xlstm/README.md#known-approximations-and-deviations-from-the-primary-source) |
| Claim level | `educational implementation` |

**Verification limitation, stated plainly.** The recurrences are verified step-by-step
against reference implementations of the published equations written directly in the
tests, and the stabilizer is verified to keep 400-step sequences finite. They were not
compared numerically against the authors' code, the paper was not re-downloaded and
diffed inside this environment, and no language-modelling experiment was run. The most
consequential omission is the systems one: this implementation steps the recurrence in
Python, so **no throughput number in this track is evidence about xLSTM's achievable
speed**.

### Track 03 — Mamba-3

| Field | Value |
|---|---|
| Primary source | Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu, *Mamba-3: Improved Sequence Modeling using State Space Principles*, arXiv:2603.15569v1, ICLR 2026 |
| Retrieved | The PDF was fetched and read inside this environment. Sections 2-3.3 and Table 1 were read directly; the equations in the track README are transcribed from Propositions 1-4 and equations (1)-(14), not reconstructed from a summary. |
| Formulation used | Exponential-trapezoidal discretization (Prop. 1, eqs. 5-6); complex dynamics via data-dependent rotary embeddings (Props. 2-3, eqs. 8-10); MIMO state update (eqs. 12-14) |
| Official repository | Triton kernels are referenced by the paper; **not vendored and not invoked**. No reference-integration claim is made. |
| Reference comparison | **None run.** No number in `reports/mamba3.md` is compared against the paper's. |
| Datasets | The same three synthetic sequence tasks as Track 02, generated in-repo with identical parameters and seed. No external data. |
| Deviations | Six, enumerated in [`src/modern_nn_lab/tracks/mamba3/README.md`](../src/modern_nn_lab/tracks/mamba3/README.md#known-approximations-and-deviations-from-the-primary-source) |
| Claim level | `educational implementation` |

**What was verified and what was not.** The recurrence, the discretization coefficients,
and the two rotation formulations were transcribed from the paper and are checked against
each other in the tests: the Euler ablation reproduces Table 1's exponential-Euler row,
the trapezoidal coefficients match Proposition 1 term by term, the RoPE form is asserted
equal to the block-rotation form, and the MIMO update is asserted equal to the sum of
rank-1 recurrences from equations (12)-(14).

Not verified: appendices A-B (the proofs), the released Triton kernels, the block
architecture of Section 3.4, and every experimental result in the paper. Sections 4
onwards were not read. The systems claims — arithmetic intensity, decode latency, kernel
benchmarks — are **untestable in this implementation** and no attempt is made to evaluate
them.

### Track 04 — Test-Time Training

| Field | Value |
|---|---|
| Primary source | Sun, Li, Dalal, Xu, Vikram, Zhang, Dubois, Chen, Wang, Koyejo, Hashimoto, Guestrin, *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*, arXiv:2407.04620 |
| Retrieved | The PDF was fetched and read inside this environment. Sections 2.1-2.7 and Theorems 1-2 were read directly; the equations in the track README are transcribed from equations (4)-(6) and Subsection 2.7, not reconstructed from a summary. |
| Formulation used | Online gradient descent on the multi-view reconstruction loss (eq. 4), output rule (eq. 5), `f(x) = x + LN(f_res(x))`, learnable `W_0` and learnable token-dependent inner learning rate (Subsection 2.7) |
| Official repository | Referenced by the paper (JAX/EasyLM); **not vendored and not invoked**. No reference-integration claim is made. |
| Reference comparison | **None run.** No number in `reports/ttt.md` is compared against the paper's. |
| Datasets | A new in-repo rebinding task plus the selective-recall task shared with Tracks 02 and 03. No external data. |
| Deviations | Five, enumerated in [`src/modern_nn_lab/tracks/ttt/README.md`](../src/modern_nn_lab/tracks/ttt/README.md#known-approximations-and-deviations-from-the-primary-source) |
| Claim level | `educational implementation` |

**What was verified and what was not.** The inner update is checked against a hand-derived
gradient, the inner learning rate against the source's formula, and the batch-gradient-descent
instantiation against the source's Theorem 1 (it must equal causal linear attention exactly).

Not verified: Appendix A (the dual form for nonlinear `f`), Appendix B, the released
kernels, and every experimental result in the paper. Sections 3 onwards were not read
beyond the protocol description. **Deviation 1 matters for interpretation**: the source's
own ablation reports that moving from batch GD to mini-batch GD with `b = 16` is its single
largest quality improvement, and this implementation uses `b = 1` (pure online GD), so it
should be expected to underperform the paper's configuration on quality, not only on speed.

### Track 05 — Titans

| Field | Value |
|---|---|
| Primary source | Behrouz, Zhong, Mirrokni, *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663 |
| Retrieved | The PDF was fetched and read inside this environment. Sections 3.1-3.3 and 4.1-4.3 were read directly; equations 8, 11-15, 19 and 26-31 are transcribed, not reconstructed. |
| Variant implemented | **Memory as a Gate (MAG)**, Section 4.2, equations 26-28. MAC (4.1) and MAL (4.3) are not implemented; the choice and its cost are argued in the track README. |
| Official repository | Not vendored and not invoked. No reference-integration claim is made. |
| Reference comparison | **None run.** No number in `reports/titans.md` is compared against the paper's. |
| Datasets | A new in-repo needle task with controlled write-to-query distance. No external data. |
| Deviations | Five, enumerated in [`src/modern_nn_lab/tracks/titans/README.md`](../src/modern_nn_lab/tracks/titans/README.md#known-approximations-and-deviations-from-the-primary-source) |
| Claim level | `educational implementation` |

**What was verified and what was not.** Each term of the update rule is checked in
isolation against the source's equations: the associative loss, the first write, momentum
accumulation, the reduction to equation 8 when momentum is disabled, the forgetting gate
at `alpha = 1`, and purely additive writes when forgetting is disabled.

Not verified: the chunked parallel formulation of Section 3.2, the appendices, and every
experimental result in the paper. **Deviation 1 is material and was forced by measurement,
not preference**: the update rule as literally written diverges to `NaN` within five
tokens at this scale, because `theta_t` has no specified numeric range. Bounding it,
normalizing keys, and averaging the loss over features makes it stable. A reader should
treat the resulting dynamics as *a* stable instantiation of the source's rule, not as the
authors' configuration.

### Track 06 — Nested Learning

| Field | Value |
|---|---|
| Primary source | Behrouz, Razaviyayn, Zhong, Mirrokni, *Nested Learning: The Illusion of Deep Learning Architectures*, arXiv:2512.24695 |
| Retrieved | The PDF was fetched and read inside this environment. **Sections 2-4.2 and 5 were read**; equations 1, 8-13, 58-60 and 64-65 are transcribed. **Sections 7 and 8 were NOT read.** |
| Formulation used | The level decomposition of gradient descent and its momentum variant, the Hebbian and Delta inner rules, and the Generalized Gradient Descent self-referential rule |
| Official repository | Not vendored and not invoked. No reference-integration claim is made. |
| Reference comparison | **None run.** |
| Datasets | A synthetic continual stream of linear tasks, generated in-repo. No external data. |
| Claim level | `research prototype` — the weakest level in the policy, and the correct one here |

**What is deliberately absent.** The Continuum Memory System (§7) and the Hope
architecture (§8) are **not implemented**, because those sections were not read and the
track prompt permits that stage only if it can be mapped unambiguously to the source. **No
model in this track is named Hope**, and no result here bears on the source's headline
architectural contribution. The formalization audit in
[`docs/nested_learning_audit.md`](nested_learning_audit.md) records this decision and its
reasoning before any code was written, as the prompt requires.

**What was verified.** Each level's state transition is tested independently, and the
two-level composition is asserted equal to `torch.optim.SGD(momentum=0.9)` step for step —
which is the falsifiable content of the source's claim that momentum is a second
optimization level.

### Track 07 — Prior-Fitted Networks

| Field | Value |
|---|---|
| Primary source | **Not retrieved.** No paper was fetched or read in this environment for this track |
| Formulation used | The PFN protocol as described in the track prompt: a predictor trained over tasks sampled from a synthetic prior, conditioning on a labelled context at inference with no per-dataset gradient step |
| Official package | `tabpfn` — installed once to establish the nature of the blocker, then uninstalled. **Never executed** |
| Reference comparison | **None run.** See below |
| Datasets | Synthetic task priors generated in-repo (`linear`, `imbalanced`, `mlp`, `xor`). No external data |
| Claim level | `educational implementation` |

**Why deliverable B is absent.** TabPFN 8.2.0 gates its checkpoint download behind an
interactive browser license acceptance (`ensure_license_accepted` → `try_browser_login`),
which cannot complete non-interactively. Circumventing a license gate was not an option.
The adapter in `modern_nn_lab.tracks.pfn.reference` is written and documented but **not
executed**, the `tabpfn` extra is optional, and **no TabPFN number appears anywhere in this
track**. The track's acceptance criterion — never compare the pre-trained checkpoint with a
from-scratch neural baseline without discussing pre-training compute and data advantages —
is therefore satisfied vacuously; `pretraining_advantage_note()` holds the text that must
accompany any such table if the adapter is ever run.

**What this means for the claim level.** Because no primary source was read, no equation
here is checked against a paper, and nothing in this track may be cited as evidence about
TabPFN. What *is* verified is structural: that queries cannot attend to one another, that
predictions are invariant to context order, and that prediction moves no parameter — the
three properties that make in-context prediction well posed, each asserted in
`tests/test_pfn.py`.

## Source verification checklist

For every track, record:

- paper version/date;
- official repository URL;
- commit/tag used for reference comparisons;
- license;
- checkpoint license if different;
- datasets and licenses;
- equations/sections used for implementation;
- known errata or author corrections;
- deviations in this repository.
