# Primary Source Registry

This file is a starting point, not permission to skip source verification. The coding agent must check the current version of each primary paper and the authors' official repository before implementing a track.

| Track | Primary source |
|---|---|
| KAN | Liu et al., *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756 |
| xLSTM | Beck et al., *xLSTM: Extended Long Short-Term Memory*, arXiv:2405.04517 |
| Mamba-3 | Lahoti et al., *Mamba-3: Improved Sequence Modeling using State Space Principles*, arXiv:2603.15569 |
| TTT | Sun et al., *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*, arXiv:2407.04620 |
| Titans | Behrouz et al., *Titans: Learning to Memorize at Test Time*, arXiv:2501.00663 |
| Nested Learning / Hope | Behrouz et al., *Nested Learning: The Illusion of Deep Learning Architectures*, arXiv:2512.24695 |
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
