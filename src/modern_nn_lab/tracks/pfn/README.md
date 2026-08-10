# Prior-Fitted Networks

**Claim level: educational implementation.** A compact PFN built from first principles,
plus an unexecuted adapter for the official TabPFN checkpoint. This is not a reproduction
of TabPFN's reported numbers and does not attempt to be one.

## The idea

A conventional model is fitted to a dataset. A Prior-Fitted Network is fitted to a
*distribution over datasets* — a prior — once, and thereafter answers a new dataset by
reading it as context in a single forward pass. No gradient step happens at prediction
time. Supervised learning becomes a sequence-modelling problem:

```
f ~ p(T),   (x_i, y_i)_{i≤n} ~ f,   (x_q, ?)_{q≤m} ~ f
model(context, query) → p(y_q | x_q, context)
```

Two structural properties make that well posed, and both are asserted in
[tests/test_pfn.py](../../../../tests/test_pfn.py) rather than assumed:

1. **Queries cannot see each other.** Each query attends to the context and to itself,
   never to another query. Without this the model would be doing transduction over the
   whole test set at once, and a "prediction" would depend on which other questions
   happened to be asked alongside it.
2. **Order does not matter.** There is no positional encoding, so permuting the context
   leaves predictions unchanged. A dataset is a set; a model sensitive to row order would
   be exploiting an artefact of how the file was written.

## Layout

| File | What it holds |
| --- | --- |
| [prior.py](prior.py) | `TaskPrior` and the three priors — `linear`, `mlp`, `xor` |
| [model.py](model.py) | `PriorFittedNetwork`: the masked encoder and `predict_proba` |
| [config.py](config.py) | `PFNConfig` (architecture), `PFNExperimentConfig` (suite) |
| [reference.py](reference.py) | Adapter for the official TabPFN checkpoint — **not executed** |

The three priors are nested on purpose, so that "in prior" and "out of prior" are
structural rather than a matter of degree: `linear` labels are the sign of a random
separator; `mlp` labels come from a random two-layer network, so boundaries are curved;
`xor` labels are a parity that *no* linear boundary can express at all.

## What the experiments compare

The PFN is prior-fitted once on `linear` and then never touched again. Every baseline
(logistic regression, random forest, gradient boosting, a small MLP) is fitted from scratch
on each task's context, which is the normal way those models are used. What is held equal
is the *evidence*: identical context rows, identical queries.

That asymmetry — one training cost paid up front on synthetic tasks, versus a training cost
paid per dataset — is the mechanism under study, not a confound. The out-of-prior
evaluations are where it is expected to fail, and the calibration column is the point: an
out-of-prior model that stays confident is worse than one that becomes uncertain.

Parameter counts are deliberately **not** matched. A prior-fitted transformer and a
per-task logistic regression have no comparable notion of capacity, and pretending
otherwise would be a worse distortion than stating the mismatch.

## The official checkpoint

TabPFN 8.x gates its checkpoint download behind interactive browser license acceptance,
which cannot be completed in a non-interactive environment. [reference.py](reference.py) is
written and documented, but **no TabPFN number appears anywhere in this track**, and
nothing here is compared against one. If you have accepted the license locally:

```bash
poetry install --extras tabpfn
```

Any table that does eventually contain both TabPFN and a from-scratch baseline must carry
`pretraining_advantage_note()`; `build_tabpfn_classifier` attaches it to the estimator so
the disclosure travels with the record rather than living only in prose.

## Running it

```bash
modern-nn run-track pfn          # writes results/pfn/
python scripts/report_pfn.py     # regenerates every table in reports/pfn.md
```

See [reports/pfn.md](../../../../reports/pfn.md) for results and
[configs/tracks/07_tabpfn.yaml](../../../../configs/tracks/07_tabpfn.yaml) for the settings.
