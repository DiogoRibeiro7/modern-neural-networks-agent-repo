# Track 06 — Nested Learning

**Claim level: `research prototype`** (see [`docs/claim_policy.md`](../../../../docs/claim_policy.md)).

The **formalization audit** is the primary artefact of this track and was written before
any code: [`docs/nested_learning_audit.md`](../../../../docs/nested_learning_audit.md).
Read it first — this file is the code-level companion.

## Problem statement

Nested Learning claims that a machine-learning model *is* a set of nested optimization
problems, each with its own state, objective, and update frequency, and that the apparent
heterogeneity of deep-learning components is an artefact of not seeing that axis.

Most of that claim is a reframing, and a reframing cannot be tested. One consequence is
sharp enough to test:

> Gradient descent **with momentum** is a *two-level* associative memory: the inner level
> compresses gradients into its state, and the outer level updates the slow weights with
> the inner state's value.

If that is right, then "add an optimization level" and "use momentum" must be **the same
edit**, and the second level's contents must behave like stored knowledge. Both are
checkable, and this track checks them.

## Mathematical formulation

```text
one level   W_{t+1} = W_t - eta grad_W L(W_t; x_t)                         (eqs. 1, 8)
                    = W_t - eta [grad_y L] (x) x_t

two levels  m_{t+1} = m_t + eta grad_W L(W_t; x_{t+1})                     (eq. 11)
            W_{t+1} = W_t - m_{t+1}                                        (eq. 10)

self-ref    W_{t+1} = W_t + eta u_t (x) x_t,  u_t = f_{W_t}(x_t)           (eqs. 58-60)
```

Inner learning rules available at L0, all shown by the source to be gradient steps on an
associative objective:

```text
Hebbian     M_t = alpha M_{t-1} + eta v_t k_t^T                            (eq. 64)
Delta       M_t = (I - eta k_t k_t^T) M_{t-1} + eta v_t k_t^T              (eq. 65)
```

## The level table

| Level | State | Objective | Updates | Code |
|---|---|---|---|---|
| L0 data memory | `W` | maps inputs to their Local Surprise Signal (eq. 9) | every sample | `levels.DataMemory` |
| L1 gradient memory | `m` | maps inputs to their LSS-value (eq. 13) | every `period` samples | `levels.GradientMemory` |

`period` is the operational content of "nested timescales", and
`test_gradient_memory_period_controls_its_update_frequency` asserts a slower level really
does hold its value between updates. The experiment's artefact records each level's update
count, so the claim is audited rather than assumed.

## Source-equation to code mapping

| Source | Code |
|---|---|
| eqs. 1, 8 (GD, surprise factorization) | `levels.weight_gradient`, `levels.local_surprise_signal` |
| eq. 9 (GD as associative memory) | `levels.DataMemory.update` |
| eqs. 10-11 (momentum) | `levels.GradientMemory.update` + `learner.NestedLearner.step` |
| eq. 13 (momentum as its own objective) | the same, documented in the audit |
| eq. 64 (Hebbian) | `DataMemory.associative_update`, `rule="hebbian"` |
| eq. 65 (Delta) | `DataMemory.associative_update`, `rule="delta"` |
| eqs. 58-60 (self-referential / GGD) | `learner.SelfReferentialLearner` |

## What is deliberately not implemented

| Source component | Status | Reason |
|---|---|---|
| Continuum Memory System (§7) | **not implemented** | Section not read in this environment; the prompt permits this stage only if it maps unambiguously, and it does not. |
| Hope architecture (§8) | **not implemented** | Same. **No model in this track is called Hope**, and no result here bears on it. |
| Multi-scale Momentum Muon (M3) | **not implemented** | Depends on the above. |

Implementing a guess and naming it Hope would be exactly the reproduction inflation the
repository forbids.

## Numerical stability

The learners are first-order methods on a convex least-squares problem, so the only
instability is a step size above the stability threshold `2 / lambda_max`. That threshold
is *different for one and two levels*: the two-level learner's effective step is
`eta / (1 - decay)`, ten times larger at `decay = 0.9`. A shared learning rate would
therefore compare step sizes rather than level structures — and did, in the first run of
this suite, where the two-level learner diverged to a loss of `1.4e8`. Every learner now
selects its rate from the same grid on validation.

## Expected failure modes

1. **Momentum trades retention for plasticity.** Larger effective steps adapt faster to
   the current task and overwrite the previous one more thoroughly.
2. **Momentum's horizon is short.** Its state decays with time constant `1 / (1 - decay)`,
   about ten samples at `decay = 0.9`, so it cannot carry knowledge across a task
   boundary hundreds of samples long. The reset ablation is expected to show *nothing* for
   that reason, which is itself informative.
3. **The self-referential rule is not a new algorithm** at the `L2` instance — it provably
   equals gradient descent, which is the source's point rather than a defect.

## Ablations implemented

| Variant | What it changes |
|---|---|
| `one-level` | No L1 at all: plain gradient descent. |
| `two-level` | Adds L1, i.e. momentum. |
| `two-level-slow` | L1 updates once every four samples. |
| `two-level-reset` | L1 discarded at every task boundary. |
| `self-referential` | Targets generated by the memory's own state (eq. 58). |

## Reproducing

```bash
poetry run modern-nn run-track hope
poetry run modern-nn summarize --track hope
```

The whole suite runs in about twelve seconds. Diagnostics land in
`results/hope/artefacts/continual_diagnostics.json`; the report is
[`reports/hope.md`](../../../../reports/hope.md).
