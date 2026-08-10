# Formalization Audit — Nested Learning (Track 06)

This document is the **first milestone** of Track 06, written before any code. Its purpose
is to state exactly which optimization levels exist in the compact experiment, what each
one's state is, what objective it minimizes, and how often it updates. Nothing is
implemented that is not derived here, and anything speculative is labelled as such.

Source: Behrouz, Razaviyayn, Zhong, Mirrokni, *Nested Learning: The Illusion of Deep
Learning Architectures*, arXiv:2512.24695. Equation numbers refer to that paper.

## 1. The central claim being audited

> A neural learning module is a set of nested and/or parallel optimization problems, each
> with its own context flow, objective, and **update frequency**.

The claim that makes this testable rather than philosophical is narrower and specific:

> Gradient descent **with momentum** is a *two-level* associative memory. The inner level
> learns to store gradients in its parameters; the outer level updates the slow weights
> with the value of the inner memory. (Paper, §4.2 discussion following eq. 13.)

That is a falsifiable structural claim about an algorithm everybody already uses, and it
is what this track builds on.

## 2. Derivation of the levels

### 2.1 Level structure of plain gradient descent

Starting from SGD (eq. 1):

```text
W_{t+1} = W_t - eta_t grad_W L(W_t; x_t)                                   (1)
```

The paper decomposes the gradient for a linear layer with output `y = W x` (eq. 8):

```text
W_{t+1} = W_t - eta_{t+1} grad_{y_{t+1}} L(W_t; x_{t+1}) (x) x_{t+1}       (8)
                         \_______ local surprise signal ______/
```

so the update is an **outer product of an input with its own error**. Reading that as an
associative memory (eq. 9):

```text
W_{t+1} = argmin_W  <W x_{t+1}, u_{t+1}> + (1/2 eta) ||W - W_t||^2         (9)
          where u_{t+1} = grad_{y} L(W_t; x_{t+1})
```

**Conclusion.** Plain GD is a **one-level** associative memory that maps each data point
to its own Local Surprise Signal (LSS). There is exactly one state, `W`, updated once per
sample.

### 2.2 Level structure of gradient descent with momentum

Replacing GD with its momentum variant (eqs. 10-11):

```text
W_{t+1} = W_t - m_{t+1}                                                    (10)
m_{t+1} = m_t + eta_{t+1} grad_W L(W_t; x_{t+1})                           (11)
```

and rewriting the momentum step itself as a proximal problem (eq. 13):

```text
m_{t+1} = argmin_m  -<m x_{t+1}, grad_y L(W_t; x_{t+1})> + (1/2 eta)||m - m_t||^2   (13)
```

**Conclusion.** Momentum is not bookkeeping. It is an associative memory in its own right,
with its own objective, whose *contents* are then consumed by a second update. That gives
two levels.

### 2.3 The level table for this track

This is the operational core of the audit. Every level implemented in the track appears
here with its state, objective, and update frequency.

| Level | State | Objective it minimizes | Updated | Context it compresses |
|---|---|---|---|---|
| **L0 — data memory** | `W` (fast weights) | maps each input to its Local Surprise Signal, eq. 9 | every sample | the data stream |
| **L1 — gradient memory** | `m` (momentum) | maps each input to its LSS-value, eq. 13 | every sample | the gradient stream |
| **L2 — schedule** | `eta`, `alpha` | none (fixed or scheduled) | every `K` samples, or never | — |

Two properties follow directly and are what the tests assert:

1. **L0 with no L1 is exactly SGD.** Setting `m = 0` and consuming the gradient directly
   recovers eq. 1. So "add a level" and "use momentum" must be the *same operation*.
2. **Frequency is what separates levels.** L1 updates as often as L0 in classical
   momentum; a genuinely *slower* level is one whose update frequency is lower. The
   paper's continual-learning argument (§4.2, "A Note on Optimizers in Continual Learning
   Setup") rests on this: knowledge about the loss landscape lives in the momentum state,
   and discarding that state discards knowledge.

### 2.4 Inner learning rules available at L0

The paper shows several familiar recurrences *are* gradient steps on an associative
objective, which lets the same level be instantiated with different learning rules:

| Rule | Update | Source |
|---|---|---|
| Hebbian / linear attention | `M_t = alpha_t M_{t-1} + eta_t v_t k_t^T` | eq. 64 |
| Delta rule | `M_t = (I - eta_t k_t k_t^T) M_{t-1} + eta_t v_t k_t^T` | eq. 65 |

The delta rule's extra term `-eta k k^T M` is a *removal* of the previously stored value
before writing the new one, which is why it manages memory better than the purely additive
Hebbian rule.

### 2.5 The self-referential (self-modifying) level

The paper reformulates backpropagation itself as self-referential (eq. 58):

```text
W_{t+1} = W_t + eta_{t+1} v_t (x) x_t,      v_t = f_{W_t}(x_t) = -grad_y L(W_t; x_t)   (58)
```

and generalizes it (Definition 5, eqs. 59-60):

```text
W_{t+1} = argmin_W  Lhat(x_t, u_t) + Ret(W, {W_i})                        (59)
u_t = f_{W_t}(x_t)                                                         (60)
```

**The defining property is that the target `u_t` is generated by the memory's own current
state.** Ordinary supervised learning takes `u_t` from outside; here it is a function of
`W_t`. That is precisely testable: for the `L2` instance, `u_t` must equal
`-grad_y L(W_t; x_t)`, and the general rule must reduce to eq. 58 in that case.

## 3. What this track will and will not implement

Following the track prompt's staged strategy:

| Stage | Status | Reason |
|---|---|---|
| 1. Minimal two-timescale nested learner | **implemented** | Derived above from eqs. 1, 8-13; every level has an explicit state transition. |
| 2. Verify each inner/outer update independently | **implemented** | One test per level; see §4 below. |
| 3. Self-modifying update module | **implemented** | Equations 58-60 were read and transcribed, so the mapping is unambiguous. Added only after stage 1 was stable, as the prompt requires. |
| 4. Continuum Memory System / full Hope | **NOT implemented** | Sections 7 and 8 of the source were **not read** in this environment. The prompt permits this stage *only if it can be mapped unambiguously to the primary source*, and it cannot be. Implementing a guess and calling it Hope would be exactly the reproduction inflation the repository forbids. |

**No model in this track is called Hope**, and no result here should be read as evidence
about the Hope architecture.

## 4. Verification plan

Each level must have an explicit state transition and a test that pins it down
independently of the others. This is the track's acceptance criterion.

| Claim | Test |
|---|---|
| L0 alone is exactly SGD | a one-level learner reproduces a hand-computed SGD step |
| L0+L1 is exactly SGD with momentum | a two-level learner reproduces `torch.optim.SGD(momentum=...)` |
| Levels are separable | disabling L1 turns the two-level learner into the one-level learner, bitwise |
| Frequency is a real knob | a level with period `k` changes state on exactly every `k`-th sample |
| Delta rule matches eq. 65 | hand-computed `(I - eta k k^T) M + eta v k^T` |
| Hebbian rule matches eq. 64 | hand-computed `alpha M + eta v k^T` |
| Self-generated values match eq. 58 | `u_t == -grad_y L(W_t; x_t)` under the `L2` instance |
| Gradient memory holds knowledge | resetting `m` between tasks measurably changes continual-learning behaviour |

## 5. Speculative content, explicitly labelled

The following are **interpretations**, not results, and are labelled as such wherever they
appear:

1. That momentum "stores knowledge about the loss landscape" is the paper's framing. This
   track tests only its *observable consequence* — that discarding the momentum state
   changes continual-learning behaviour — not the framing itself.
2. Any claim that more levels are generally better is unsupported here. The experiment
   measures two levels against one on one synthetic stream.
3. The brain analogy in the source's §1.1 and §5.1 is not evaluated in any way.
