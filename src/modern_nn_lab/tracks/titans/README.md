# Track 05 — Titans-Style Neural Long-Term Memory

**Claim level: `educational implementation`** (see [`docs/claim_policy.md`](../../../../docs/claim_policy.md)).

Primary source retrieved and read in-environment; equations below are transcribed from it.
See [`docs/source_registry.md`](../../../../docs/source_registry.md#track-05--titans).

## Problem statement

Attention is an exact memory over a bounded window and no memory at all beyond it. A
recurrent state is an unbounded but lossy summary. Titans proposes a third thing: a
**module that learns to memorize during inference**, whose state is the weights of a small
network, written by gradient descent on an associative objective.

The design question is not "can a network store things" — a linear attention layer already
compresses keys and values additively. It is *what to store and what to discard*. Titans
answers with three signals: surprise, momentum over surprise, and adaptive forgetting.

## Mathematical formulation

```text
k_t = x_t W_K,  v_t = x_t W_V,  q_t = x_t W_Q                       (eq. 11)

l(M_{t-1}; x_t) = || M_{t-1}(k_t) - v_t ||^2                        (eq. 12)

M_t = (1 - alpha_t) M_{t-1} + S_t                                   (eq. 13)
S_t = eta_t S_{t-1} - theta_t grad l(M_{t-1}; x_t)                  (eq. 14)

y_t = M*_t(q_t)                                                      (eq. 15)
```

- `theta_t` — how much *momentary* surprise to absorb.
- `eta_t` — how fast *past* surprise decays. Its purpose is specific: after a surprising
  token, the gradient at subsequent tokens can go flat, and a bare gradient rule (the
  source's eq. 8) would stop recording the rest of a memorable episode. Momentum carries
  the write forward.
- `alpha_t` — weight decay, i.e. adaptive forgetting. `alpha -> 1` clears the memory.

All three are **data dependent**, and all are outer-loop parameters: the inner loop
optimizes `M` only.

**Persistent memory** (eq. 19): `N_p` learnable, data-independent tokens are prepended to
the sequence. They hold task knowledge rather than context and are never written.

## Which variant, and why

The source gives three ways to wire the memory into an architecture. This track implements
**Memory as a Gate (MAG)**, source Section 4.2:

```text
x~ = [p_1 ... p_Np] || x
y  = SW-Attn*(x~)                                                    (eq. 27)
o  = y (x) M(x~)                                                     (eq. 28)
```

| Variant | Source section | Why not / why |
|---|---|---|
| Memory as Context (MAC) | 4.1 | The source's strongest performer. Rejected here because it *segments* the sequence and updates the memory from the **attention output**, entangling the two memory systems — a per-token write diagnostic could not attribute cleanly. |
| Memory as Gate (MAG) | 4.2 | **Chosen.** Both systems run over the same tokens and combine only at the end, so each branch is separately removable and each write separately measurable. |
| Memory as Layer (MAL) | 4.3 | Stacks memory before attention, so the memory's contribution is visible only after attention has reprocessed it. |

The choice is driven by this track's acceptance criterion — explicit memory-write/read
diagnostics — not by expected accuracy. **The cost is that no result here speaks to MAC**,
which is the variant the source actually recommends.

## Tensor-shape table

| Symbol | Code | Shape | Meaning |
|---|---|---|---|
| `D` | `d_model` | — | model width |
| `d` | `d_memory` | `D // 2` | key/value/query width |
| `W` | `window` | — | sliding-window size (short-term memory) |
| `N_p` | `persistent_tokens` | — | persistent prefix length |
| `M` | `MemoryState.weights` | `(B, d, h)`, `(B, h, d)` | **the memory** — a hidden state, per example |
| `S_t` | `MemoryState.momentum` | same as `M` | past surprise |
| `theta, eta, alpha` | `NeuralMemory.gates` | `(B, 1)` each | write rate, momentum, forgetting |
| trace | `MemoryTrace` | `T` floats each | per-token write/read diagnostics |

## Source-equation to code mapping

| Source | Code |
|---|---|
| eq. 11 (key/value/query) | `NeuralMemory.key/value/query` |
| eq. 12 (associative loss) | `NeuralMemory.associative_loss` |
| eq. 13 (forgetting + write) | the `weights` update in `NeuralMemory.step` |
| eq. 14 (surprise + momentum) | the `momentum` update in `NeuralMemory.step` |
| eq. 8 (bare gradient rule) | `use_momentum=False` |
| eq. 15 (retrieval) | `NeuralMemory.read` |
| eq. 19 (persistent memory) | `TitansMAG.persistent` |
| eqs. 26-28 (MAG) | `TitansMAG.forward` |

## Numerical stability

**This is the part that required a real deviation.** The update as written diverges. With
`theta_t = sigmoid(.)` unbounded above by 1, a squared loss, and momentum, the loop is
self-reinforcing: a larger memory produces a larger reconstruction error, which produces a
larger gradient, which grows the memory further. Measured on random input, the memory
reached `4.9e37` by token 4 and `NaN` by token 5.

Three bounded-scale changes fix it, all recorded as deviations:

| Issue | Handling |
|---|---|
| `theta_t` unbounded relative to the loss scale | `theta_t = theta_base * sigmoid(.)` with `theta_base = 0.1`. The source specifies a learnable `theta_t` but no numeric range. |
| Write magnitude scaling with key norm | keys and queries are L2-normalized, so a write's scale does not depend on the input's magnitude |
| Gradient scale growing with memory width | the loss is averaged over features and summed over the batch, keeping the gradient `O(1)` in `d` while leaving each example's gradient its own |
| `softmax` backward through `-inf` producing `NaN` | the sliding-window mask uses a large finite penalty |

`test_memory_stays_finite_over_long_sequences` holds the line at 200 steps with inputs
scaled by 3.

## Expected failure modes

1. **Short-term-only is fine inside its window.** That is not a bug; the interaction with
   distance is the experiment.
2. **A memory that never learns to write** still reads a non-zero `M_0`, so the frozen
   ablation is not the same as having no memory. Both are run.
3. **Slow writes underfit within a short sequence.** With only ~12 tokens, a `theta`
   scaled down by 10 may not accumulate a usable trace.
4. **Cost.** Every token is a forward and a backward through the memory MLP.

## Known approximations and deviations from the primary source

| # | Deviation | Reason |
|---|---|---|
| 1 | `theta_t` bounded by `theta_base = 0.1`; keys/queries L2-normalized; loss feature-averaged. | Without these the recurrence diverges within four tokens (measured). The source gives no numeric scale for `theta_t`. |
| 2 | MAG only. MAC and MAL are not implemented. | Justified above; MAC is the source's recommended variant, so its results are out of scope here. |
| 3 | Sequential Python scan; no chunked parallel form (source Section 3.2), no matmul-based formulation. | Transparency first. No throughput number here is evidence about Titans' achievable speed. |
| 4 | One block, one head-group, memory depth 2. | Compactness. The source studies deeper memories in its Section 5.5. |
| 5 | No language modelling and no needle-in-a-haystack at the scales the source uses. | Out of reach on CPU. |

## Ablations implemented

| Variant | Flag | What it removes |
|---|---|---|
| `short-term-only` | `use_long_term=False` | The neural memory entirely. |
| `frozen-memory` | `memory_updates=False` | Writing, but not reading — isolates learning from capacity. |
| `no-momentum` | `use_momentum=False` | Past surprise, reducing eq. 14 to the source's eq. 8. |
| `slow-updates` | `learning_rate_scale=0.1` | Write rate, ten-fold. |

## Reproducing

```bash
poetry run modern-nn run-track titans
poetry run modern-nn summarize --track titans
```

Memory diagnostics land in `results/titans/artefacts/memory_diagnostics.json`.
The report is [`reports/titans.md`](../../../../reports/titans.md).
