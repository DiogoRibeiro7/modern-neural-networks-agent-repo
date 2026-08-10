# Track 02 — xLSTM

**Claim level: `educational implementation`** (see [`docs/claim_policy.md`](../../../../docs/claim_policy.md)).

## Problem statement

A conventional LSTM writes into its cell state through a sigmoid input gate:

```text
c_t = f_t * c_{t-1} + i_t * z_t,      i_t = sigmoid(...) ∈ (0, 1)
```

Because `i_t < 1`, a token arriving late can never outweigh what the state already
accumulated. The state can be *decayed* by the forget gate, but it cannot be *overwritten*
in one step. This is the concrete failure behind an LSTM's difficulty with selective
recall: a decision made early is hard to revise.

xLSTM's answer is to replace the sigmoid with an exponential, `i_t = exp(...)`, which is
unbounded. That immediately creates two problems this track has to handle explicitly:

1. the cell state is no longer on a comparable scale between steps, so a **normalizer
   state** must track total gate mass;
2. the exponential overflows within a few dozen steps, so a **running maximum** must
   rescale numerator and denominator together.

The question this track answers: on tasks that require revising or selectively retrieving
stored information, does exponential gating beat conventional gating **at a matched
parameter budget**, and what does it cost?

## Mathematical formulation

### sLSTM — scalar memory, with memory mixing

```text
z_t = tanh(W_z x_t + R_z h_{t-1} + b_z)          candidate
i_t = exp (W_i x_t + R_i h_{t-1} + b_i)          exponential input gate
f_t = sig (W_f x_t + R_f h_{t-1} + b_f)          forget gate
o_t = sig (W_o x_t + R_o h_{t-1} + b_o)          output gate

c_t = f_t c_{t-1} + i_t z_t                       cell state
n_t = f_t n_{t-1} + i_t                           normalizer state
h_t = o_t * (c_t / n_t)                           hidden state
```

The gates read `h_{t-1}` ("memory mixing"), so the recurrence has a hidden-to-hidden path
and **cannot** be unrolled in parallel across time.

### mLSTM — matrix memory, no memory mixing

```text
q_t, k_t, v_t = W_q x_t, W_k x_t / sqrt(d), W_v x_t

C_t = f_t C_{t-1} + i_t v_t k_t^T                 matrix memory
n_t = f_t n_{t-1} + i_t k_t                       normalizer vector
h_t = o_t ⊙ ( C_t q_t / max(|n_t^T q_t|, 1) )
```

The gates depend only on `x_t`, so the recurrence has no hidden-to-hidden path and is in
principle parallelizable. The update `v_t k_t^T` is an outer product: mLSTM stores
key-value *associations*, which is why it is the branch that helps on associative recall.

### Stabilization

Both cells carry `m_t = max(log f_t + m_{t-1}, log i_t)` and use

```text
i'_t = exp(log i_t - m_t)                  ≤ 1
f'_t = exp(log f_t + m_{t-1} - m_t)        ≤ 1
```

Numerator and denominator are rescaled by the same factor, so `h_t` is mathematically
unchanged while every exponential stays representable. `test_stabilizer_keeps_long_sequences_finite`
is the guard: with large inputs, an unstabilized exponential gate overflows well within
400 steps.

## Tensor-shape table

| Symbol | Code | Shape | Meaning |
|---|---|---|---|
| `B` | batch | — | batch size |
| `T` | seq_len | — | sequence length |
| `D` | `d_model` | — | model width |
| `H` | `hidden_size` | — | sLSTM hidden width |
| `h` | `heads` | — | mLSTM memory heads |
| `d` | `head_dim` | `D / h` | width of one head |
| `x_t` | `inputs` | `(B, D)` | one step's input |
| `c_t` | `SLSTMState.cell` | `(B, H)` | scalar cell state |
| `n_t` | `SLSTMState.normalizer` | `(B, H)` | gate-mass state |
| `m_t` | `*.stabilizer` | `(B, H)` / `(B, h)` | running log-maximum |
| `C_t` | `MLSTMState.memory` | `(B, h, d, d)` | matrix memory |
| `n_t` | `MLSTMState.normalizer` | `(B, h, d)` | normalizer vector |
| tokens | model input | `(B, T)` | integer ids |
| logits | model output | `(B, T, V)` | per-position scores |

## Source-equation to code mapping

| Source concept | Code |
|---|---|
| exponential vs sigmoid gate, in log space | [`cells._log_gate`](cells.py) |
| sLSTM recurrence | [`cells.SLSTMCell.forward`](cells.py) |
| mLSTM recurrence | [`cells.MLSTMCell.forward`](cells.py) |
| stabilizer state `m_t` | the `torch.maximum(...)` line in each `forward` |
| normalizer state `n_t` | `normalizer` in each `forward` |
| covering rule `max(|n^T q|, 1)` | `denominator ... .clamp_min(1.0)` |
| residual block stack | [`model.XLSTM.forward`](model.py) |

## Complexity

Per sequence of length `T`, width `D`, heads `h`, head width `d = D/h`:

| Model | Time | State memory | Parallel over `T`? |
|---|---|---|---|
| sLSTM | `O(T · D²)` | `O(D)` | no — memory mixing |
| mLSTM | `O(T · h · d²) = O(T · D · d)` | `O(h · d²) = O(D · d)` | in principle yes; **not** in this implementation |
| LSTM / GRU | `O(T · D²)` | `O(D)` | no |
| Transformer | `O(T² · D)` | `O(T · D)` (KV cache) | yes |

The crossover matters: the recurrent models are linear in `T` where attention is
quadratic, but at the sequence lengths used here (8-17) the quadratic term is negligible
and the Python-level step loop dominates. **No throughput claim in this track's report is
a claim about the mechanism's achievable speed** — a stepwise Python recurrence is not a
fused kernel.

## Numerical stability

| Issue | Handling |
|---|---|
| `exp` of a large gate pre-activation overflows | running-maximum stabilizer, tested to 400 steps |
| `log(sigmoid(a))` underflows for very negative `a` | computed as `-softplus(-a)` |
| `c_t / n_t` with negligible gate mass early in a sequence | `n_t` starts at 1 for sLSTM and is floored at 1e-6 |
| `C_t q_t / (n_t^T q_t)` amplifying noise | the source's `max(|·|, 1)` covering rule |
| Forgetting before learning to remember | forget-gate bias initialized to +1 |

## Expected failure modes

1. **sLSTM is slow.** Memory mixing forces a sequential scan; at these sizes Python loop
   overhead, not FLOPs, sets the wall-clock cost.
2. **An exponential forget gate diverges.** Sigmoid is the default forget gate for that
   reason; the exponential variant is available but is not the reported configuration.
3. **Small matrix memory saturates.** With `d` small, `C_t` cannot store many distinct
   associations, so selective recall degrades as the number of pairs grows.
4. **Transformer wins at short context.** With `T ≤ 64` attention has no scaling
   disadvantage and full access to every position; a recurrent model has no structural
   advantage to exploit.

## Known approximations and deviations from the primary source

| # | Deviation | Reason |
|---|---|---|
| 1 | No block-parallel or chunkwise-recurrent kernel; mLSTM is stepped sequentially in Python. | Transparency first, per the master prompt. The consequence — that measured throughput says nothing about the architecture's achievable speed — is stated wherever throughput appears. |
| 2 | Blocks are plain pre-norm residual wrappers, not the source's up/down-projection block with causal convolution and learnable skip. | Those components are not the mechanism under study, and including them would confound gating with block design. |
| 3 | No multi-head sLSTM with block-diagonal recurrent matrices. | Simplicity; the sLSTM here uses a single dense recurrent projection. |
| 4 | Forget gate defaults to sigmoid. | The source permits either. The exponential forget gate makes the state grow without bound on long sequences, which is a distraction from the input-gate question. |
| 5 | No language-model-scale experiment. | Out of reach on CPU. Nothing here supports any claim about xLSTM at scale. |

## Ablations implemented

| Variant | Flag | What it isolates |
|---|---|---|
| `sigmoid-input-gate` | `input_gate="sigmoid"` | **The mechanism itself.** Everything else — normalizer, stabilizer, matrix memory, block structure — is unchanged; only the gate's ceiling differs. |
| `slstm-blocks` | `block_kinds=("slstm", ...)` | Scalar memory with memory mixing versus matrix memory without it. |

## Reproducing

```bash
poetry run modern-nn run-track xlstm
poetry run modern-nn summarize --track xlstm
```

The report is [`reports/xlstm.md`](../../../../reports/xlstm.md).
