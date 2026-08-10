# Track 03 — Mamba-3 / Modern State-Space Models

**Claim level: `educational implementation`** (see [`docs/claim_policy.md`](../../../../docs/claim_policy.md)).

Unlike Tracks 01 and 02, the primary source for this track was retrieved and read inside
this environment, so the equations below are transcribed from it rather than reconstructed.
See [`docs/source_registry.md`](../../../../docs/source_registry.md#track-03--mamba-3) for
exactly what was and was not verified.

## Problem statement

A state-space model describes continuous linear dynamics

```text
h'(t) = A(t) h(t) + B(t) x(t),        y(t) = C(t)ᵀ h(t)
```

which must be *discretized* before it can consume a token sequence. Two choices then
determine what the resulting recurrence can and cannot compute:

1. **How the state-input integral is approximated.** Mamba-1/2 evaluate it at the right
   endpoint only (Euler). That is a first-order approximation with local truncation error
   `O(Δ³)`.
2. **What the transition is allowed to be.** Successive Mamba generations simplified it
   from complex normal-plus-low-rank, to a diagonal of reals, to a single scaled identity.
   A real non-negative transition cannot represent rotation — and therefore cannot
   represent parity.

Mamba-3 addresses both, plus a hardware-utilization concern, with three changes. This
track implements each behind a flag so that switching it off recovers the prior method
**exactly**, which is what makes the ablations interpretable.

## Mathematical formulation

### Baseline: Mamba-2's exponential-Euler recurrence (source eq. 1)

```text
h_t = α_t h_{t-1} + γ_t B_t x_t,      y_t = C_tᵀ h_t
α_t = exp(Δ_t A_t) ∈ (0, 1),          γ_t = Δ_t
```

### 1. Exponential-trapezoidal discretization (Proposition 1, eqs. 5-6)

Approximating the state-input integral with a *data-dependent convex combination of both
interval endpoints* gives a second-order rule:

```text
h_t = exp(Δ_t A_t) h_{t-1} + (1-λ_t) Δ_t exp(Δ_t A_t) B_{t-1} x_{t-1} + λ_t Δ_t B_t x_t
    =  α_t h_{t-1}          +  β_t B_{t-1} x_{t-1}                     +  γ_t B_t x_t
```

with `λ_t ∈ [0,1]` data dependent, `α_t = exp(Δ_t A_t)`, `β_t = (1-λ_t) Δ_t exp(Δ_t A_t)`,
`γ_t = λ_t Δ_t`.

- `λ_t = 1` → `β_t = 0` → exponential-Euler, i.e. Mamba-1/2 (source Remark 2).
- `λ_t = ½` → the classical trapezoid.

The `β_t` term is a **width-two convolution on the state input, inside the recurrence** —
distinct from the short convolutions applied to `x`, `B`, `C` *outside* the recurrence in
earlier models (source Remark 4).

### 2. Complex dynamics as data-dependent rotations (Propositions 2-3, eqs. 8-10)

A complex diagonal SSM with state in `C^{N/2}` is equivalent to a real SSM with state in
`R^N` whose transition is a scalar decay times a block diagonal of 2×2 rotations
(Proposition 2, eq. 9):

```text
h_t = exp(Δ_t A_t) R_t h_{t-1} + Δ_t B_t x_t,     R_t = Block({R(Δ_t θ_t[i])})
R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]
```

and that in turn equals a **scalar** recurrence in which the accumulated rotation is
applied to `B` and `C` instead of to the state — the "RoPE trick" (Proposition 3, eq. 10):

```text
h_t = exp(Δ_t A_t) h_{t-1} + (Π_{i≤t} R_iᵀ) Δ_t B_t x_t
y_t = [(Π_{i≤t} R_iᵀ) C_t]ᵀ h_t
```

**This implementation runs the RoPE form**, and
`test_rope_form_equals_the_block_rotation_form` asserts it against the block-rotation
form. Because all rotations act in the same plane they commute, so the accumulated
rotation is just `R(-Φ_t)` with `Φ_t = Σ_{i≤t} Δ_i θ_i`, and the product can be carried as
a running angle.

Why it matters: parity on `{0,1}` is solved exactly by `h_t = R(π x_t) h_{t-1}`, and real
eigenvalues cannot express that. `parity_reference` in `model.py` is that construction.

### 3. MIMO (eqs. 12-14)

Raise the rank of the state update from 1 to `R`:

```text
h_t^{(j)} = α_t h_{t-1}^{(j)} + Δ_t B_t^{(j)} x_t^{(j)ᵀ}      (12)
h_t       = Σ_j h_t^{(j)}                                      (13)
y_t^{(i)} = (C_t^{(i)})ᵀ h_t                                   (14)
```

The carried state stays `N×P`, so decode memory traffic is unchanged while FLOPs rise by
`R` — the point being arithmetic intensity, not capacity. Following the source's
parameter-efficient instantiation, the `R` copies of the head input are produced by a
learnable **data-independent** per-rank scale (`DP + PR` parameters per head rather than
`DPR`).

## Tensor-shape table

| Symbol | Code | Shape | Meaning |
|---|---|---|---|
| `T` | seq_len | — | sequence length |
| `D` | `d_model` | — | model width |
| `H` | `heads` | — | SSM heads |
| `P` | `head_dim` | `D / H` | head width |
| `N` | `state_size` | — | state size per head (even when rotary) |
| `R` | `rank` | — | MIMO rank |
| `Δ_t` | `coefficients()["delta"]` | `(B, H)` | step size, `softplus` |
| `A_t` | `coefficients()["a"]` | `(B, H)` | transition, `-softplus` so `α ∈ (0,1)` |
| `α, β, γ` | `coefficients()[...]` | `(B, H)` | discretization coefficients |
| `λ_t` | `coefficients()["lam"]` | `(B, H)` | trapezoid mixing, `sigmoid` |
| `θ_t` | `theta_projection` | `(B, H, N/2)` | rotation rate per state pair |
| `Φ_t` | `SSMState.angle` | `(B, H, N/2)` | accumulated rotation |
| `B_t, C_t` | `b_projection`, `c_projection` | `(B, H, N, R)` | state input / read-out |
| `h_t` | `SSMState.state` | `(B, H, N, P)` | SSM state |
| — | `SSMState.pending` | `(B, H, N, P)` | previous rotated `B x` outer product |

## Source-equation to code mapping

| Source | Code |
|---|---|
| eq. 1 (Mamba-2 recurrence) | `SelectiveSSM` with `trapezoidal=False, rotary=False` |
| Table 1, exponential-Euler row | `coefficients()` with `lambda_projection is None` |
| Prop. 1 / eqs. 5-6 (`α, β, γ`) | `SelectiveSSM.coefficients` |
| eq. 6's `β B_{t-1} x_{t-1}` term | `SSMState.pending` in `SelectiveSSM.step` |
| Prop. 2 / eq. 9 (block rotation) | `block_rotation_reference` in the tests |
| Prop. 3 / eq. 10 (RoPE trick) | `rotate_pairs` + `SSMState.angle` in `step` |
| eqs. 12-14 (MIMO) | the two `einsum` contractions in `step` |
| parity via `h_t = R(π x_t) h_{t-1}` | `model.parity_reference` |

## Complexity

Per sequence, per head, with state `N`, head width `P`, rank `R`:

| Quantity | This SSM | Attention |
|---|---|---|
| Time | `O(T · N · P · R)` | `O(T² · D)` |
| Carried state | `O(N · P)`, independent of `T` | `O(T · D)` KV cache |
| Parallel over `T` | yes in principle (SSD/chunked form) — **not here** | yes |

The source's arithmetic-intensity argument (its Table 2) is that MIMO raises intensity
from `Θ(1)` to `Θ(R)` for `R ≪ N, P`, which matters only on hardware whose decode is
memory-bound. **None of that is observable in this implementation**, which steps the
recurrence in Python.

## Numerical stability

| Issue | Handling |
|---|---|
| `α_t` leaving `(0,1)` and the state diverging | `A_t = -softplus(·)` so `Δ_t A_t < 0` |
| Step size collapsing to zero or exploding | `Δ_t = softplus(·)`, bias initialized log-uniformly in `[10⁻³, 10⁻¹]` |
| Accumulated rotation drifting | carried as an angle and applied with `cos`/`sin`, so it stays exactly orthogonal; `test_rotations_preserve_pair_norms` checks it |
| Odd state size under rotation | rejected at construction |

## Expected failure modes

1. **The rotary ablation cannot do parity.** Predicted by the source; tested directly.
2. **A Python scan is slow.** Roughly an order of magnitude slower than a fused LSTM
   kernel at these sizes, which says nothing about the architecture.
3. **MIMO buys nothing here.** Its benefit is hardware arithmetic intensity during
   memory-bound decode; on CPU at `d_model = 32` it should be invisible or negative.
4. **Second-order discretization may not help at short sequence length.** The trapezoidal
   rule reduces truncation error in `Δ`; if `Δ` is already small the gain is negligible.

## Known approximations and deviations from the primary source

| # | Deviation | Reason |
|---|---|---|
| 1 | Sequential Python scan; no SSD/chunked parallel form, no Triton kernel. | Transparency first. The consequence — no throughput claim is meaningful — is stated wherever cost appears. |
| 2 | The block is a plain pre-norm residual wrapper. No gated MLP branch, no short causal convolution on `x`/`B`/`C`, no normalization inside the block. | Those are not the mechanisms under study and would confound the three ablations. |
| 3 | MIMO read-out sums the `R` outputs. The source specifies `y_t^{(i)}` per output index and leaves their combination to the surrounding block. | The surrounding block is not implemented (deviation 2), so a sum is the neutral choice. |
| 4 | No multi-value-attention head structure, so `B` and `C` are not shared across heads. | Simplicity; this changes parameter counts, not the recurrence. |
| 5 | `Δ_t` is a scalar per head, not per channel. | Matches the source's scaled-identity transition; keeps the shape table small. |
| 6 | No language-modelling experiment at any scale. | Out of reach on CPU. Nothing here bears on the source's reported 1.5B-scale results. |

## Ablations implemented

| Variant | Flag | What it isolates |
|---|---|---|
| `euler-discretization` | `trapezoidal=False` | Contribution 1. Recovers Mamba-1/2's rule exactly (verified in tests). |
| `no-rotation` | `rotary=False` | Contribution 2. Leaves a real non-negative transition, which the source predicts cannot do parity. |
| `siso-rank-1` | `rank=1` | Contribution 3. Recovers the SISO recurrence exactly (verified in tests). |

## Reproducing

```bash
poetry run modern-nn run-track mamba3
poetry run modern-nn summarize --track mamba3
```

The report is [`reports/mamba3.md`](../../../../reports/mamba3.md).
