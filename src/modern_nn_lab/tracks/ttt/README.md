# Track 04 — Test-Time Training Layers

**Claim level: `educational implementation`** (see [`docs/claim_policy.md`](../../../../docs/claim_policy.md)).

The primary source was retrieved and read inside this environment; equations below are
transcribed from it. See
[`docs/source_registry.md`](../../../../docs/source_registry.md#track-04--test-time-training).

## Problem statement

Every RNN compresses its past into a fixed-size state. What that state *is* determines what
it can hold: a vector for an LSTM, a matrix for mLSTM, a decaying linear operator for an
SSM. TTT asks a different question — **what if the state were a model, and the recurrence
were learning?**

The state is the weight matrix `W` of an inner model `f`. Reading a token means taking one
gradient step on a self-supervised reconstruction loss built from that token. Emitting an
output means running `f` with the freshly updated weights.

## Mathematical formulation

```text
l(W; x_t) = || f(θ_K x_t; W) − θ_V x_t ||²          (source eq. 4)
W_t       = W_{t−1} − η_t ∇l(W_{t−1}; x_t)          (source eq. 6, online GD)
z_t       = f(θ_Q x_t; W_t)                          (source eq. 5)
```

- `θ_K` — **training view**: what the inner model reads.
- `θ_V` — **label view**: what it must reconstruct. Not `x_t` itself, because not all of
  `x_t` is worth remembering.
- `θ_Q` — **test view**: what is asked of the updated model to produce the output.

Instantiations (source §2.7):

```text
f(x) = x + LN(f_res(x))
f_res = f_lin(x) = Wᵀx                    → TTT-Linear   (η_base = 1.0)
f_res = MLP with 4× hidden width, GELU    → TTT-MLP      (η_base = 0.1)
η(x)  = η_base · sigmoid(θ_lr · x)        learnable, token dependent
W_0   = learnable ("θ_init")
```

## The distinction this track exists to make

Two optimization problems run at once, and conflating them is the standard misreading:

| | inner loop | outer loop |
|---|---|---|
| optimizes | `W`, the inner model's weights | `θ_K, θ_V, θ_Q, W_0, θ_lr`, and everything else |
| when | during the forward pass, at every token | during training only |
| objective | self-supervised reconstruction (eq. 4) | the task loss |
| persists across sequences? | **no** — reset to `W_0` every sequence | yes |
| is it an `nn.Parameter`? | **no** — it is a hidden state | yes |

So a forward pass performs gradient descent **without changing a single parameter**.
`test_forward_pass_never_mutates_outer_parameters` asserts precisely that, and
`test_hidden_state_does_change_during_a_forward_pass` asserts its mirror image. Together
they are the operational definition of "test-time training of the hidden state".

A practical consequence that is easy to get wrong: the inner loop must run **inside
`torch.no_grad()`**, because evaluation and profiling wrap the forward pass in it. A TTT
layer that silently stopped learning at evaluation time would not be doing test-time
training at all. `inner_gradient` therefore re-enables grad locally, and
`test_inner_loop_runs_under_no_grad` guards it.

## Tensor-shape table

| Symbol | Code | Shape | Meaning |
|---|---|---|---|
| `D` | `d_model` | — | model width |
| `d` | `d_inner` | `D // 2` | view width (a low-rank projection) |
| `θ_K x_t` | `train_view(x)` | `(B, d)` | training view |
| `θ_V x_t` | `label_view(x)` | `(B, d)` | label view |
| `θ_Q x_t` | `test_view(x)` | `(B, d)` | test view |
| `W` (linear) | `LearnerState.weights[0]` | `(B, d, d)` | **per-example** hidden state |
| `W` (MLP) | `LearnerState.weights` | `(B, d, 4d)`, `(B, 4d, d)` | two inner matrices |
| `η_t` | `inner_learning_rate(x)` | `(B, 1)` | inner learning rate |
| `z_t` | layer output | `(B, D)` | after the output projection |

Note the leading `B`: **every example in the batch carries its own learner.**
`test_each_example_carries_its_own_learner` checks that batching does not couple them.

## Source-equation to code mapping

| Source | Code |
|---|---|
| eq. 4 (reconstruction loss) | `TTTLayer.inner_loss` |
| eq. 6 (gradient step) | `TTTLayer.step` |
| eq. 5 (output rule) | the `apply_inner_model(test_view(...), updated)` call in `step` |
| §2.7 `f(x) = x + LN(f_res(x))` | `TTTLayer.apply_inner_model` |
| §2.7 learnable `η(x)` | `TTTLayer.inner_learning_rate` |
| §2.7 learnable `W_0` | `TTTLayer.initial_weights` |
| Theorem 1 (batch GD ≡ linear attention) | `update_rule="batch"`, asserted in the tests |
| "gradients of gradients" (§2.2) | `create_graph=True` in `inner_gradient` |

## Complexity

Per token, with view width `d`:

| | inner state | forward FLOPs | notes |
|---|---|---|---|
| TTT-Linear | `d²` | `O(d²)` | one gradient step is two matmuls |
| TTT-MLP | `8d²` | `O(d²)` | 4× hidden width, two matrices |
| Frozen ablation | `d²` | `O(d²)` | no gradient, so materially cheaper in practice |

Cost is linear in sequence length, like any RNN. The constant is large here because the
scan is a Python loop *and* every step builds a small autograd graph. The source addresses
exactly this with mini-batch TTT (`b = 16`) and the dual form; neither is implemented here.

## Numerical stability

| Issue | Handling |
|---|---|
| The inner loss growing without bound | `f` contains a LayerNorm (source §2.7), which the residual form keeps well scaled |
| Inner learning rate too large | `η = η_base · sigmoid(·)` is bounded by `η_base`, which follows the source: 1.0 for Linear, 0.1 for MLP |
| Second-order gradients exploding | gradient-norm clipping in the shared training loop |
| Inner loop silently disabled under `no_grad` | `torch.enable_grad()` inside `inner_gradient`, guarded by a test |

## Expected failure modes

1. **Slow.** Every token costs a forward *and* a backward through the inner model, plus
   graph construction. Expect roughly an order of magnitude over a fused LSTM.
2. **The frozen ablation is not a weaker model, it is a different one.** With `W_t = W_0`
   for all `t` the layer stops being sequential at all: it becomes a position-independent
   function of each token. `test_frozen_learner_is_position_invariant` states this exactly.
   It is the right ablation *because* it is drastic — it removes the recurrence itself.
3. **Batch GD collapses to linear attention** (Theorem 1), so that ablation is not "no
   learning" but "a different, weaker learner".
4. **Small `d` limits what the learner can store**, independently of how well it learns.

## Known approximations and deviations from the primary source

| # | Deviation | Reason |
|---|---|---|
| 1 | Online gradient descent (`b = 1`) only. No mini-batch TTT and no dual form. | Both are systems optimizations. The source reports `b = 16` gives the single largest quality gain in its ablation table, so **this implementation is expected to underperform the paper's on quality as well as speed**, and no comparison to its numbers is made. |
| 2 | No Mamba backbone; blocks are plain pre-norm residual wrappers with no temporal convolution. | Isolates the TTT layer from the backbone, which the source's own ablation shows contributes separately. |
| 3 | Single head. | Simplicity; changes capacity, not the mechanism. |
| 4 | Inner gradients via `torch.autograd.grad` rather than hand-derived formulas. | Correctness over speed, and it makes TTT-MLP tractable. The linear case is checked against the analytic gradient in the tests. |
| 5 | No language-modelling experiment at any scale. | Out of reach on CPU. |

## Ablations implemented

| Variant | Flag | What it removes |
|---|---|---|
| `frozen-learner` | `learner_updates=False` | **The mechanism.** Same parameters, same views, same `W_0`; the inner loop simply does not run. |
| `batch-gradient-descent` | `update_rule="batch"` | Online adaptation, but not learning: every gradient is taken at `W_0`, which the source proves equals linear attention. |
| `ttt-mlp` | `inner_model="mlp"` | Nothing — it *adds* inner-model expressiveness, testing whether a richer learner helps. |

## Reproducing

```bash
poetry run modern-nn run-track ttt
poetry run modern-nn summarize --track ttt
```

The report is [`reports/ttt.md`](../../../../reports/ttt.md).
