# Benchmark Protocol

## Shared benchmark philosophy

Each architecture must be tested on a task designed to expose its claimed mechanism before being tested on a broad benchmark.

Examples:

- KAN: controlled function approximation before tabular regression.
- xLSTM / Mamba-3 / TTT: copy, selective recall, or state tracking before real sequence data.
- Titans: long-range retrieval and memory-over-time diagnostics before language-like tasks.
- Sparse MoE: routing/load balance and expert specialization before accuracy benchmarks.
- Flow Matching: known 2D distributions before image generation.
- JEPA: controllable latent factors before high-dimensional representation learning.
- PFN: synthetic task distributions before real tabular datasets.
- Relational models: synthetic foreign-key tasks with explicit leakage checks before public multi-table datasets.

## Baseline policy

Use the simplest strong baseline appropriate to the task. Typical baseline families:

- MLP;
- LSTM/GRU;
- Transformer encoder;
- gradient-boosted trees for tabular data;
- dense feed-forward network for KAN;
- dense MoE-equivalent FFN for sparse MoE;
- diffusion/score baseline only when a fair implementation is available for flow matching.

## Data leakage

Temporal and relational tasks require split logic that prevents future or target-derived information from crossing the split boundary. Tests should explicitly attempt to detect leakage.
