# Program Milestones

## M0 — Repository contracts

Complete when the package installs, the registry works, the scaffold validator passes, CI runs, and the shared experiment schema is implemented.

## M1 — Transparent architecture primitives

Complete KAN, Flow Matching, and Sparse MoE. These establish the repository's conventions for mathematical tests, baselines, ablations, metrics, and reports.

## M2 — Modern sequence models

Complete xLSTM, TTT, and Mamba-3. Reuse one common sequence-task API and add scaling/profiling infrastructure.

## M3 — Learned memory and continual adaptation

Complete Titans-style memory and then Nested Learning / Hope. Add explicit memory diagnostics, forgetting curves, and update-timescale reporting.

## M4 — Foundation models for structured data

Complete the PFN-from-scratch experiment, TabPFN reference benchmark, and relational foundation-model prototype. Enforce strong non-neural baselines and leakage tests.

## M5 — Predictive representations

Complete JEPA/world-model experiments with latent-factor diagnostics and linear probing.

## M6 — Cross-track synthesis

Produce a single comparative report that does **not** collapse all tracks into one leaderboard. Compare architectures only on dimensions for which comparison is scientifically meaningful:

- memory mechanism;
- inference-time adaptation;
- computational complexity;
- parameter/activated-parameter efficiency;
- data regime;
- robustness;
- calibration;
- interpretability;
- failure modes.

## Exit condition

The repository is mature when every completed track satisfies its local acceptance criteria and the cross-track report can be regenerated entirely from saved machine-readable experiment outputs.
