# Modern Neural Networks Lab

A research-oriented, reproducible implementation lab for modern neural-network architectures and learning paradigms beyond conventional MLPs and Transformers.

This repository is intentionally **scaffold-first**. It is designed for a coding/research agent to develop one architecture at a time under a shared experimental contract, so results remain comparable and claims remain auditable.

## Scope

The initial roadmap covers eleven tracks:

| ID | Track | Core question | Implementation target |
|---|---|---|---|
| 01 | Kolmogorov-Arnold Networks (KAN) | Must neural edges be scalar weights? | First-principles |
| 02 | xLSTM | Can recurrent memory scale competitively? | First-principles compact |
| 03 | Mamba-3 / SSMs | Do sequence models need attention? | First-principles compact + reference integration |
| 04 | Test-Time Training (TTT) | Can the hidden state itself learn at inference? | First-principles compact |
| 05 | Titans-style neural memory | Can explicit long-term neural memory learn online? | Faithful compact interpretation + reference integration when available |
| 06 | Nested Learning / Hope | Can learning be organized across nested timescales? | Research prototype; no reproduction claim without evidence |
| 07 | Prior-Fitted Networks / TabPFN | Must we fit a model separately for every tabular dataset? | PFN-from-scratch toy + TabPFN reference benchmark |
| 08 | Relational Foundation Models | Can models consume connected tables without flattening? | Relational prototype + reference benchmark when feasible |
| 09 | Sparse Mixture of Experts | Can conditional computation improve capacity/compute trade-offs? | First-principles compact |
| 10 | Flow Matching | Can generative learning be formulated as vector-field regression? | First-principles |
| 11 | JEPA / World Models | Should models predict useful representations rather than raw observations? | First-principles compact |

## Research principles

1. **No benchmark theatre.** Every headline result must include seeds, uncertainty, parameter counts, training budget, hardware, and wall-clock time.
2. **No reproduction inflation.** A compact educational implementation is not a reproduction of a large paper result.
3. **Baselines first.** Every track begins with a simple, strong baseline before the target architecture.
4. **Ablations are mandatory.** A new mechanism is not useful if its claimed contribution is never isolated.
5. **Matched budgets.** Compare models at comparable parameter counts and, where practical, comparable compute.
6. **Failure cases are results.** Document instability, sensitivity, and cases where the proposed architecture loses.
7. **Theory accompanies code.** Each track must include equations, shape contracts, complexity notes, and a mapping from paper notation to implementation names.
8. **Source-first development.** Architecture details must be checked against primary papers and official repositories before implementation.

## Repository layout

```text
modern-neural-networks-lab/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── common.yaml
│   └── tracks/
├── docs/
│   ├── architecture.md
│   ├── benchmark_protocol.md
│   ├── claim_policy.md
│   ├── experiment_contract.md
│   ├── mathematical_notation.md
│   ├── source_registry.md
│   └── track_matrix.md
├── prompts/
│   ├── 00_master_agent_prompt.md
│   ├── 01_kan.md
│   └── ...
├── src/modern_nn_lab/
│   ├── cli.py
│   ├── contracts.py
│   ├── registry.py
│   ├── reproducibility.py
│   └── tracks/
├── tests/
└── artifacts/                # generated outputs; ignored by Git
```

## Development order

Recommended implementation sequence:

1. KAN — small enough to establish the experiment framework.
2. Flow Matching — mathematically clear and self-contained.
3. Sparse MoE — establishes routing, load-balancing, and compute accounting.
4. xLSTM — establishes sequence benchmarks and recurrent-state tests.
5. TTT — extends the sequence framework with inference-time updates.
6. Mamba-3 — state-space implementation plus systems benchmarking.
7. Prior-Fitted Networks / TabPFN — establishes tabular benchmark infrastructure.
8. Titans-style memory — explicit long-term memory and online updates.
9. Relational foundation-model prototype — multi-table datasets and temporal leakage controls.
10. JEPA / world models — representation-prediction benchmark.
11. Nested Learning / Hope — last because it is the most research-prototype-oriented track and should benefit from all prior infrastructure.

The order is intentionally not a ranking of scientific importance. It minimizes framework churn.

## Quick start

```bash
poetry install
poetry run pytest
poetry run ruff check .
poetry run mypy src
poetry run modern-nn list-tracks
```

The project begins with scaffolding and contract tests. Track implementations are intentionally incomplete until the coding agent executes the prompts in `prompts/`.

## Definition of done for a track

A track is complete only when it has:

- primary sources recorded;
- mathematical specification;
- typed model implementation;
- unit tests for the mechanism, not just smoke tests;
- baseline(s);
- one synthetic diagnostic benchmark;
- one real benchmark where appropriate;
- matched-budget comparison;
- at least one ablation;
- deterministic/reproducible run path;
- metrics exported as machine-readable JSON/CSV;
- plots generated from saved metrics, not hard-coded values;
- limitations and failed experiments documented;
- a concise track report.

## License

No license is selected in this scaffold. Choose the repository license deliberately before public release.
