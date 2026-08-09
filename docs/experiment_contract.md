# Experiment Contract

Every reported experiment must serialize a record with at least:

- track name;
- architecture name;
- architecture version / git commit;
- dataset and immutable dataset fingerprint where possible;
- split strategy;
- seed;
- parameter count;
- effective tokens/samples processed;
- optimizer and scheduler;
- training steps/epochs;
- peak accelerator memory where measurable;
- training wall-clock time;
- inference latency/throughput when relevant;
- primary metric;
- secondary metrics;
- hardware descriptor;
- numerical precision;
- configuration snapshot;
- status: success / failed / diverged / interrupted.

## Comparison rules

At least two views are required when architecture size matters:

1. **Matched parameter budget**.
2. **Matched training/inference compute**, when practical.

Do not report only the favorable matching criterion.

## Statistical reporting

Default to at least 5 seeds for small/medium experiments. Report individual seeds plus mean, standard deviation, and a confidence interval or bootstrap interval when meaningful.

For expensive reference models, fewer seeds may be used only when explicitly justified in the report.
