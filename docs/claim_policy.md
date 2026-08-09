# Claim Policy

Use one of these labels in reports.

## `educational implementation`

The core mechanism is implemented from primary sources, but scale, kernels, data, or training recipe differ materially from the paper.

## `compact reproduction`

The architecture and experiment are sufficiently faithful to reproduce a small paper experiment or a clearly specified subset, with deviations documented.

## `reference integration`

The official authors' implementation/checkpoint is invoked for comparison. The repository does not claim authorship of that implementation.

## `independent reproduction`

Use only if architecture, data, preprocessing, optimizer, evaluation, scale, and relevant systems details are sufficiently aligned and the reported result is actually reproduced within stated tolerance.

## Forbidden wording without evidence

Do not write:

- "state of the art";
- "reproduces the paper";
- "equivalent to the official implementation";
- "faster" without hardware and measurement protocol;
- "more efficient" without defining the efficiency metric;
- "better" without uncertainty and comparison conditions.
