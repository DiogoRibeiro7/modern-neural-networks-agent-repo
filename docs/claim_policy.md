# Claim Policy

Use one of these labels in reports.

## `research prototype`

The weakest label. The mechanism is built from a specification — a track prompt, a textual
description, or first principles — **without a primary source having been read**, so nothing
in the track is checked against a published equation and no statement about the original work
is supported. Use this whenever the source was not retrieved, regardless of how carefully the
mechanism was implemented or how thoroughly it was tested.

## `educational implementation`

The core mechanism is implemented from primary sources, but scale, kernels, data, or training recipe differ materially from the paper.

**"From primary sources" is a precondition, not a description of intent.** If the source was
not read, this label is unsupported and `research prototype` applies instead — however
faithful the implementation may be to a remembered or inferred formulation.

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
