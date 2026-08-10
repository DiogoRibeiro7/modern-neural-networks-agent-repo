"""Which relational paths can affect a prediction — the prompt's second acceptance criterion.

There are two different questions here and they are answered separately, because conflating
them is how explanation tools mislead.

**Which paths *can* affect this prediction?** A structural question, answered exactly. The
row set handed to the model is finite and its foreign-key pointers are explicit, so the set
of reachable paths can be enumerated with certainty rather than estimated. Every entry
carries the elapsed time of the rows on the path, which makes a temporal-leakage claim
checkable by inspection: over the rows that *have* a timestamp, every age must be strictly
positive. Rows from static tables are exempt and are marked as such, because a product's
price has no event time and its zero age is not evidence of anything.

**Which paths *did* affect it?** A quantitative question, answered by gradient attribution,
and therefore an approximation. A near-zero attribution means the model was insensitive to
that row at this input, not that the row is irrelevant in general.

The first is what the acceptance criterion asks for. The second is included because a
reachability list alone would let a model appear to use a relation it ignores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from modern_nn_lab.tracks.relational.sampler import (
    DT_INDEX,
    MASK_INDEX,
    NUMERIC_SLICE,
    PARENT_INDEX,
    TYPE_DISTRACTOR,
    TYPE_ENTITY,
    TYPE_EVENT,
    TYPE_LINKED,
)

TYPE_TO_TABLE = {
    TYPE_ENTITY: "customers",
    TYPE_EVENT: "orders",
    TYPE_LINKED: "products",
    TYPE_DISTRACTOR: "signals",
}
"""Row type to the table it came from, for human-readable paths."""


@dataclass(frozen=True, slots=True)
class PathTrace:
    """One relational path that can reach a prediction.

    Attributes:
        path: Table names from the target entity outwards, for example
            ``("customers", "orders", "products")``.
        slots: Row slots along the path, aligned with ``path``.
        age: Elapsed time of the outermost row at prediction time. Meaningful only when
            ``is_static`` is false — a static row has no timestamp and reports zero.
        is_static: Whether the outermost row came from a table with no time column. Static
            rows are facts rather than events and cannot carry future information, so they
            are exempt from the age check and must be excluded from it; treating their
            zero age as a leak would make the audit cry wolf on every product row.
        value: The outermost row's leading numeric column.
        attribution: Gradient-based influence, when computed.
    """

    path: tuple[str, ...]
    slots: tuple[int, ...]
    age: float
    value: float
    is_static: bool = False
    attribution: float | None = None

    def describe(self) -> str:
        """Render the path as a readable string."""

        arrow = " -> ".join(self.path)
        suffix = "" if self.attribution is None else f", attribution {self.attribution:+.4f}"
        return (
            f"{arrow} [slots {list(self.slots)}] "
            f"age {self.age:.3f}, value {self.value:+.3f}{suffix}"
        )


def reachable_paths(rows: Tensor, point: int) -> list[PathTrace]:
    """Enumerate every relational path present in one encoded neighbourhood.

    Args:
        rows: Shape ``(N, R, ROW_WIDTH)``.
        point: Index of the prediction point to explain.

    Returns:
        One :class:`PathTrace` per present non-entity row, ordered by slot.

    Raises:
        IndexError: If ``point`` is out of range.
    """

    if not 0 <= point < rows.shape[0]:
        raise IndexError(f"prediction point {point} out of range for {rows.shape[0]} points")

    sample = rows[point]
    traces: list[PathTrace] = []

    for slot in range(int(sample.shape[0])):
        if sample[slot, MASK_INDEX] <= 0:
            continue
        row_type = int(sample[slot, TYPE_ENTITY : TYPE_DISTRACTOR + 1].argmax())
        if row_type == TYPE_ENTITY:
            continue

        chain = [slot]
        cursor = slot
        # Walk parent pointers back to the root. The row set is a tree by construction, so
        # this terminates; the bound is a guard against a malformed input, not a heuristic.
        for _ in range(int(sample.shape[0])):
            parent = int(sample[cursor, PARENT_INDEX])
            if parent < 0:
                break
            chain.append(parent)
            cursor = parent

        chain.reverse()
        tables = tuple(
            TYPE_TO_TABLE[int(sample[step, TYPE_ENTITY : TYPE_DISTRACTOR + 1].argmax())]
            for step in chain
        )
        traces.append(
            PathTrace(
                path=tables,
                slots=tuple(chain),
                age=float(sample[slot, DT_INDEX]),
                value=float(sample[slot, NUMERIC_SLICE.start]),
                is_static=row_type == TYPE_LINKED,
            )
        )

    return traces


def attribute(model: nn.Module, rows: Tensor, point: int) -> list[PathTrace]:
    r"""Enumerate paths and attach a gradient attribution to each.

    The attribution is :math:`\sum_c x_c \, \partial \hat y / \partial x_c` over the
    row's channels — a first-order estimate of how much removing the row would change the
    logit.

    Args:
        model: A model mapping ``(B, R, ROW_WIDTH)`` to ``(B,)`` logits.
        rows: Shape ``(N, R, ROW_WIDTH)``.
        point: Index of the prediction point to explain.

    Returns:
        Paths with ``attribution`` populated.
    """

    was_training = model.training
    model.eval()
    try:
        sample = rows[point : point + 1].clone().requires_grad_(True)
        logit = model(sample).sum()
        (gradient,) = torch.autograd.grad(logit, sample)
    finally:
        model.train(was_training)

    influence = (gradient[0] * rows[point]).sum(dim=-1)
    return [
        PathTrace(
            path=trace.path,
            slots=trace.slots,
            age=trace.age,
            value=trace.value,
            is_static=trace.is_static,
            attribution=float(influence[trace.slots[-1]]),
        )
        for trace in reachable_paths(rows, point)
    ]


def summarize(traces: list[PathTrace]) -> dict[str, Any]:
    """Summarize a trace for the diagnostics artefact.

    Args:
        traces: Paths for one prediction point.

    Returns:
        A JSON-serializable summary. ``min_event_age`` is what a temporal-leakage audit
        reads first: over rows that carry a timestamp, it must be strictly positive.
    """

    # Only timed rows are eligible for the age check; see PathTrace.is_static.
    ages = [trace.age for trace in traces if not trace.is_static]
    by_depth: dict[int, int] = {}
    for trace in traces:
        by_depth[len(trace.path)] = by_depth.get(len(trace.path), 0) + 1

    return {
        "n_paths": len(traces),
        "paths_by_depth": {str(depth): count for depth, count in sorted(by_depth.items())},
        "n_timed_paths": len(ages),
        "min_event_age": min(ages) if ages else None,
        "max_event_age": max(ages) if ages else None,
        "distinct_paths": sorted({" -> ".join(trace.path) for trace in traces}),
        "examples": [trace.describe() for trace in traces[:6]],
    }
