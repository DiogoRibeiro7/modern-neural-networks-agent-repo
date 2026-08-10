"""Leakage-safe feature engineering — the flattening this track is arguing against.

This is the baseline the prompt asks for, and it is built to be a *strong* one rather than
a strawman. It receives the same time-gated row sets every neural model receives, so it
inherits the sampler's guarantee: no aggregate here can contain a post-timestamp row,
because no post-timestamp row was ever in the input.

What it gives up is structure, not information about *which rows exist*. Counts, sums,
means, extrema and a category histogram are computed per related table and concatenated
into one feature matrix. That is exactly the flattening the relational representation
avoids, and the interesting question the report asks is which regimes survive it. A
one-hop sum is preserved perfectly by this encoding; a multi-hop dependency is preserved
only because a human anticipated it and wrote the join. Nothing here adapts if the schema
gains a table.
"""

from __future__ import annotations

import torch
from torch import Tensor

from modern_nn_lab.tracks.relational.sampler import (
    CATEGORY_SLICE,
    DT_INDEX,
    MASK_INDEX,
    NUMERIC_SLICE,
    TYPE_DISTRACTOR,
    TYPE_ENTITY,
    TYPE_EVENT,
    TYPE_LINKED,
)

FEATURE_NAMES: tuple[str, ...] = (
    "entity_f0",
    "entity_f1",
    "entity_f2",
    "entity_region_0",
    "entity_region_1",
    "entity_region_2",
    "entity_region_3",
    "event_count",
    "event_amount_sum",
    "event_amount_mean",
    "event_amount_max",
    "event_amount_min",
    "event_recency_min",
    "event_recency_mean",
    "linked_price_mean",
    "linked_price_max",
    "linked_price_sum",
    "linked_category_0",
    "linked_category_1",
    "linked_category_2",
    "linked_category_3",
    "distractor_count",
    "distractor_value_mean",
)
"""Column names of the flattened matrix, in order. Used by the report and the tests."""


def _masked_stats(values: Tensor, present: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute count, sum, mean and max over the present entries of each row.

    Args:
        values: Shape ``(B, R)``.
        present: Shape ``(B, R)`` presence indicator.

    Returns:
        ``(count, total, mean, maximum)``, each of shape ``(B,)``. Empty groups yield zero
        rather than a NaN, so a missing relation is a value the tree can split on.
    """

    count = present.sum(dim=1)
    total = (values * present).sum(dim=1)
    mean = total / count.clamp_min(1.0)
    filled = torch.where(present > 0, values, torch.full_like(values, float("-inf")))
    maximum = filled.max(dim=1).values
    maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
    return count, total, mean, maximum


def flatten(rows: Tensor) -> Tensor:
    """Aggregate a typed row set into one feature vector per prediction point.

    Args:
        rows: Shape ``(N, R, ROW_WIDTH)`` time-gated row sets.

    Returns:
        Shape ``(N, len(FEATURE_NAMES))``.
    """

    mask = rows[..., MASK_INDEX]
    is_entity = (rows[..., TYPE_ENTITY] > 0).float() * mask
    is_event = (rows[..., TYPE_EVENT] > 0).float() * mask
    is_linked = (rows[..., TYPE_LINKED] > 0).float() * mask
    is_distractor = (rows[..., TYPE_DISTRACTOR] > 0).float() * mask

    numeric = rows[..., NUMERIC_SLICE.start]
    elapsed = rows[..., DT_INDEX]

    entity_numeric = (rows[..., NUMERIC_SLICE] * is_entity.unsqueeze(-1)).sum(dim=1)
    entity_region = (rows[..., CATEGORY_SLICE] * is_entity.unsqueeze(-1)).sum(dim=1)

    event_count, amount_sum, amount_mean, amount_max = _masked_stats(numeric, is_event)
    negated_min = _masked_stats(-numeric, is_event)[3]
    amount_min = torch.where(event_count > 0, -negated_min, torch.zeros_like(negated_min))

    # Recency is measured as elapsed time, so the *smallest* elapsed value is the most
    # recent event. Absent history yields zero, matching the count column beside it.
    _, elapsed_sum, elapsed_mean, _ = _masked_stats(elapsed, is_event)
    recency_min = -_masked_stats(-elapsed, is_event)[3]
    recency_min = torch.where(event_count > 0, recency_min, torch.zeros_like(recency_min))

    linked_count, price_sum, price_mean, price_max = _masked_stats(numeric, is_linked)
    linked_category = (rows[..., CATEGORY_SLICE] * is_linked.unsqueeze(-1)).sum(dim=1)

    distractor_count, _, distractor_mean, _ = _masked_stats(numeric, is_distractor)

    del elapsed_sum, linked_count

    return torch.stack(
        [
            entity_numeric[:, 0],
            entity_numeric[:, 1],
            entity_numeric[:, 2],
            entity_region[:, 0],
            entity_region[:, 1],
            entity_region[:, 2],
            entity_region[:, 3],
            event_count,
            amount_sum,
            amount_mean,
            amount_max,
            amount_min,
            recency_min,
            elapsed_mean,
            price_mean,
            price_max,
            price_sum,
            linked_category[:, 0],
            linked_category[:, 1],
            linked_category[:, 2],
            linked_category[:, 3],
            distractor_count,
            distractor_mean,
        ],
        dim=1,
    )
