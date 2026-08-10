r"""Temporal neighbourhood sampling — the one place time gating happens.

Every model in this track, and every baseline, receives its view of the database from this
module. That is a deliberate architectural choice rather than a convenience: if each model
applied its own timestamp filter, the leakage rules would be a matter of per-model
discipline, and the automated leakage tests would only be testing whichever model they
happened to call. Routing everything through one chokepoint makes "no event at or after the
prediction timestamp may influence this prediction" a property of the *pipeline*, provable
once.

The rule enforced here, from the track prompt:

.. math::

    \text{row } r \text{ is visible to prediction } (e, t^*)
    \iff r \text{ is static, or } t_r < t^*

Note the strict inequality. A row timestamped exactly at the prediction instant is not
visible; "as of time :math:`t^*`" means *before* :math:`t^*`, and admitting simultaneous
events is a common and easily-missed leak.

Three further things are gated on the same rule, because the prompt names all of them:

- **features** — only visible rows are encoded;
- **neighbourhoods** — a row that is not visible cannot bring in its own neighbours either,
  so an invisible order does not reveal the product it points at;
- **normalization** — statistics are fitted on training rows only, via :class:`Normalizer`,
  never on the split being scored.

:func:`build_row_sets` accepts ``include_future`` solely so the leakage tests have a
positive control: with it enabled a trivial model finds the canary of
:mod:`~modern_nn_lab.tracks.relational.generator` and scores near-perfectly, which is what
makes the default's inability to do so evidence rather than an assertion. **No experiment
sets it.**
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from modern_nn_lab.tracks.relational.generator import (
    N_CATEGORIES,
    N_REGIONS,
    RelationalProblem,
)

TYPE_ENTITY = 0
TYPE_EVENT = 1
TYPE_LINKED = 2
TYPE_DISTRACTOR = 3
N_TYPES = 4
"""Row types: the target entity, its events, what those events point at, and noise."""

N_NUMERIC = 3
N_CATEGORY = max(N_REGIONS, N_CATEGORIES)

TYPE_SLICE = slice(0, N_TYPES)
MASK_INDEX = N_TYPES
DT_INDEX = N_TYPES + 1
NUMERIC_SLICE = slice(N_TYPES + 2, N_TYPES + 2 + N_NUMERIC)
CATEGORY_SLICE = slice(N_TYPES + 2 + N_NUMERIC, N_TYPES + 2 + N_NUMERIC + N_CATEGORY)
PARENT_INDEX = N_TYPES + 2 + N_NUMERIC + N_CATEGORY
ROW_WIDTH = PARENT_INDEX + 1
"""Channel layout of one encoded row. Documented here and asserted in the tests."""


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """How much of the neighbourhood to materialize.

    Attributes:
        max_events: Events kept per entity, most recent first.
        max_distractors: Rows kept from the distractor table.
    """

    max_events: int = 8
    max_distractors: int = 6

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If either budget is negative.
        """

        if self.max_events < 0 or self.max_distractors < 0:
            raise ValueError("neighbourhood budgets must be non-negative")

    @property
    def n_rows(self) -> int:
        """Rows in one encoded neighbourhood: entity, events, their targets, distractors."""

        return 1 + 2 * self.max_events + self.max_distractors


def build_row_sets(
    problem: RelationalProblem,
    config: SamplingConfig | None = None,
    *,
    include_future: bool = False,
) -> Tensor:
    """Encode each prediction point as a set of typed, linked rows.

    Rows are kept as rows. Nothing is aggregated here — a model that wants a sum must
    compute one, which is what distinguishes this representation from the flattened feature
    matrix built by :mod:`modern_nn_lab.tracks.relational.features`.

    Args:
        problem: Database and task.
        config: Neighbourhood budgets. Defaults to :class:`SamplingConfig`.
        include_future: Disable time gating. **Only** the leakage tests pass ``True``,
            as a positive control; enabling it admits the generator's canary rows.

    Returns:
        Shape ``(N, R, ROW_WIDTH)``, with ``R = config.n_rows``.
    """

    settings = config or SamplingConfig()
    task = problem.task
    database = problem.database

    customers = database.table("customers")
    orders = database.table("orders")
    products = database.table("products")
    signals = database.table("signals")

    n_points = len(task)
    rows = torch.zeros((n_points, settings.n_rows, ROW_WIDTH))

    # --- Slot 0: the entity itself, a static row -------------------------------------
    entity = task.entity_ids.to(torch.long)
    rows[:, 0, TYPE_ENTITY] = 1.0
    rows[:, 0, MASK_INDEX] = 1.0
    for offset, name in enumerate(("f0", "f1", "f2")):
        rows[:, 0, NUMERIC_SLICE.start + offset] = customers.data[name][entity]
    region = customers.data["region"][entity].to(torch.long)
    rows[torch.arange(n_points), 0, CATEGORY_SLICE.start + region] = 1.0
    rows[:, 0, PARENT_INDEX] = -1.0  # the root has no parent

    order_owner = orders.data["customer_id"].to(torch.long)
    order_time = orders.data["order_time"]
    order_amount = orders.data["amount"]
    order_product = orders.data["product_id"].to(torch.long)

    signal_owner = signals.data["customer_id"].to(torch.long)
    signal_time = signals.data["signal_time"]
    signal_value = signals.data["value"]

    by_entity_orders = _group_rows(order_owner, n_points)
    by_entity_signals = _group_rows(signal_owner, n_points)

    for point in range(n_points):
        cutoff = float(task.timestamps[point])

        chosen = _visible_recent(
            by_entity_orders[point], order_time, cutoff, settings.max_events, include_future
        )
        for slot, row_index in enumerate(chosen):
            event_slot = 1 + slot
            rows[point, event_slot, TYPE_EVENT] = 1.0
            rows[point, event_slot, MASK_INDEX] = 1.0
            rows[point, event_slot, DT_INDEX] = _elapsed(cutoff, float(order_time[row_index]))
            rows[point, event_slot, NUMERIC_SLICE.start] = order_amount[row_index]
            rows[point, event_slot, PARENT_INDEX] = 0.0

            # The linked row arrives only because a *visible* event points at it. An
            # invisible event does not reveal its product.
            product = int(order_product[row_index])
            linked_slot = 1 + settings.max_events + slot
            rows[point, linked_slot, TYPE_LINKED] = 1.0
            rows[point, linked_slot, MASK_INDEX] = 1.0
            rows[point, linked_slot, NUMERIC_SLICE.start] = products.data["price"][product]
            category = int(products.data["category"][product])
            rows[point, linked_slot, CATEGORY_SLICE.start + category] = 1.0
            rows[point, linked_slot, PARENT_INDEX] = float(event_slot)

        distractors = _visible_recent(
            by_entity_signals[point],
            signal_time,
            cutoff,
            settings.max_distractors,
            include_future,
        )
        for slot, row_index in enumerate(distractors):
            noise_slot = 1 + 2 * settings.max_events + slot
            rows[point, noise_slot, TYPE_DISTRACTOR] = 1.0
            rows[point, noise_slot, MASK_INDEX] = 1.0
            rows[point, noise_slot, DT_INDEX] = _elapsed(cutoff, float(signal_time[row_index]))
            rows[point, noise_slot, NUMERIC_SLICE.start] = signal_value[row_index]
            rows[point, noise_slot, PARENT_INDEX] = 0.0

    return rows


def _elapsed(cutoff: float, event_time: float) -> float:
    """Encode how long before the prediction an event happened.

    Args:
        cutoff: Prediction timestamp.
        event_time: Event timestamp.

    Returns:
        ``log(1 + age)``, which compresses old history without discarding it.
    """

    return float(torch.log1p(torch.tensor(max(cutoff - event_time, 0.0))))


def _group_rows(owner: Tensor, n_points: int) -> list[list[int]]:
    """Group row positions by owning entity.

    Args:
        owner: Shape ``(n_rows,)`` owning entity per row.
        n_points: Number of entities.

    Returns:
        One list of row positions per entity.
    """

    grouped: list[list[int]] = [[] for _ in range(n_points)]
    for position, entity in enumerate(owner.tolist()):
        if 0 <= int(entity) < n_points:
            grouped[int(entity)].append(position)
    return grouped


def _visible_recent(
    candidates: list[int],
    times: Tensor,
    cutoff: float,
    budget: int,
    include_future: bool,
) -> list[int]:
    """Select the most recent rows that are visible at the cutoff.

    Args:
        candidates: Row positions belonging to this entity.
        times: Shape ``(n_rows,)`` event times.
        cutoff: Prediction timestamp.
        budget: Maximum rows to return.
        include_future: Skip the visibility filter. Positive control only.

    Returns:
        Up to ``budget`` row positions, most recent first.
    """

    if budget == 0:
        return []
    if include_future:
        visible = list(candidates)
    else:
        # Strict: a row timestamped exactly at the prediction instant is not visible.
        visible = [row for row in candidates if float(times[row]) < cutoff]
    visible.sort(key=lambda row: float(times[row]), reverse=True)
    return visible[:budget]


@dataclass(frozen=True, slots=True)
class Normalizer:
    """Per-channel standardization fitted on training rows only.

    Fitting on the split being scored would let the test set influence its own features
    through the normalization statistics, which the prompt forbids by name.

    Attributes:
        mean: Shape ``(ROW_WIDTH,)`` channel means.
        std: Shape ``(ROW_WIDTH,)`` channel standard deviations.
    """

    mean: Tensor
    std: Tensor

    @classmethod
    def fit(cls, rows: Tensor) -> Normalizer:
        """Fit statistics over the present rows of a training tensor.

        Args:
            rows: Shape ``(N, R, ROW_WIDTH)`` training rows.

        Returns:
            A fitted normalizer.
        """

        present = rows[..., MASK_INDEX] > 0
        flat = rows[present]
        mean = torch.zeros((ROW_WIDTH,))
        std = torch.ones((ROW_WIDTH,))
        if flat.numel():
            # Only the continuous channels are standardized. Type flags, the mask, and the
            # parent pointer are structure, and rescaling them would corrupt it.
            for index in (DT_INDEX, *range(NUMERIC_SLICE.start, NUMERIC_SLICE.stop)):
                mean[index] = flat[:, index].mean()
                std[index] = flat[:, index].std().clamp_min(1e-6)
        return cls(mean=mean, std=std)

    def __call__(self, rows: Tensor) -> Tensor:
        """Apply the fitted statistics.

        Args:
            rows: Shape ``(N, R, ROW_WIDTH)``.

        Returns:
            Standardized rows, with absent rows left at zero.
        """

        standardized = (rows - self.mean) / self.std
        present = (rows[..., MASK_INDEX] > 0).unsqueeze(-1)
        # Structural channels pass through untouched.
        structural = torch.zeros((ROW_WIDTH,), dtype=torch.bool)
        structural[TYPE_SLICE] = True
        structural[MASK_INDEX] = True
        structural[CATEGORY_SLICE] = True
        structural[PARENT_INDEX] = True
        standardized = torch.where(structural, rows, standardized)
        return standardized * present
