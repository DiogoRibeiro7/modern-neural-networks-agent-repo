r"""Synthetic relational databases whose targets depend on a known relational path.

Each regime is built so that exactly one kind of relational reasoning suffices, which is
what makes a failure interpretable: a model that scores at chance on ``multi_hop`` but well
on ``one_hop`` is not following foreign keys past the first hop, and no other explanation
is needed.

The five regimes required by the track prompt:

===============  ==========================================================================
``one_hop``      the label is a function of the entity's own events
``multi_hop``    the label is a function of the *attributes of what those events point at*
``temporal``     only recent events count, and older ones point the other way
``irrelevant``   the one-hop signal is present but buried under a much larger noise table
``cold_start``   the entity has **no** events before the prediction time
===============  ==========================================================================

Every regime also carries a **leakage canary**. After each prediction timestamp, orders are
inserted whose amount encodes the label exactly. They are legitimate rows of the database —
they simply have not happened yet at prediction time. Any pipeline that fails to gate on
time will find them and score near-perfectly, so a suspiciously good result is diagnostic
rather than mysterious. :func:`leakage_canary_strength` reports how strong that shortcut is,
and ``tests/test_relational.py`` asserts that the sampler never surfaces one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from modern_nn_lab.tracks.relational.schema import Column, Database, ForeignKey, Table

Regime = Literal["one_hop", "multi_hop", "temporal", "irrelevant", "cold_start"]
"""The five diagnostic regimes required by the track prompt."""

REGIMES: tuple[Regime, ...] = ("one_hop", "multi_hop", "temporal", "irrelevant", "cold_start")

N_REGIONS = 4
N_CATEGORIES = 4
CANARY_AMOUNT = 50.0
"""Amount written on post-timestamp orders. Large enough to dominate any aggregate."""


@dataclass(frozen=True, slots=True)
class PredictionTask:
    """What is being predicted, for whom, and as of when.

    Attributes:
        entity_ids: Shape ``(N,)`` primary keys of the target table.
        timestamps: Shape ``(N,)`` prediction times. Nothing at or after this time may
            influence the prediction for that row.
        labels: Shape ``(N,)`` binary targets.
        regime: Which relational structure generated the labels.
    """

    entity_ids: Tensor
    timestamps: Tensor
    labels: Tensor
    regime: Regime

    def __post_init__(self) -> None:
        """Validate the task.

        Raises:
            ValueError: If the three tensors disagree in length.
        """

        lengths = {
            int(self.entity_ids.shape[0]),
            int(self.timestamps.shape[0]),
            int(self.labels.shape[0]),
        }
        if len(lengths) != 1:
            raise ValueError(f"entity ids, timestamps and labels must align; got {lengths}")

    def __len__(self) -> int:
        """Number of prediction points."""

        return int(self.entity_ids.shape[0])


@dataclass(frozen=True, slots=True)
class RelationalProblem:
    """A database together with the task defined over it.

    Attributes:
        database: The tables and their links.
        task: The prediction points.
    """

    database: Database
    task: PredictionTask


def build_schema() -> tuple[tuple[Column, ...], ...]:
    """Return the column declarations for the four tables, in a fixed order.

    Returns:
        ``(customers, orders, products, signals)`` column tuples.
    """

    customers = (
        Column("customer_id", "primary_key"),
        Column("f0", "numeric"),
        Column("f1", "numeric"),
        Column("f2", "numeric"),
        Column("region", "categorical", N_REGIONS),
        Column("signup_time", "time"),
    )
    orders = (
        Column("order_id", "primary_key"),
        Column("customer_id", "foreign_key"),
        Column("product_id", "foreign_key"),
        Column("amount", "numeric"),
        Column("order_time", "time"),
    )
    products = (
        Column("product_id", "primary_key"),
        Column("price", "numeric"),
        Column("category", "categorical", N_CATEGORIES),
    )
    signals = (
        Column("signal_id", "primary_key"),
        Column("customer_id", "foreign_key"),
        Column("value", "numeric"),
        Column("signal_time", "time"),
    )
    return customers, orders, products, signals


def _poisson(rate: float, size: int, generator: torch.Generator) -> Tensor:
    """Draw Poisson counts.

    Args:
        rate: Mean count.
        size: Number of draws.
        generator: Seeded generator.

    Returns:
        Shape ``(size,)`` integer counts.
    """

    return torch.poisson(torch.full((size,), rate), generator=generator).to(torch.long)


def generate(
    regime: Regime,
    *,
    n_entities: int = 600,
    n_products: int = 40,
    orders_per_entity: float = 8.0,
    signals_per_entity: float = 3.0,
    horizon: float = 100.0,
    recent_window: float = 20.0,
    seed: int = 0,
) -> RelationalProblem:
    """Generate a database and a labelled task for one regime.

    Args:
        regime: Which relational structure the label depends on.
        n_entities: Number of customers, one prediction point each.
        n_products: Number of products.
        orders_per_entity: Mean pre-timestamp order count.
        signals_per_entity: Mean count in the distractor table; raised for ``irrelevant``.
        horizon: Length of the time axis.
        recent_window: Width of the window that counts in the ``temporal`` regime.
        seed: Seed for every draw.

    Returns:
        The database and its task.

    Raises:
        ValueError: If the regime is unknown.
    """

    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; available: {list(REGIMES)}")

    generator = torch.Generator().manual_seed(seed)
    customer_cols, order_cols, product_cols, signal_cols = build_schema()

    # --- Static tables -----------------------------------------------------------------
    product_price = torch.rand((n_products,), generator=generator) * 10.0
    products = Table(
        "products",
        product_cols,
        {
            "product_id": torch.arange(n_products),
            "price": product_price,
            "category": torch.randint(0, N_CATEGORIES, (n_products,), generator=generator).float(),
        },
    )

    customer_ids = torch.arange(n_entities)
    static = torch.randn((n_entities, 3), generator=generator)
    region = torch.randint(0, N_REGIONS, (n_entities,), generator=generator)
    # Prediction times sit in the second half of the horizon, leaving room for history.
    timestamps = horizon * 0.5 + torch.rand((n_entities,), generator=generator) * horizon * 0.4

    customers = Table(
        "customers",
        customer_cols,
        {
            "customer_id": customer_ids,
            "f0": static[:, 0],
            "f1": static[:, 1],
            "f2": static[:, 2],
            "region": region.float(),
            "signup_time": torch.zeros((n_entities,)),
        },
    )

    # --- Events ------------------------------------------------------------------------
    counts = _poisson(orders_per_entity, n_entities, generator)
    if regime == "cold_start":
        # The defining property: nothing of this entity's own history is visible.
        counts = torch.zeros_like(counts)

    owner: list[int] = []
    order_time: list[float] = []
    for entity in range(n_entities):
        for _ in range(int(counts[entity])):
            owner.append(entity)
            order_time.append(
                float(torch.rand((), generator=generator)) * float(timestamps[entity])
            )

    n_orders = len(owner)
    order_owner = torch.tensor(owner, dtype=torch.long)
    order_times = torch.tensor(order_time, dtype=torch.float32)
    order_product = torch.randint(0, n_products, (n_orders,), generator=generator)
    order_amount = torch.rand((n_orders,), generator=generator) * 4.0 - 2.0

    if regime == "temporal":
        # Older orders point the other way, so a model that ignores time is worse than one
        # that sees nothing at all.
        age = timestamps[order_owner] - order_times
        stale = age > recent_window
        order_amount = torch.where(stale, -2.0 * order_amount, order_amount)

    # --- Labels: computed from pre-timestamp rows only, by construction ----------------
    statistic = _label_statistic(
        regime,
        n_entities=n_entities,
        order_owner=order_owner,
        order_times=order_times,
        order_amount=order_amount,
        order_product=order_product,
        product_price=product_price,
        timestamps=timestamps,
        recent_window=recent_window,
        static=static,
        region=region,
    )
    # Thresholded at zero, not at the population median. Every statistic below is a sum of
    # terms already centred on zero, so this is balanced without consulting a quantile
    # computed over entities whose prediction timestamps lie in the future — which would be
    # a cross-sectional leak into target construction, the subtlest kind the prompt forbids.
    labels = (statistic > 0).long()

    # --- The leakage canary: real rows, but all of them are in the future --------------
    canary_owner = customer_ids
    canary_time = timestamps + 1.0 + torch.rand((n_entities,), generator=generator) * 5.0
    canary_amount = torch.where(
        labels.bool(),
        torch.full((n_entities,), CANARY_AMOUNT),
        torch.full((n_entities,), -CANARY_AMOUNT),
    )
    canary_product = torch.randint(0, n_products, (n_entities,), generator=generator)

    orders = Table(
        "orders",
        order_cols,
        {
            "order_id": torch.arange(n_orders + n_entities),
            "customer_id": torch.cat([order_owner, canary_owner]).float(),
            "product_id": torch.cat([order_product, canary_product]).float(),
            "amount": torch.cat([order_amount, canary_amount]),
            "order_time": torch.cat([order_times, canary_time]),
        },
    )

    # --- The distractor table ----------------------------------------------------------
    signal_rate = signals_per_entity * (5.0 if regime == "irrelevant" else 1.0)
    signal_counts = _poisson(signal_rate, n_entities, generator)
    signal_owner_list: list[int] = []
    signal_time_list: list[float] = []
    for entity in range(n_entities):
        for _ in range(int(signal_counts[entity])):
            signal_owner_list.append(entity)
            signal_time_list.append(
                float(torch.rand((), generator=generator)) * float(timestamps[entity])
            )

    n_signals = len(signal_owner_list)
    signals = Table(
        "signals",
        signal_cols,
        {
            "signal_id": torch.arange(n_signals),
            "customer_id": torch.tensor(signal_owner_list, dtype=torch.float32),
            "value": torch.randn((n_signals,), generator=generator),
            "signal_time": torch.tensor(signal_time_list, dtype=torch.float32),
        },
    )

    database = Database(
        tables={
            "customers": customers,
            "orders": orders,
            "products": products,
            "signals": signals,
        },
        foreign_keys=(
            ForeignKey("orders", "customer_id", "customers"),
            ForeignKey("orders", "product_id", "products"),
            ForeignKey("signals", "customer_id", "customers"),
        ),
    )
    task = PredictionTask(
        entity_ids=customer_ids, timestamps=timestamps, labels=labels, regime=regime
    )
    return RelationalProblem(database=database, task=task)


def _label_statistic(
    regime: Regime,
    *,
    n_entities: int,
    order_owner: Tensor,
    order_times: Tensor,
    order_amount: Tensor,
    order_product: Tensor,
    product_price: Tensor,
    timestamps: Tensor,
    recent_window: float,
    static: Tensor,
    region: Tensor,
) -> Tensor:
    """Compute the statistic the label thresholds, per regime.

    Every branch reads only rows strictly before the entity's prediction timestamp, so the
    label itself cannot encode future information — the canary rows are added afterwards.

    Args:
        regime: Which relational structure the label depends on.
        n_entities: Number of entities.
        order_owner: Shape ``(n_orders,)`` owning entity per order.
        order_times: Shape ``(n_orders,)`` event times.
        order_amount: Shape ``(n_orders,)`` amounts.
        order_product: Shape ``(n_orders,)`` product keys.
        product_price: Shape ``(n_products,)`` prices.
        timestamps: Shape ``(n_entities,)`` prediction times.
        recent_window: Window width for the temporal regime.
        static: Shape ``(n_entities, 3)`` static features.
        region: Shape ``(n_entities,)`` region codes.

    Returns:
        Shape ``(n_entities,)`` statistic.
    """

    if regime == "cold_start":
        # No history exists, so the only recoverable signal is the entity's own attributes
        # and the region it belongs to.
        region_effect = torch.tensor([1.0, -1.0, 0.5, -0.5])[region]
        return static[:, 0] + 0.5 * static[:, 1] + region_effect

    visible = order_times < timestamps[order_owner]
    statistic = torch.zeros((n_entities,))

    if regime == "multi_hop":
        # Requires following orders -> products: the amount is irrelevant here.
        contribution = product_price[order_product] - float(product_price.mean())
    elif regime == "temporal":
        recent = (timestamps[order_owner] - order_times) <= recent_window
        contribution = torch.where(recent, order_amount, torch.zeros_like(order_amount))
    else:  # one_hop and irrelevant share the same rule
        contribution = order_amount

    statistic.index_add_(0, order_owner[visible], contribution[visible])
    return statistic


def leakage_canary_strength(problem: RelationalProblem) -> float:
    """Accuracy achievable by reading post-timestamp orders — the shortcut's payoff.

    A pipeline that gates on time correctly cannot reach this number; one that does not
    will land near it. Reporting it makes "suspiciously high accuracy" a testable claim.

    Args:
        problem: The generated problem.

    Returns:
        Accuracy of thresholding each entity's future-order amount at zero.
    """

    orders = problem.database.table("orders")
    owner = orders.data["customer_id"].to(torch.long)
    times = orders.data["order_time"]
    amounts = orders.data["amount"]

    task = problem.task
    future = times >= task.timestamps[owner]
    score = torch.zeros((len(task),))
    score.index_add_(0, owner[future], amounts[future])
    return float(((score > 0).long() == task.labels).float().mean())
