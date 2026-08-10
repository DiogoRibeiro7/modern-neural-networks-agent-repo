"""Invariants of the relational track, including the two acceptance criteria.

The prompt requires automated temporal-leakage tests and an explainable trace of which
relational paths can affect a prediction. Both are tested here as properties of the
pipeline rather than of any one model, because every model in the track receives its view
of the database from the same sampler.

The leakage tests are built around a *positive control*. It is easy to write a test that
passes because nothing was checked; these tests first show that a deliberately ungated
sampler does find the generator's post-timestamp canary and does reach near-perfect
accuracy on it, and only then assert that the real sampler cannot.
"""

from __future__ import annotations

import pytest
import torch

from modern_nn_lab.tracks.relational import (
    FEATURE_NAMES,
    REGIMES,
    Column,
    Database,
    ForeignKey,
    Normalizer,
    RelationalConfig,
    RelationalEncoder,
    SamplingConfig,
    Table,
    TargetOnlyModel,
    attribute,
    build_row_sets,
    flatten,
    generate,
    leakage_canary_strength,
    reachable_paths,
    summarize,
)
from modern_nn_lab.tracks.relational.sampler import (
    DT_INDEX,
    MASK_INDEX,
    NUMERIC_SLICE,
    PARENT_INDEX,
    ROW_WIDTH,
    TYPE_ENTITY,
    TYPE_EVENT,
    TYPE_LINKED,
)

SMALL = {"n_entities": 60, "n_products": 12, "seed": 7}

# --------------------------------------------------------------------------------------
# Acceptance criterion 1: temporal leakage
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("regime", REGIMES)
def test_no_sampled_row_is_at_or_after_the_prediction_time(regime: str) -> None:
    """The rule, checked against the source tables rather than the encoding.

    Every event row the sampler emits must correspond to a row of the database strictly
    earlier than the prediction timestamp. This walks back to the original tables instead
    of trusting the elapsed-time channel, so an encoding bug cannot hide a leak.
    """

    problem = generate(regime, **SMALL)  # type: ignore[arg-type]
    rows = build_row_sets(problem)

    orders = problem.database.table("orders")
    order_times = orders.data["order_time"]
    order_owner = orders.data["customer_id"].to(torch.long)

    for point in range(len(problem.task)):
        cutoff = float(problem.task.timestamps[point])
        visible_times = {
            float(order_times[i])
            for i in range(int(order_times.shape[0]))
            if int(order_owner[i]) == point and float(order_times[i]) < cutoff
        }
        for slot in range(rows.shape[1]):
            if rows[point, slot, MASK_INDEX] <= 0 or rows[point, slot, TYPE_EVENT] <= 0:
                continue
            age = float(rows[point, slot, DT_INDEX])
            assert age > 0.0, f"{regime}: event row at slot {slot} has non-positive age"
            recovered = cutoff - (torch.expm1(torch.tensor(age)).item())
            assert any(abs(recovered - t) < 1e-3 for t in visible_times), (
                f"{regime}: sampled an event that is not a visible row of the database"
            )


@pytest.mark.parametrize("regime", REGIMES)
def test_the_canary_is_findable_when_gating_is_disabled(regime: str) -> None:
    """Positive control: the shortcut exists, so the gated test is not vacuous."""

    problem = generate(regime, **SMALL)  # type: ignore[arg-type]
    assert leakage_canary_strength(problem) > 0.99

    gated = build_row_sets(problem)
    ungated = build_row_sets(problem, include_future=True)
    assert (ungated[..., MASK_INDEX] > 0).sum() > (gated[..., MASK_INDEX] > 0).sum()


def test_a_trivial_model_solves_the_ungated_task_and_fails_the_gated_one() -> None:
    """The end-to-end leakage test.

    Summing event amounts is enough to read the canary. Given ungated rows that rule is
    near-perfect; given the sampler's own rows it is not, and the gap is the evidence that
    gating happened.
    """

    problem = generate("cold_start", **SMALL)
    labels = problem.task.labels

    def sum_rule(rows: torch.Tensor) -> torch.Tensor:
        is_event = (rows[..., TYPE_EVENT] > 0) & (rows[..., MASK_INDEX] > 0)
        totals = (rows[..., NUMERIC_SLICE.start] * is_event).sum(dim=1)
        return (totals > 0).long()

    ungated = float(
        (sum_rule(build_row_sets(problem, include_future=True)) == labels).float().mean()
    )
    gated = float((sum_rule(build_row_sets(problem)) == labels).float().mean())

    assert ungated > 0.99, "the canary should be trivially readable without gating"
    # cold_start has no visible events at all, so the rule has nothing to go on.
    assert gated < 0.75, f"gated rows still leak: the same rule scores {gated:.3f}"


def test_a_row_timestamped_exactly_at_the_prediction_instant_is_excluded() -> None:
    """ "As of t" means strictly before t. Admitting simultaneity is an easy leak."""

    problem = generate("one_hop", n_entities=4, n_products=3, seed=0)
    orders = problem.database.table("orders")
    # Move one order onto entity 0's prediction instant exactly.
    orders.data["order_time"][0] = problem.task.timestamps[int(orders.data["customer_id"][0])]

    rows = build_row_sets(problem)
    owner = int(orders.data["customer_id"][0])
    cutoff = float(problem.task.timestamps[owner])
    ages = [
        float(rows[owner, slot, DT_INDEX])
        for slot in range(rows.shape[1])
        if rows[owner, slot, MASK_INDEX] > 0 and rows[owner, slot, TYPE_EVENT] > 0
    ]
    assert all(age > 0.0 for age in ages)
    assert all(abs(cutoff - float(torch.expm1(torch.tensor(age)))) > 1e-9 for age in ages)


def test_an_invisible_event_does_not_reveal_the_row_it_points_at() -> None:
    """Neighbourhood expansion is gated too, not only feature extraction."""

    problem = generate("multi_hop", **SMALL)
    rows = build_row_sets(problem)

    for point in range(len(problem.task)):
        events = [
            slot
            for slot in range(rows.shape[1])
            if rows[point, slot, MASK_INDEX] > 0 and rows[point, slot, TYPE_EVENT] > 0
        ]
        linked = [
            slot
            for slot in range(rows.shape[1])
            if rows[point, slot, MASK_INDEX] > 0 and rows[point, slot, TYPE_LINKED] > 0
        ]
        assert len(linked) == len(events), "a linked row appeared without its referencing event"
        parents = {int(rows[point, slot, PARENT_INDEX]) for slot in linked}
        assert parents <= set(events), "a linked row points at an event that is not present"


def test_normalization_statistics_come_from_training_rows_only() -> None:
    """Fitting on the scored split would let it influence its own features."""

    problem = generate("one_hop", **SMALL)
    rows = build_row_sets(problem)
    train, test = rows[:40], rows[40:]

    fitted = Normalizer.fit(train)
    baseline = fitted(train).clone()

    # Corrupting the held-out rows must not move the training encoding at all.
    corrupted = test.clone()
    corrupted[..., NUMERIC_SLICE.start] += 1000.0
    assert torch.equal(fitted(train), baseline)

    refitted = Normalizer.fit(torch.cat([train, corrupted]))
    assert not torch.allclose(refitted(train), baseline), "the test is not sensitive enough"


# --------------------------------------------------------------------------------------
# Acceptance criterion 2: an explainable trace of the reachable paths
# --------------------------------------------------------------------------------------


def test_the_trace_enumerates_exactly_the_present_rows() -> None:
    problem = generate("multi_hop", **SMALL)
    rows = build_row_sets(problem)
    traces = reachable_paths(rows, 0)

    present_non_entity = int(((rows[0, :, MASK_INDEX] > 0) & (rows[0, :, TYPE_ENTITY] <= 0)).sum())
    assert len(traces) == present_non_entity
    assert all(trace.path[0] == "customers" for trace in traces)
    assert all(trace.slots[0] == 0 for trace in traces)


def test_multi_hop_paths_are_reported_at_depth_three() -> None:
    """A products row must be reported as reached *through* an order, not directly."""

    problem = generate("multi_hop", **SMALL)
    traces = reachable_paths(build_row_sets(problem), 0)
    product_paths = [trace for trace in traces if trace.path[-1] == "products"]

    assert product_paths, "expected at least one two-hop path"
    for trace in product_paths:
        assert trace.path == ("customers", "orders", "products")
        assert len(trace.slots) == 3


@pytest.mark.parametrize("regime", REGIMES)
def test_every_traced_event_path_has_a_positive_age(regime: str) -> None:
    """The audit quantity, checked over every prediction point rather than sampled."""

    problem = generate(regime, **SMALL)  # type: ignore[arg-type]
    rows = build_row_sets(problem)
    for point in range(len(problem.task)):
        entry = summarize(reachable_paths(rows, point))
        if entry["min_event_age"] is not None:
            assert entry["min_event_age"] > 0.0


def test_static_rows_are_exempt_from_the_age_check() -> None:
    """A product has no event time; its zero age must not be read as a leak."""

    problem = generate("multi_hop", **SMALL)
    traces = reachable_paths(build_row_sets(problem), 0)
    static = [trace for trace in traces if trace.is_static]

    assert static, "expected static product rows in a multi-hop neighbourhood"
    assert all(trace.path[-1] == "products" for trace in static)
    entry = summarize(traces)
    assert entry["n_timed_paths"] < entry["n_paths"]


def test_attribution_covers_every_reachable_path() -> None:
    problem = generate("one_hop", **SMALL)
    rows = build_row_sets(problem)
    model = RelationalEncoder(RelationalConfig(d_model=16))
    traces = attribute(model, rows, 0)

    assert len(traces) == len(reachable_paths(rows, 0))
    assert all(trace.attribution is not None for trace in traces)
    assert any(abs(trace.attribution or 0.0) > 0 for trace in traces)


def test_attribution_leaves_the_model_in_its_original_mode() -> None:
    problem = generate("one_hop", **SMALL)
    model = RelationalEncoder(RelationalConfig(d_model=16))
    model.train()
    attribute(model, build_row_sets(problem), 0)
    assert model.training


def test_reachable_paths_rejects_an_out_of_range_point() -> None:
    problem = generate("one_hop", n_entities=4, n_products=3, seed=0)
    with pytest.raises(IndexError, match="out of range"):
        reachable_paths(build_row_sets(problem), 99)


# --------------------------------------------------------------------------------------
# The architecture claims, asserted structurally
# --------------------------------------------------------------------------------------


def build_rows(n_points: int = 3) -> torch.Tensor:
    """Return a small batch of encoded neighbourhoods."""

    problem = generate("multi_hop", n_entities=n_points, n_products=8, seed=3)
    return build_row_sets(problem)


def test_one_round_cannot_reach_a_two_hop_attribute() -> None:
    """The claim that ``multi_hop`` needs two rounds, tested rather than argued.

    With a single round, a linked row's information has only reached the event that
    references it — not the entity that is read out. Changing a product's price must
    therefore leave a one-round model's prediction untouched, and must change a
    two-round model's.
    """

    rows = build_rows()
    perturbed = rows.clone()
    is_linked = (perturbed[..., TYPE_LINKED] > 0) & (perturbed[..., MASK_INDEX] > 0)
    perturbed[..., NUMERIC_SLICE.start] += is_linked.float() * 5.0

    torch.manual_seed(0)
    shallow = RelationalEncoder(RelationalConfig(d_model=16, n_rounds=1)).eval()
    torch.manual_seed(0)
    deep = RelationalEncoder(RelationalConfig(d_model=16, n_rounds=2)).eval()

    with torch.no_grad():
        assert torch.allclose(shallow(rows), shallow(perturbed), atol=1e-6)
        assert not torch.allclose(deep(rows), deep(perturbed), atol=1e-5)


def test_disabling_links_makes_the_model_ignore_foreign_keys() -> None:
    """The homogeneous baseline must genuinely discard structure."""

    rows = build_rows()
    rewired = rows.clone()
    # Point every non-root row at a different parent. A structure-aware model must notice.
    movable = rewired[..., PARENT_INDEX] > 0
    rewired[..., PARENT_INDEX] = torch.where(
        movable, torch.zeros_like(rewired[..., PARENT_INDEX]), rewired[..., PARENT_INDEX]
    )

    torch.manual_seed(0)
    flat = RelationalEncoder(RelationalConfig(d_model=16, use_links=False)).eval()
    torch.manual_seed(0)
    linked = RelationalEncoder(RelationalConfig(d_model=16, use_links=True)).eval()

    with torch.no_grad():
        assert torch.allclose(flat(rows), flat(rewired), atol=1e-6)
        assert not torch.allclose(linked(rows), linked(rewired), atol=1e-5)


def test_disabling_time_makes_the_model_ignore_elapsed_time() -> None:
    rows = build_rows()
    retimed = rows.clone()
    retimed[..., DT_INDEX] = retimed[..., DT_INDEX] * 3.0 + 1.0

    torch.manual_seed(0)
    timeless = RelationalEncoder(RelationalConfig(d_model=16, use_time=False)).eval()
    torch.manual_seed(0)
    timed = RelationalEncoder(RelationalConfig(d_model=16, use_time=True)).eval()

    with torch.no_grad():
        assert torch.allclose(timeless(rows), timeless(retimed), atol=1e-6)
        assert not torch.allclose(timed(rows), timed(retimed), atol=1e-5)


def test_typed_encoders_treat_identical_features_differently_by_table() -> None:
    """Without this, `use_types` would be a flag that changes nothing."""

    rows = build_rows()
    torch.manual_seed(0)
    typed = RelationalEncoder(RelationalConfig(d_model=16, use_types=True)).eval()
    torch.manual_seed(0)
    shared = RelationalEncoder(RelationalConfig(d_model=16, use_types=False)).eval()

    with torch.no_grad():
        assert not torch.allclose(typed(rows), shared(rows), atol=1e-4)


def test_target_only_model_ignores_every_related_row() -> None:
    rows = build_rows()
    scrambled = rows.clone()
    scrambled[:, 1:, :] = 0.0

    model = TargetOnlyModel(RelationalConfig(d_model=16)).eval()
    with torch.no_grad():
        assert torch.allclose(model(rows), model(scrambled), atol=1e-6)


def test_models_reject_a_row_set_of_the_wrong_width() -> None:
    model = RelationalEncoder(RelationalConfig(d_model=16))
    with pytest.raises(ValueError, match="width"):
        model(torch.zeros((2, 5, ROW_WIDTH + 1)))


# --------------------------------------------------------------------------------------
# Generation, flattening, and schema validation
# --------------------------------------------------------------------------------------


def test_cold_start_entities_have_no_visible_history() -> None:
    """The regime is defined by the absence, so the absence is what gets checked."""

    problem = generate("cold_start", **SMALL)
    rows = build_row_sets(problem)
    events = (rows[..., TYPE_EVENT] > 0) & (rows[..., MASK_INDEX] > 0)
    assert int(events.sum()) == 0


def test_the_distractor_table_is_larger_in_the_irrelevant_regime() -> None:
    baseline = generate("one_hop", **SMALL)
    buried = generate("irrelevant", **SMALL)
    assert buried.database.table("signals").n_rows > 2 * baseline.database.table("signals").n_rows


@pytest.mark.parametrize("regime", REGIMES)
def test_labels_are_not_degenerate(regime: str) -> None:
    problem = generate(regime, n_entities=300, n_products=20, seed=11)  # type: ignore[arg-type]
    balance = float(problem.task.labels.float().mean())
    assert 0.3 < balance < 0.7, f"{regime}: label balance {balance:.2f}"


def test_generate_rejects_an_unknown_regime() -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        generate("nope", n_entities=4)  # type: ignore[arg-type]


def test_flatten_produces_the_documented_columns() -> None:
    rows = build_rows(n_points=5)
    features = flatten(rows)
    assert features.shape == (5, len(FEATURE_NAMES))
    assert torch.isfinite(features).all(), "an empty relation must yield zero, not NaN"


def test_flatten_of_an_empty_neighbourhood_is_all_zero_counts() -> None:
    rows = torch.zeros((1, 9, ROW_WIDTH))
    rows[0, 0, TYPE_ENTITY] = 1.0
    rows[0, 0, MASK_INDEX] = 1.0
    features = flatten(rows)
    assert float(features[0, FEATURE_NAMES.index("event_count")]) == 0.0
    assert float(features[0, FEATURE_NAMES.index("event_amount_sum")]) == 0.0
    assert torch.isfinite(features).all()


def test_schema_validation() -> None:
    with pytest.raises(ValueError, match="cardinality"):
        Column("c", "categorical")
    with pytest.raises(ValueError, match="only categorical"):
        Column("c", "numeric", 3)
    with pytest.raises(ValueError, match="duplicate"):
        Table("t", (Column("a", "numeric"), Column("a", "numeric")))
    with pytest.raises(ValueError, match="differing lengths"):
        Table(
            "t",
            (Column("a", "numeric"), Column("b", "numeric")),
            {"a": torch.zeros(3), "b": torch.zeros(2)},
        )
    with pytest.raises(ValueError, match="does not match columns"):
        Table("t", (Column("a", "numeric"),), {"z": torch.zeros(3)})
    with pytest.raises(ValueError, match="unknown table"):
        Database({}, (ForeignKey("a", "b", "c"),))


def test_static_and_timed_tables_are_distinguished() -> None:
    problem = generate("one_hop", **SMALL)
    assert problem.database.table("products").is_static
    assert not problem.database.table("orders").is_static
    assert problem.database.table("orders").time_column is not None


def test_sampling_config_validates_its_budgets() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SamplingConfig(max_events=-1)
    assert SamplingConfig(max_events=3, max_distractors=2).n_rows == 1 + 3 + 3 + 2


def test_relational_config_validates_its_sizes() -> None:
    with pytest.raises(ValueError, match="d_model"):
        RelationalConfig(d_model=0)
    with pytest.raises(ValueError, match="n_rounds"):
        RelationalConfig(n_rounds=0)
