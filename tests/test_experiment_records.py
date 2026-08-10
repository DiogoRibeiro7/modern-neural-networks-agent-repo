"""Invariants of the experiment-record schema."""

from __future__ import annotations

import json

import pytest
import torch
from pydantic import ValidationError

from modern_nn_lab.experiments.records import (
    RESULT_SCHEMA_VERSION,
    fingerprint,
    format_markdown_table,
    iter_records,
    load_record,
    records_to_rows,
    save_record,
    tensor_fingerprint,
)
from tests.conftest import make_record


def test_record_defaults_to_current_schema_version() -> None:
    assert make_record().schema_version == RESULT_SCHEMA_VERSION


def test_record_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        make_record(unexpected_field="nope")


def test_record_rejects_negative_seed_and_parameters() -> None:
    with pytest.raises(ValidationError):
        make_record(seed=-1)
    with pytest.raises(ValidationError):
        make_record(parameter_count=-5)
    with pytest.raises(ValidationError):
        make_record(train_wall_clock_s=-0.1)


def test_record_is_immutable() -> None:
    record = make_record()
    with pytest.raises(ValidationError):
        record.seed = 3  # type: ignore[misc]


def test_round_trip_preserves_content(tmp_path) -> None:
    record = make_record(
        variant="ablation-a",
        train_loss_trajectory=[1.0, 0.5, 0.25],
        secondary_metrics={"val_mse": 0.3},
    )
    path = save_record(record, tmp_path)
    assert load_record(path) == record


def test_loader_rejects_foreign_schema_version(tmp_path) -> None:
    path = tmp_path / "foreign.json"
    payload = json.loads(make_record().model_dump_json())
    payload["schema_version"] = RESULT_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_record(path)


def test_filename_separates_variants_and_seeds() -> None:
    first = make_record(variant="dense", seed=0).filename()
    second = make_record(variant="sparse", seed=0).filename()
    third = make_record(variant="dense", seed=1).filename()
    assert len({first, second, third}) == 3


def test_filename_separates_splits_that_share_a_dataset_label(tmp_path) -> None:
    """Regression: a scaled task variant must not overwrite the original's records.

    The TTT track ran the same task at two sizes under one dataset label. Because the
    filename keyed only on (architecture, variant, dataset, seed), the second run
    silently replaced three seeds of the first, leaving groups with missing seeds and
    two different datasets merged under one name.
    """

    small = make_record(dataset="rebinding", dataset_fingerprint="aaaaaaaaaaaaaaaa")
    large = make_record(dataset="rebinding", dataset_fingerprint="bbbbbbbbbbbbbbbb")
    assert small.filename() != large.filename()

    save_record(small, tmp_path)
    save_record(large, tmp_path)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_iter_records_reads_every_file(tmp_path) -> None:
    for seed in range(3):
        save_record(make_record(seed=seed), tmp_path)
    assert sorted(record.seed for record in iter_records(tmp_path)) == [0, 1, 2]


def test_fingerprint_is_order_independent_but_content_sensitive() -> None:
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_tensor_fingerprint_detects_content_and_shape_changes() -> None:
    base = torch.arange(12, dtype=torch.float32)
    assert tensor_fingerprint(base) == tensor_fingerprint(base.clone())
    assert tensor_fingerprint(base) != tensor_fingerprint(base.reshape(3, 4))
    changed = base.clone()
    changed[0] += 1.0
    assert tensor_fingerprint(base) != tensor_fingerprint(changed)


def test_config_fingerprint_tracks_configuration() -> None:
    assert make_record(config={"lr": 1e-3}).config_fingerprint != (
        make_record(config={"lr": 1e-2}).config_fingerprint
    )


def test_rows_and_markdown_table() -> None:
    rows = records_to_rows([make_record(secondary_metrics={"val": 0.1})])
    assert rows[0]["secondary.val"] == pytest.approx(0.1)
    table = format_markdown_table(rows, ["architecture", "seed", "value"])
    assert table.startswith("| architecture | seed | value |")
    assert format_markdown_table([], ["a"]) == "_No records._"
