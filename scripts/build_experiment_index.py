"""Build the cross-track experiment index with immutable configuration hashes.

The index is a single machine-readable file describing every experiment the repository has
run: which track, which architecture, how many seeds, what it was measured on, and — the
part that makes it an *index* rather than a summary — a hash over the exact configuration
each group of records was produced under.

The hash is what makes a claim checkable later. A record says "this architecture scored
0.92"; the hash says "under precisely this configuration", so a rerun that produces a
different number can be attributed to a changed setting rather than argued about. It is
computed from the record's own stored configuration, not from the source tree, so it stays
valid even as the code around it changes.

Deliberately absent: any aggregate score. The tracks measure different quantities on
different data, and a column that averaged them would be the aggregate leaderboard the
integration prompt forbids.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import (  # noqa: E402
    ARTEFACT_DIRNAME,
    RESULT_SCHEMA_VERSION,
    ExperimentRecord,
    fingerprint,
    iter_records,
)
from modern_nn_lab.registry import TRACKS  # noqa: E402

RESULTS = ROOT / "results"
# Under `artefacts/` because `iter_records` skips that directory. Writing the index beside
# the records would make the next build try to parse the index as a record.
DESTINATION = RESULTS / ARTEFACT_DIRNAME / "experiment_index.json"

# Excluded from the configuration hash because they are *outcomes of a run*, not the
# configuration it ran under. The seed is the replicate index; a learning rate chosen per
# seed by a validation search is a result of that search, and the search itself — the grid,
# and the fact that selection happened on validation — is part of the design and stays in
# the hash. Including these made every seed of a group hash differently, which would have
# made the index useless for its purpose: telling whether two results came from the same
# experiment.
VOLATILE_KEYS = frozenset(
    {
        "seed",
        "learning_rate",
        "selected_learning_rate",
        "learning_rate_validation_accuracy",
        "measured_tokens_per_s",
    }
)


def config_hash(record: ExperimentRecord) -> str:
    """Return a stable hash of the configuration a record was produced under.

    Args:
        record: The record.

    Returns:
        A hex digest over the record's configuration, its dataset identity, and the
        architecture version — excluding measured quantities that vary run to run.
    """

    stable = {key: value for key, value in record.config.items() if key not in VOLATILE_KEYS}
    return fingerprint(
        {
            "track": record.track,
            "architecture": record.architecture,
            "variant": record.variant,
            "architecture_version": record.architecture_version,
            "dataset": record.dataset,
            "dataset_fingerprint": record.dataset_fingerprint,
            "split_strategy": record.split_strategy,
            "config": stable,
        }
    )


def group_key(record: ExperimentRecord) -> tuple[str, str, str, str]:
    """Return the identity of the run group a record belongs to."""

    return (record.track, record.architecture, record.variant or "default", record.dataset)


def summarize_group(records: list[ExperimentRecord]) -> dict[str, Any]:
    """Summarize one group of seeded records.

    Args:
        records: Records sharing a track, architecture, variant, and dataset.

    Returns:
        A JSON-serializable entry.
    """

    first = records[0]
    values = [record.primary_metric.value for record in records]
    hashes = sorted({config_hash(record) for record in records})

    entry: dict[str, Any] = {
        "track": first.track,
        "architecture": first.architecture,
        "variant": first.variant,
        "dataset": first.dataset,
        "dataset_fingerprint": first.dataset_fingerprint,
        "metric": first.primary_metric.name,
        "higher_is_better": first.primary_metric.higher_is_better,
        "seeds": sorted(record.seed for record in records),
        "n_records": len(records),
        "config_hash": hashes[0],
        "parameter_count": first.parameter_count,
        "activated_parameter_count": first.activated_parameter_count,
        "schema_versions": sorted({record.schema_version for record in records}),
        "statuses": sorted({record.status for record in records}),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }
    if len(hashes) > 1:
        # Should never happen: a group is defined by its configuration. Recorded rather
        # than raised so the index still builds and the inconsistency is visible.
        entry["config_hash_conflict"] = hashes
    return entry


def build() -> dict[str, Any]:
    """Build the index from every committed record.

    Returns:
        The index payload.

    Raises:
        FileNotFoundError: If the results directory does not exist.
    """

    if not RESULTS.exists():
        raise FileNotFoundError(f"no results directory at {RESULTS}")

    grouped: dict[tuple[str, str, str, str], list[ExperimentRecord]] = defaultdict(list)
    for record in iter_records(RESULTS):
        grouped[group_key(record)].append(record)

    entries = [summarize_group(records) for _, records in sorted(grouped.items())]

    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_track[entry["track"]].append(entry)

    tracks = []
    for spec in TRACKS:
        track_entries = by_track.get(spec.key, [])
        tracks.append(
            {
                "key": spec.key,
                "name": spec.name,
                "domain": spec.domain,
                "core_question": spec.core_question,
                "claim_level": spec.target_claim,
                "status": spec.status,
                "n_groups": len(track_entries),
                "n_records": sum(entry["n_records"] for entry in track_entries),
                "metrics": sorted({entry["metric"] for entry in track_entries}),
                "config_hashes": sorted({entry["config_hash"] for entry in track_entries}),
            }
        )

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "note": (
            "One entry per (track, architecture, variant, dataset) group. `config_hash` is "
            "computed from the record's own stored configuration, so it remains valid as "
            "the source tree changes. There is deliberately no aggregate score: the tracks "
            "measure different quantities on different data and averaging them would "
            "manufacture a comparison none of these records support."
        ),
        "hash_excludes": sorted(VOLATILE_KEYS),
        "hash_excludes_note": (
            "These keys are outcomes of a run rather than configuration: the seed is the "
            "replicate index, and a learning rate chosen per seed on validation is the "
            "result of a search whose design — the grid, and selection on validation — is "
            "itself hashed. Excluding them is what makes two seeds of one experiment share "
            "a hash."
        ),
        "n_records": sum(entry["n_records"] for entry in entries),
        "n_groups": len(entries),
        "tracks": tracks,
        "groups": entries,
    }


def main() -> int:
    """Write the index and print a summary.

    Returns:
        ``0`` on success.
    """

    payload = build()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")

    print(f"Wrote {DESTINATION.relative_to(ROOT)}")
    print(f"  {payload['n_records']} records in {payload['n_groups']} groups")
    conflicts = [entry for entry in payload["groups"] if "config_hash_conflict" in entry]
    if conflicts:
        print(f"  WARNING: {len(conflicts)} group(s) mix configurations")
        for entry in conflicts:
            print(f"    {entry['track']}/{entry['architecture']}/{entry['dataset']}")
        return 1
    for track in payload["tracks"]:
        print(
            f"  {track['key']:12s} {track['n_records']:4d} records  "
            f"{track['n_groups']:3d} groups  {track['claim_level']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
