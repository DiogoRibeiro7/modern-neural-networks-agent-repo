"""Validate every committed experiment record against the current schema.

Run in CI. A record that cannot be loaded is a reproducibility defect: the figures and
tables derived from it can no longer be regenerated.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Allow running from a plain checkout, without an editable install.
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import (  # noqa: E402  (path bootstrap must run first)
    ARTEFACT_DIRNAME,
    RESULT_SCHEMA_VERSION,
    load_record,
)

RESULTS = ROOT / "results"


def main() -> int:
    """Load every record under ``results/`` and report failures.

    Returns:
        Process exit code: ``0`` when all records validate, ``1`` otherwise.
    """

    if not RESULTS.exists():
        print("No results/ directory; nothing to validate.")
        return 0

    paths = [path for path in sorted(RESULTS.rglob("*.json")) if ARTEFACT_DIRNAME not in path.parts]
    if not paths:
        print("No result records committed yet.")
        return 0

    failures: list[str] = []
    statuses: dict[str, int] = {}
    for path in paths:
        try:
            record = load_record(path)
        # Report every invalid record rather than aborting on the first one.
        except Exception as error:
            failures.append(f"{path.relative_to(ROOT)}: {error}")
            continue
        statuses[record.status] = statuses.get(record.status, 0) + 1

    print(f"Schema version {RESULT_SCHEMA_VERSION}: checked {len(paths)} record(s).")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")

    if failures:
        print("\nInvalid records:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("All records valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
