"""Run the reproducibility audit and record what it could and could not establish.

The integration prompt asks for an audit "from a clean environment". This runs every check
that is possible here and is explicit about the one that is not: `poetry install` fails on
this Windows host because of path-length limits while unpacking Torch metadata, so a
genuinely fresh dependency resolution could not be exercised. That is recorded as an
unverified step rather than quietly skipped, because "the audit passed" would otherwise mean
less than it appears to.

What *is* verified:

- every committed record parses and validates against the current schema;
- the experiment index rebuilds and its configuration hashes are stable across rebuilds;
- a fresh interpreter can import the package and enumerate every track;
- each track's experiment suite is importable and exposes a ``run`` entry point.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "results" / "artefacts" / "reproducibility_audit.json"


def run_step(name: str, command: list[str], *, required: bool = True) -> dict[str, Any]:
    """Run one audit step and capture its outcome.

    Args:
        name: Human-readable step name.
        command: Command to run.
        required: Whether a failure should fail the audit.

    Returns:
        A JSON-serializable record of the step.
    """

    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    passed = result.returncode == 0
    print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    if not passed:
        print("        " + (result.stderr.strip() or result.stdout.strip())[:300])
    return {
        "step": name,
        "command": " ".join(command),
        "passed": passed,
        "required": required,
        "returncode": result.returncode,
    }


def hash_stability() -> dict[str, Any]:
    """Rebuild the index twice and confirm the configuration hashes do not move.

    A hash that changed between rebuilds would make the index worthless for its purpose.

    Returns:
        A JSON-serializable record of the check.
    """

    index_path = ROOT / "results" / "artefacts" / "experiment_index.json"
    builder = [sys.executable, str(ROOT / "scripts" / "build_experiment_index.py")]

    subprocess.run(builder, cwd=ROOT, capture_output=True, check=False)
    first = json.loads(index_path.read_text(encoding="utf-8"))
    subprocess.run(builder, cwd=ROOT, capture_output=True, check=False)
    second = json.loads(index_path.read_text(encoding="utf-8"))

    first_hashes = {entry["config_hash"] for entry in first["groups"]}
    second_hashes = {entry["config_hash"] for entry in second["groups"]}
    stable = first_hashes == second_hashes

    print(f"  {'PASS' if stable else 'FAIL'}  configuration hashes stable across rebuilds")
    return {
        "step": "configuration hashes stable across rebuilds",
        "passed": stable,
        "required": True,
        "n_hashes": len(first_hashes),
    }


def main() -> int:
    """Run every audit step and write the result.

    Returns:
        ``0`` when every required step passed.
    """

    # Report regeneration is deliberately NOT a step here. It is verified by
    # `verify_reports_regenerate.py`, which the quality gate runs, and folding it in would
    # make this audit circular: writing this file's result changes the synthesis report,
    # which would then fail a cleanliness check performed by the same run.
    print("Reproducibility audit")
    steps: list[dict[str, Any]] = [
        run_step(
            "every record validates against the schema",
            [sys.executable, str(ROOT / "scripts" / "validate_results.py")],
        ),
        run_step(
            "scaffold is complete for every track",
            [sys.executable, str(ROOT / "scripts" / "validate_scaffold.py")],
        ),
        hash_stability(),
        run_step(
            "package imports in a fresh interpreter and lists every track",
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, 'src');"
                " from modern_nn_lab.registry import TRACKS;"
                " assert len(TRACKS) == 11, len(TRACKS);"
                " assert all(t.status == 'complete' for t in TRACKS)",
            ],
        ),
        run_step(
            "every track exposes a runnable experiment suite",
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0, 'src');"
                " from modern_nn_lab.registry import TRACKS;"
                " from modern_nn_lab.experiments.tracks import get_track_suite;"
                " [get_track_suite(t.key) for t in TRACKS]",
            ],
        ),
    ]

    unverified = [
        {
            "step": "clean dependency resolution via `poetry install`",
            "passed": None,
            "required": False,
            "reason": (
                "Fails on this Windows host because of path-length limits while unpacking "
                "Torch metadata. The quality gate is executed with the interpreter's "
                "installed toolchain instead, which runs the identical commands CI runs "
                "through Poetry — but a genuinely fresh resolution was never exercised "
                "here. CI does exercise it on Linux across Python 3.11-3.13."
            ),
        },
        {
            "step": "results reproduced by re-running every suite end to end",
            "passed": None,
            "required": False,
            "reason": (
                "Not attempted. Several suites take 20-30 minutes each on this machine, and "
                "re-running all eleven would take hours. Determinism is exercised per track "
                "by seeded tests instead, which is weaker: it checks that a run repeats, not "
                "that the committed records are what a rerun produces today."
            ),
        },
    ]

    failed = [step for step in steps if step["required"] and not step["passed"]]
    for entry in unverified:
        print(f"  SKIP  {entry['step']}")

    payload = {
        "audit": "reproducibility",
        "n_steps": len(steps),
        "n_failed": len(failed),
        "steps": steps,
        "unverified": unverified,
        "note": (
            "A passing audit means the committed artefacts are internally consistent: "
            "records validate, the scaffold is complete, and configuration hashes are "
            "stable. It does NOT mean the numbers were reproduced from scratch — see "
            "`unverified`. Report regeneration is checked separately by "
            "`scripts/verify_reports_regenerate.py`; including it here would be circular, "
            "since writing this file changes the report that summarizes it."
        ),
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(steps) - len(failed)}/{len(steps)} verified, {len(unverified)} unverified")
    print(f"Wrote {DESTINATION.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
