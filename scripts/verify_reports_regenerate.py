"""Check that every generated table can be rebuilt from the committed raw results.

The repository's central promise is that no number in any report is typed by hand. That is
easy to assert and easy to violate quietly: a table edited directly, or a report generator
that silently no-ops when its records are missing, would both leave the claim standing while
making it false.

This runs every ``scripts/report_*.py`` and fails if any report changes. A clean run means
the committed reports are exactly what the committed records produce.

It also fails when a report generator reports zero regenerated blocks, which catches the
case where a generator runs successfully but matches nothing — the failure mode a plain
exit-code check would miss.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATORS = sorted(ROOT.glob("scripts/report_*.py"))


def git_diff(paths: list[Path]) -> str:
    """Return the working-tree diff for the given paths.

    Args:
        paths: Files to inspect.

    Returns:
        The unified diff, empty when nothing changed.
    """

    result = subprocess.run(
        ["git", "diff", "--", *[str(path) for path in paths]],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout


def main() -> int:
    """Regenerate every report and verify none of them changed.

    Returns:
        ``0`` when every report is reproducible from the records.
    """

    reports = sorted(ROOT.glob("reports/*.md"))
    before = git_diff(reports)
    if before.strip():
        print("Reports already differ from HEAD; commit or revert before verifying.")
        return 1

    failures: list[str] = []
    for generator in GENERATORS:
        result = subprocess.run(
            [sys.executable, str(generator)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        name = generator.name
        if result.returncode != 0:
            failures.append(f"{name}: exited {result.returncode}\n{result.stderr.strip()}")
            continue

        match = re.search(r"Regenerated (\d+) block", result.stdout)
        if match is None:
            failures.append(f"{name}: produced no 'Regenerated N block(s)' line")
        elif int(match.group(1)) == 0:
            failures.append(f"{name}: regenerated 0 blocks — its markers match nothing")
        else:
            print(f"  {name}: {match.group(1)} block(s)")

    after = git_diff(reports)
    if after.strip():
        changed = sorted(
            {line.split("/")[-1] for line in after.splitlines() if line.startswith("+++")}
        )
        failures.append(
            "reports changed when regenerated, so they are not reproducible from the "
            f"committed records: {changed}"
        )

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"\nAll {len(GENERATORS)} report generators reproduce their reports exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
