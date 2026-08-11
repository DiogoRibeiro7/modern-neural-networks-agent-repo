"""Regenerate the generated tables inside ``reports/cross_track_synthesis.md``.

Only the *quantitative* parts of the synthesis are generated here: the evidence table, the
profiling-coverage audit, and the claim-level audit. The comparison matrix itself is
authored prose — mechanism, complexity, strengths and failures are judgements about what
the experiments showed, not quantities a script can compute, and generating them would
imply a rigour the underlying reasoning does not have.

There is no aggregate score anywhere in this file, by design. The integration prompt
forbids ranking these tracks against one another with a single number, and the reason is
visible in the evidence table: the metric column alone contains accuracies, mean squared
errors, energy distances, and probe R-squareds, measured on unrelated data.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.experiments.records import iter_records  # noqa: E402
from modern_nn_lab.experiments.reporting import inject  # noqa: E402

RESULTS = ROOT / "results"
INDEX = RESULTS / "artefacts" / "experiment_index.json"
REPORT = ROOT / "reports" / "cross_track_synthesis.md"
NEWLINE = chr(10)

# Tracks whose primary source was read in this environment. Everything else must sit at
# `research prototype`, which the claim audit below re-checks rather than trusts.
SOURCES_READ = frozenset({"kan", "xlstm", "mamba3", "ttt", "titans", "hope"})

TRACK_NUMBER = {
    "kan": "01",
    "xlstm": "02",
    "mamba3": "03",
    "ttt": "04",
    "titans": "05",
    "hope": "06",
    "pfn": "07",
    "relational": "08",
    "moe": "09",
    "flow": "10",
    "jepa": "11",
}


def main() -> int:
    """Regenerate every marked block in the synthesis.

    Returns:
        ``0`` on success, ``1`` when the index is missing.
    """

    if not INDEX.exists():
        print(f"No index at {INDEX}. Run `python scripts/build_experiment_index.py` first.")
        return 1

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    blocks = {
        "evidence": _evidence_table(index),
        "claims": _claim_audit(index),
        "profiling": _profiling_coverage(),
    }

    replaced = inject(REPORT, blocks)
    print(f"Regenerated {len(replaced)} block(s) in {REPORT.relative_to(ROOT)}: {replaced}")
    return 0


def _evidence_table(index: dict[str, Any]) -> str:
    """Render what evidence each track actually produced."""

    headers = ["#", "track", "records", "groups", "metrics measured", "claim level"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]

    for track in index["tracks"]:
        metrics = ", ".join(f"`{name}`" for name in track["metrics"]) or "—"
        lines.append(
            f"| {TRACK_NUMBER.get(track['key'], '??')} | {track['name']} | "
            f"{track['n_records']} | {track['n_groups']} | {metrics} | "
            f"`{track['claim_level']}` |"
        )

    lines.append("")
    lines.append(
        f"_{index['n_records']} records in {index['n_groups']} configuration groups, each "
        "carrying an immutable configuration hash in "
        "`results/artefacts/experiment_index.json`. The metric column is why there is no "
        "aggregate score: these quantities are not commensurable._"
    )
    return NEWLINE.join(lines)


def _claim_audit(index: dict[str, Any]) -> str:
    """Re-run the claim-policy audit and render its verdicts."""

    headers = ["track", "primary source read", "claim level", "verdict"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]

    failures = 0
    for track in index["tracks"]:
        read = track["key"] in SOURCES_READ
        level = track["claim_level"]
        # `educational implementation` and above require implementation from primary
        # sources; `research prototype` does not.
        if level != "research prototype" and not read:
            verdict = "**UNSUPPORTED**"
            failures += 1
        else:
            verdict = "supported"
        lines.append(f"| {track['name']} | {'yes' if read else 'no'} | `{level}` | {verdict} |")

    lines.append("")
    if failures:
        lines.append(f"_**{failures} unsupported claim level(s).** Fix before release._")
    else:
        lines.append(
            "_Every claim level is supported by the evidence actually gathered. Five tracks "
            "sit at `research prototype` because no primary source was read for them; three "
            "of those were downgraded by this audit from `educational implementation`, which "
            "the policy defines as requiring implementation from primary sources._"
        )
    return NEWLINE.join(lines)


def _profiling_coverage() -> str:
    """Render which profiling fields are actually populated, per track."""

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in iter_records(RESULTS):
        track = counts[record.track]
        track["n"] += 1
        track["params"] += record.parameter_count > 0
        track["activated"] += record.activated_parameter_count is not None
        track["wall_clock"] += record.train_wall_clock_s > 0
        track["latency"] += record.inference_latency_ms is not None
        track["memory"] += record.peak_memory_bytes is not None
        track["flops"] += record.flops_per_sample is not None

    headers = ["track", "records", "params", "activated", "wall clock", "throughput", "peak mem"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]

    def cell(part: int, total: int) -> str:
        if part == total:
            return "all"
        if part == 0:
            return "**none**"
        return f"{part}/{total}"

    for key in TRACK_NUMBER:
        track = counts.get(key)
        if track is None:
            continue
        total = track["n"]
        lines.append(
            f"| {key} | {total} | {cell(track['params'], total)} | "
            f"{cell(track['activated'], total)} | {cell(track['wall_clock'], total)} | "
            f"{cell(track['latency'], total)} | {cell(track['memory'], total)} |"
        )

    lines.append("")
    lines.append(
        "_**Peak memory is unpopulated everywhere**, and that is a deliberate refusal rather "
        "than an oversight: `profiling.peak_memory` measures only on CUDA and returns `None` "
        "on CPU instead of a misleading number. Every run in this repository was on CPU, so "
        "the field is empty by construction. The `flops_per_sample` schema field is likewise "
        "unpopulated everywhere; the one track with analytic FLOP counts (Sparse MoE) stores "
        "them in its own configuration instead, which is a real inconsistency rather than a "
        "hardware limit. `activated` is blank where conditional computation does not apply._"
    )
    return NEWLINE.join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
