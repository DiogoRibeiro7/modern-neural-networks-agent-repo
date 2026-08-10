"""Generate report tables from saved records.

Reports are prose plus *generated* tables. Prose is written by hand; every number is
regenerated from ``results/`` into marked blocks, so a report can never drift away from
the evidence it cites.

A marked block looks like:

```markdown
<!-- generated:comparison -->
...table, overwritten on every regeneration...
<!-- /generated:comparison -->
```
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from modern_nn_lab.experiments.evaluation import aggregate_values
from modern_nn_lab.experiments.records import ExperimentRecord

GroupKey = tuple[str, ...]
KeyFn = Callable[[ExperimentRecord], GroupKey]


def group_records(
    records: Iterable[ExperimentRecord], key: KeyFn
) -> dict[GroupKey, list[ExperimentRecord]]:
    """Group records by an arbitrary key.

    Args:
        records: Records to group.
        key: Maps a record to a tuple used as the group identity.

    Returns:
        Mapping from key tuple to the records sharing it, in insertion order.
    """

    grouped: dict[GroupKey, list[ExperimentRecord]] = defaultdict(list)
    for record in records:
        grouped[key(record)].append(record)
    return dict(grouped)


def aggregate_table(
    records: Iterable[ExperimentRecord],
    *,
    key: KeyFn,
    key_columns: Sequence[str],
    include_cost: bool = True,
    decimals: int = 4,
) -> str:
    """Render a Markdown table of across-seed aggregates.

    Groups that contain a non-successful run are shown with their status instead of a
    mean, so a divergence can never be hidden behind an average.

    Args:
        records: Records to summarize.
        key: Group identity, typically architecture and variant.
        key_columns: Column headers for the components of the key.
        include_cost: Add parameter count, activated parameters, and training seconds.
        decimals: Digits after the decimal point for metric values.

    Returns:
        A Markdown table, or a placeholder when there are no records.
    """

    grouped = group_records(records, key)
    if not grouped:
        return "_No records._"

    headers = [*key_columns, "metric", "mean ± std", "95% CI", "seeds"]
    if include_cost:
        headers += ["params", "activated", "train s"]

    rows: list[str] = []
    for group_key in sorted(grouped):
        group = grouped[group_key]
        failures = [record for record in group if record.status != "success"]
        metric_name = group[0].primary_metric.name

        if failures:
            statuses = ", ".join(sorted({record.status for record in failures}))
            cells = [
                *group_key,
                metric_name,
                f"**{statuses}** ({len(failures)}/{len(group)} seeds)",
                "-",
                str(len(group)),
            ]
            if include_cost:
                cells += [str(group[0].parameter_count), "-", "-"]
            rows.append("| " + " | ".join(cells) + " |")
            continue

        aggregate = aggregate_values(
            metric_name,
            [record.primary_metric.value for record in group],
            higher_is_better=group[0].primary_metric.higher_is_better,
        )
        arrow = "↑" if aggregate.higher_is_better else "↓"
        cells = [
            *group_key,
            f"{metric_name} {arrow}",
            f"{aggregate.mean:.{decimals}g} ± {aggregate.std:.2g}",
            f"[{aggregate.ci_low:.{decimals}g}, {aggregate.ci_high:.{decimals}g}]",
            str(aggregate.n_seeds),
        ]
        if include_cost:
            activated = group[0].activated_parameter_count
            mean_time = sum(record.train_wall_clock_s for record in group) / len(group)
            cells += [
                str(group[0].parameter_count),
                str(activated) if activated is not None else "-",
                f"{mean_time:.2f}",
            ]
        rows.append("| " + " | ".join(cells) + " |")

    header = "| " + " | ".join(headers) + " |"
    divider = "|" + "|".join("---" for _ in headers) + "|"
    return "\n".join([header, divider, *rows])


def inject(path: Path | str, blocks: dict[str, str]) -> list[str]:
    """Replace generated blocks in a Markdown file in place.

    Args:
        path: Report file containing ``<!-- generated:NAME -->`` markers.
        blocks: Mapping from block name to replacement Markdown.

    Returns:
        Names of the blocks that were found and replaced, in file order.

    Raises:
        FileNotFoundError: If the report does not exist.
        KeyError: If a supplied block name has no marker in the file.
    """

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"report not found: {target}")

    text = target.read_text(encoding="utf-8")
    replaced: list[str] = []

    for name, content in blocks.items():
        pattern = re.compile(
            rf"(<!-- generated:{re.escape(name)} -->)(.*?)(<!-- /generated:{re.escape(name)} -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            raise KeyError(f"{target}: no marker for generated block {name!r}")

        # A function replacement, not a template: table content may contain characters
        # that `re.sub` would otherwise interpret as group references. `content` is bound
        # as a default so the closure does not capture the loop variable.
        def replace(match: re.Match[str], body: str = content) -> str:
            return f"{match.group(1)}\n{body}\n{match.group(3)}"

        text = pattern.sub(replace, text)
        replaced.append(name)

    target.write_text(text, encoding="utf-8")
    return replaced
