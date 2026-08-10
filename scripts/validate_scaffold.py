"""Validate that every registered track has a package, config, and prompt."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Allow running from a plain checkout, without an editable install.
sys.path.insert(0, str(ROOT / "src"))

from modern_nn_lab.registry import TRACKS  # noqa: E402  (path bootstrap must run first)

PROMPT_BY_KEY = {
    "kan": "01_kan.md",
    "xlstm": "02_xlstm.md",
    "mamba3": "03_mamba3.md",
    "ttt": "04_ttt.md",
    "titans": "05_titans.md",
    "hope": "06_nested_learning_hope.md",
    "pfn": "07_tabpfn.md",
    "relational": "08_relational_foundation.md",
    "moe": "09_sparse_moe.md",
    "flow": "10_flow_matching.md",
    "jepa": "11_jepa.md",
}

CONFIG_BY_KEY = {
    "kan": "01_kan.yaml",
    "xlstm": "02_xlstm.yaml",
    "mamba3": "03_mamba3.yaml",
    "ttt": "04_ttt.yaml",
    "titans": "05_titans.yaml",
    "hope": "06_nested_learning_hope.yaml",
    "pfn": "07_tabpfn.yaml",
    "relational": "08_relational_foundation.yaml",
    "moe": "09_sparse_moe.yaml",
    "flow": "10_flow_matching.yaml",
    "jepa": "11_jepa.yaml",
}


REPORT_BY_KEY = {key: f"{key}.md" for key in PROMPT_BY_KEY}


def main() -> None:
    """Raise ``FileNotFoundError`` when required scaffold files are absent.

    A track marked ``complete`` must additionally carry its specification README and its
    report, so the registry cannot claim more than the repository contains.
    """

    missing: list[str] = []
    for track in TRACKS:
        package = ROOT / "src" / "modern_nn_lab" / "tracks" / track.key
        required = [
            package / "__init__.py",
            ROOT / "prompts" / PROMPT_BY_KEY[track.key],
            ROOT / "configs" / "tracks" / CONFIG_BY_KEY[track.key],
        ]
        if track.status == "complete":
            required += [
                package / "README.md",
                ROOT / "reports" / REPORT_BY_KEY[track.key],
                ROOT / "results" / track.key,
            ]
        missing.extend(str(path.relative_to(ROOT)) for path in required if not path.exists())

    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing scaffold files:\n{joined}")

    print(f"Scaffold OK: {len(TRACKS)} tracks validated.")


if __name__ == "__main__":
    main()
