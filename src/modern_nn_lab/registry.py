"""Architecture-track registry.

The registry contains metadata only. It deliberately does not import track models,
which keeps CLI inspection lightweight and prevents optional dependencies from
becoming import-time requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrackStatus = Literal["queued", "in_progress", "complete"]
ClaimTarget = Literal[
    "educational implementation",
    "compact reproduction",
    "reference integration",
    "research prototype",
]


@dataclass(frozen=True, slots=True)
class TrackSpec:
    """Static metadata describing one research track."""

    key: str
    name: str
    domain: str
    core_question: str
    target_claim: ClaimTarget
    status: TrackStatus = "queued"


TRACKS: tuple[TrackSpec, ...] = (
    TrackSpec(
        "kan",
        "Kolmogorov-Arnold Networks",
        "general",
        "Must edges be scalar weights?",
        "educational implementation",
        "complete",
    ),
    TrackSpec(
        "xlstm",
        "xLSTM",
        "sequence",
        "Can recurrent memory scale competitively?",
        "educational implementation",
        "complete",
    ),
    TrackSpec(
        "mamba3",
        "Mamba-3 / SSMs",
        "sequence",
        "Do sequence models need attention?",
        "educational implementation",
        "complete",
    ),
    TrackSpec(
        "ttt",
        "Test-Time Training",
        "sequence",
        "Can the hidden state itself learn at inference?",
        "educational implementation",
        "complete",
    ),
    TrackSpec(
        "titans",
        "Titans-style Neural Memory",
        "memory",
        "Can explicit neural memory learn online?",
        "educational implementation",
        "complete",
    ),
    TrackSpec(
        "hope",
        "Nested Learning / Hope",
        "continual learning",
        "Can learning operate across nested timescales?",
        "research prototype",
    ),
    TrackSpec(
        "pfn",
        "Prior-Fitted Networks / TabPFN",
        "tabular",
        "Must each dataset require conventional fitting?",
        "reference integration",
    ),
    TrackSpec(
        "relational",
        "Relational Foundation Models",
        "relational",
        "Can connected tables be modeled natively?",
        "research prototype",
    ),
    TrackSpec(
        "moe",
        "Sparse Mixture of Experts",
        "general",
        "Can conditional computation scale capacity efficiently?",
        "educational implementation",
    ),
    TrackSpec(
        "flow",
        "Flow Matching",
        "generative",
        "Can generation be learned as vector-field regression?",
        "compact reproduction",
    ),
    TrackSpec(
        "jepa",
        "JEPA / World Models",
        "representation",
        "Should models predict representations rather than observations?",
        "research prototype",
    ),
)


def get_track(key: str) -> TrackSpec:
    """Return a track by key.

    Args:
        key: Registry key such as ``"kan"`` or ``"flow"``.

    Raises:
        KeyError: If no track has the requested key.
    """

    for track in TRACKS:
        if track.key == key:
            return track
    available = ", ".join(track.key for track in TRACKS)
    raise KeyError(f"Unknown track {key!r}. Available tracks: {available}")
