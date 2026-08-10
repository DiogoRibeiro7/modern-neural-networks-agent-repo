"""Adapter for the official TabPFN checkpoint — deliverable B of this track.

This is **not** the from-scratch PFN in :mod:`modern_nn_lab.tracks.pfn.model`. The two are
kept in separate modules because the track prompt is explicit that they must not be
conflated: one is a mechanism built here from first principles, the other is a pre-trained
artefact whose training data and compute are not ours and are not accounted for in any
comparison we could run.

**Status in this repository: not executed.** TabPFN 8.2.0 gates its checkpoint download
behind an interactive browser license acceptance. In a non-interactive environment that
flow cannot complete, and circumventing a license gate is not an option. The adapter below
is therefore written, documented, and left ready — but no TabPFN number appears anywhere
in this track's report, and nothing here is compared against one.

If you have accepted the license locally, install the optional extra and the adapter will
work:

```bash
poetry install --extras tabpfn
```

## The comparison rule that applies if you do run it

TabPFN is pre-trained on millions of synthetic datasets. A from-scratch baseline fitted on
a few hundred rows is not its peer, and reporting the two side by side without saying so is
exactly the "pre-trained versus from-scratch" violation in
``docs/benchmark_protocol.md``. :func:`pretraining_advantage_note` returns the text that
must accompany any such table, and :func:`build_tabpfn_classifier` attaches it to the
estimator so it travels with the record.
"""

from __future__ import annotations

from typing import Any

TABPFN_IMPORT_ERROR = (
    "The official TabPFN package is not installed, or its checkpoint has not been "
    "downloaded. Install the optional extra with `poetry install --extras tabpfn`. Note "
    "that TabPFN 8.x requires interactive acceptance of its model license before the "
    "checkpoint can be downloaded, which cannot be completed in a non-interactive "
    "environment."
)


def tabpfn_available() -> bool:
    """Return whether the official TabPFN package can be imported.

    Importing is necessary but not sufficient: the checkpoint may still be missing or
    license-gated, which only surfaces on first use.

    Returns:
        ``True`` if ``import tabpfn`` succeeds.
    """

    try:
        import tabpfn  # noqa: F401
    except ImportError:
        return False
    return True


def pretraining_advantage_note() -> str:
    """Return the disclosure that must accompany any TabPFN comparison.

    Returns:
        Text stating the advantage in plain terms.
    """

    return (
        "TabPFN is a pre-trained checkpoint. It arrives having been fitted to a large "
        "synthetic prior at a compute cost that is not incurred by, or accounted for in, "
        "any baseline it is compared against here. A from-scratch model fitted on a few "
        "hundred rows is not its peer on equal terms, and any table containing both must "
        "state this. Neither the pre-training data nor its compute is measured by this "
        "repository."
    )


def build_tabpfn_classifier(**kwargs: Any) -> Any:
    """Construct an official ``TabPFNClassifier``.

    Args:
        **kwargs: Passed through to the classifier.

    Returns:
        A fitted-on-``fit`` scikit-learn-style classifier.

    Raises:
        ImportError: If the package or its checkpoint is unavailable, with instructions.
    """

    try:
        from tabpfn import TabPFNClassifier
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(TABPFN_IMPORT_ERROR) from error

    classifier = TabPFNClassifier(**kwargs)
    # Travels with the estimator so the disclosure reaches the record, not just the docs.
    classifier.pretraining_advantage_note = pretraining_advantage_note()
    return classifier
