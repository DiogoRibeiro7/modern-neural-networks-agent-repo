r"""Collapse metrics and linear probes — all independent of any training objective.

This is the part of the track that decides whether anything else means anything. A
latent-prediction loss is *trivially minimized by a constant encoder*: if
:math:`f(x) = k` for every input, the predictor need only output :math:`k` and the loss is
zero. So a low training loss is not evidence of a good representation, and might be evidence
of the opposite.

Nothing here looks at the loss. Three questions are asked of the representation directly:

**Has it collapsed?** :func:`representation_variance` and :func:`effective_rank`. **Neither
is sufficient alone, and the reason is not obvious.**

Variance misses *dimensional* collapse: a representation can have healthy per-dimension
standard deviation while every sample lies on a single line, and only the rank sees that.

Effective rank misses *total* collapse, which is the failure this track actually produces.
When an encoder becomes constant, what remains is floating-point noise, and that noise is
**isotropic** — so the covariance has roughly equal eigenvalues in every direction and the
effective rank comes out *high*. A fully collapsed representation in this repository's own
measurements reports a normalized effective rank of about 0.75 while its standard deviation
is 0.001. Reading the rank alone would have called that healthy.

So the two are reported together everywhere, and the ordering is: standard deviation first,
because near-zero variance means the rank is describing noise rather than structure.

.. math::

    \operatorname{erank}(C) = \exp\!\Big(-\sum_i p_i \log p_i\Big),
    \qquad p_i = \frac{\lambda_i}{\sum_j \lambda_j}

with :math:`\lambda_i` the eigenvalues of the representation covariance. It equals the
dimension when variance is spread evenly and falls to one under total collapse, so it is
directly comparable across representations of different widths once divided by that width.

**Does it contain the content?** :func:`linear_probe`, fitted in closed form so the number
is a property of the representation and not of a probe's optimizer or its seed.

**Has it discarded the nuisance?** The same probe pointed at the nuisance factors, where a
*low* score is the good outcome. Reporting only the content probe would let a representation
that memorizes everything look identical to one that learned what matters.
"""

from __future__ import annotations

import torch
from torch import Tensor


def representation_variance(representations: Tensor) -> float:
    """Return the mean per-dimension standard deviation.

    Args:
        representations: Shape ``(N, d)``.

    Returns:
        Mean standard deviation across dimensions. Near zero means collapse.
    """

    return float(representations.std(dim=0).mean())


def effective_rank(representations: Tensor) -> float:
    """Return the entropy-based effective rank of the representation covariance.

    Args:
        representations: Shape ``(N, d)``.

    Returns:
        A value in ``[1, d]``: the width when variance is spread evenly across directions,
        and one when every sample lies on a single direction.
    """

    centred = representations - representations.mean(dim=0, keepdim=True)
    covariance = centred.T @ centred / max(centred.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)

    total = eigenvalues.sum()
    if float(total) <= 1e-12:
        return 1.0

    proportions = eigenvalues / total
    positive = proportions[proportions > 1e-12]
    entropy = -(positive * positive.log()).sum()
    return float(entropy.exp())


def normalized_effective_rank(representations: Tensor) -> float:
    """Return the effective rank as a fraction of the representation width.

    Args:
        representations: Shape ``(N, d)``.

    Returns:
        A value in ``(0, 1]``, comparable across representations of different widths.
    """

    width = int(representations.shape[-1])
    return effective_rank(representations) / max(width, 1)


def linear_probe(
    train_features: Tensor,
    train_targets: Tensor,
    test_features: Tensor,
    test_targets: Tensor,
    *,
    ridge: float = 1e-3,
) -> float:
    """Fit a ridge regression on the representation and return held-out :math:`R^2`.

    Solved in closed form rather than by gradient descent, so the number depends only on
    the representation — no probe learning rate, no probe seed, nothing to tune into a
    better-looking result.

    Args:
        train_features: Shape ``(N, d)`` representations.
        train_targets: Shape ``(N, k)`` factors to recover.
        test_features: Shape ``(M, d)`` held-out representations.
        test_targets: Shape ``(M, k)`` held-out factors.
        ridge: Regularization strength, needed because a collapsed representation gives a
            singular normal-equation matrix.

    Returns:
        Coefficient of determination on the held-out split, clamped below at zero. A
        representation carrying no information about the target scores zero.
    """

    ones = torch.ones((train_features.shape[0], 1), dtype=train_features.dtype)
    design = torch.cat([train_features, ones], dim=-1)

    gram = design.T @ design
    penalty = ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    penalty[-1, -1] = 0.0  # never penalize the intercept
    weights = torch.linalg.solve(gram + penalty, design.T @ train_targets)

    test_ones = torch.ones((test_features.shape[0], 1), dtype=test_features.dtype)
    predictions = torch.cat([test_features, test_ones], dim=-1) @ weights

    residual = (test_targets - predictions).pow(2).sum()
    total = (test_targets - test_targets.mean(dim=0, keepdim=True)).pow(2).sum()
    if float(total) <= 1e-12:
        return 0.0
    return float(max(1.0 - residual / total, torch.zeros(())))


def collapse_report(representations: Tensor) -> dict[str, float]:
    """Summarize collapse for the experiment record.

    Args:
        representations: Shape ``(N, d)``.

    Returns:
        Variance, effective rank, and normalized effective rank.
    """

    return {
        "representation_std": representation_variance(representations),
        "effective_rank": effective_rank(representations),
        "normalized_effective_rank": normalized_effective_rank(representations),
    }
