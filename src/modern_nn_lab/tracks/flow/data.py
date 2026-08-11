"""Two-dimensional target distributions, and a sample-based distance to score them.

Three targets, chosen because each fails a different way:

``gaussian``
    An isotropic Gaussian. The only case with a closed-form marginal velocity field, and
    therefore the only case where approximation error and discretization error can be
    separated exactly. Everything quantitative in this track's analysis rests on it.
``moons``
    Two interleaved crescents. Curved, thin, and not linearly separable — a field that is
    too smooth will bridge the gap between the crescents and put mass where there is none.
``mixture``
    Well-separated Gaussian modes. Tests whether the flow reaches every mode; a model that
    drops one still produces individually plausible samples, so this failure is invisible
    without a distributional distance.

Sample quality is scored with the **energy distance**, which is zero if and only if the two
distributions match, requires no bandwidth or bin choice that could be tuned to flatter a
result, and is not the training objective — scoring a model with its own loss would measure
optimization rather than fit.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

Dataset = Literal["gaussian", "moons", "mixture"]
"""The three target distributions."""

DATASETS: tuple[Dataset, ...] = ("gaussian", "moons", "mixture")


def sample_source(n: int, dim: int, generator: torch.Generator) -> Tensor:
    """Draw from the source distribution, a standard normal.

    Args:
        n: Number of samples.
        dim: Dimensionality.
        generator: Seeded generator.

    Returns:
        Shape ``(n, dim)``.
    """

    return torch.randn((n, dim), generator=generator)


def sample_target(
    dataset: Dataset,
    n: int,
    generator: torch.Generator,
    *,
    mean: float = 2.0,
    scale: float = 0.5,
) -> Tensor:
    """Draw from one of the target distributions.

    Args:
        dataset: Which target.
        n: Number of samples.
        generator: Seeded generator.
        mean: Mean of the ``gaussian`` target, applied to every coordinate.
        scale: Standard deviation of the ``gaussian`` target.

    Returns:
        Shape ``(n, 2)``.

    Raises:
        ValueError: If the dataset name is unknown.
    """

    if dataset == "gaussian":
        return mean + scale * torch.randn((n, 2), generator=generator)

    if dataset == "moons":
        half = n // 2
        angle_top = math.pi * torch.rand((half,), generator=generator)
        angle_bottom = math.pi * torch.rand((n - half,), generator=generator)
        top = torch.stack([torch.cos(angle_top), torch.sin(angle_top)], dim=-1)
        bottom = torch.stack([1.0 - torch.cos(angle_bottom), 0.5 - torch.sin(angle_bottom)], dim=-1)
        points = torch.cat([top, bottom], dim=0)
        noise = 0.08 * torch.randn((n, 2), generator=generator)
        # Centred and scaled so every target sits on a comparable scale; otherwise the
        # energy distances would not be comparable across datasets.
        return 2.0 * (points + noise - torch.tensor([0.5, 0.25]))

    if dataset == "mixture":
        centres = torch.tensor([[2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]])
        which = torch.randint(0, centres.shape[0], (n,), generator=generator)
        return centres[which] + 0.35 * torch.randn((n, 2), generator=generator)

    raise ValueError(f"unknown dataset {dataset!r}; available: {list(DATASETS)}")


def energy_distance(x: Tensor, y: Tensor) -> float:
    r"""Return the energy distance between two sample sets.

    .. math::

        D^2(X, Y) = 2\\,\\mathbb{E}\\|X - Y\\| - \\mathbb{E}\\|X - X'\\| - \\mathbb{E}\\|Y - Y'\\|

    It is zero exactly when the distributions agree, and has no free parameter to tune.

    Args:
        x: Shape ``(N, d)``.
        y: Shape ``(M, d)``.

    Returns:
        The distance. Clamped at zero, since the estimator can go slightly negative when
        the two sets are drawn from the same distribution.
    """

    cross = torch.cdist(x, y).mean()
    within_x = torch.cdist(x, x).mean()
    within_y = torch.cdist(y, y).mean()
    return float(max(2.0 * cross - within_x - within_y, torch.zeros(())))


def mode_coverage(samples: Tensor, centres: Tensor, radius: float = 1.0) -> float:
    """Return the fraction of mixture modes that received samples.

    Energy distance is a single number and a dropped mode is only one of several ways to
    earn a poor one. This says specifically whether the flow reached everywhere it should.

    Args:
        samples: Shape ``(N, d)`` generated samples.
        centres: Shape ``(K, d)`` mode locations.
        radius: How close a sample must be to count as covering a mode.

    Returns:
        Fraction of modes with at least one sample within ``radius``.
    """

    distances = torch.cdist(samples, centres)
    covered = (distances.min(dim=0).values < radius).float()
    return float(covered.mean())


MIXTURE_CENTRES = torch.tensor([[2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]])
"""Mode locations of the ``mixture`` target, for :func:`mode_coverage`."""
