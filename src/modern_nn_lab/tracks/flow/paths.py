r"""Probability paths and their conditional vector fields.

A flow-matching model is trained to regress a velocity field. The field it is regressing
towards is *defined* by a probability path — a choice of how a sample from the source
distribution is interpolated towards a sample from the target — and different choices give
genuinely different training problems. This module makes that choice explicit and exact.

Every path here is **affine in the endpoints**:

.. math::

    x_t = \alpha_t\, x_1 + \sigma_t\, x_0

with :math:`x_0` from the source and :math:`x_1` from the target. Differentiating gives the
conditional velocity in closed form, with no approximation anywhere:

.. math::

    u_t(x_t \mid x_0, x_1) = \dot\alpha_t\, x_1 + \dot\sigma_t\, x_0

That is the entire content of the conditional target, and it is why the tests can assert
the vector field exactly rather than approximately. The two paths implemented differ only
in :math:`(\alpha_t, \sigma_t)`:

:class:`LinearPath`
    :math:`\alpha_t = t`, :math:`\sigma_t = 1 - t`, so :math:`u = x_1 - x_0` — a constant
    along each conditional trajectory. This is the optimal-transport-style path: each pair
    travels in a straight line at uniform speed, which is the displacement interpolation
    between two point masses and the shortest path between them under squared cost.
:class:`TrigonometricPath`
    :math:`\alpha_t = \sin(\pi t / 2)`, :math:`\sigma_t = \cos(\pi t / 2)`, so
    :math:`\alpha_t^2 + \sigma_t^2 = 1`. This is the variance-preserving path familiar from
    diffusion models: the conditional trajectory is a quarter-circle rather than a segment,
    and its speed varies with :math:`t`.

The variance-preserving property is worth stating precisely, because it is easy to overclaim:
if the source is standard normal *and* the target is standard normal, the marginal variance
is exactly preserved along the path. For a general target it is not, and no claim is made
that it is.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class ProbabilityPath(ABC):
    """An affine interpolation between a source and a target sample.

    Subclasses supply the four scalar schedules; everything else — sampling a point on the
    path, and the conditional velocity at that point — follows from them.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier recorded with results."""

    @abstractmethod
    def alpha(self, t: Tensor) -> Tensor:
        """Return the coefficient on the target endpoint.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            Coefficients of the same shape.
        """

    @abstractmethod
    def alpha_dot(self, t: Tensor) -> Tensor:
        """Return the time derivative of :meth:`alpha`.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            Derivatives of the same shape.
        """

    @abstractmethod
    def sigma(self, t: Tensor) -> Tensor:
        """Return the coefficient on the source endpoint.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            Coefficients of the same shape.
        """

    @abstractmethod
    def sigma_dot(self, t: Tensor) -> Tensor:
        """Return the time derivative of :meth:`sigma`.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            Derivatives of the same shape.
        """

    def interpolate(self, source: Tensor, target: Tensor, t: Tensor) -> Tensor:
        """Return the point on the conditional path at time ``t``.

        Args:
            source: Shape ``(N, d)`` samples from the source distribution.
            target: Shape ``(N, d)`` samples from the target distribution.
            t: Shape ``(N,)`` or ``(N, 1)`` times in ``[0, 1]``.

        Returns:
            Shape ``(N, d)`` interpolated points.
        """

        time = _column(t)
        return self.alpha(time) * target + self.sigma(time) * source

    def conditional_velocity(self, source: Tensor, target: Tensor, t: Tensor) -> Tensor:
        """Return the conditional target velocity — what the network regresses.

        Args:
            source: Shape ``(N, d)`` samples from the source distribution.
            target: Shape ``(N, d)`` samples from the target distribution.
            t: Shape ``(N,)`` or ``(N, 1)`` times in ``[0, 1]``.

        Returns:
            Shape ``(N, d)`` velocities.
        """

        time = _column(t)
        return self.alpha_dot(time) * target + self.sigma_dot(time) * source

    def endpoints(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((alpha_0, sigma_0), (alpha_1, sigma_1))``.

        A valid path starts at the source and ends at the target, which means
        ``(alpha_0, sigma_0) == (0, 1)`` and ``(alpha_1, sigma_1) == (1, 0)``. The tests
        assert this for every path rather than trusting the schedules.

        Returns:
            The coefficients at ``t = 0`` and ``t = 1``.
        """

        zero = torch.zeros(1)
        one = torch.ones(1)
        return (
            (float(self.alpha(zero)), float(self.sigma(zero))),
            (float(self.alpha(one)), float(self.sigma(one))),
        )


class LinearPath(ProbabilityPath):
    """Straight-line interpolation: the optimal-transport-style path."""

    @property
    def name(self) -> str:
        """Identifier recorded with results."""

        return "linear"

    def alpha(self, t: Tensor) -> Tensor:
        """Return ``t``.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            The same values.
        """

        return t

    def alpha_dot(self, t: Tensor) -> Tensor:
        """Return a constant one.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            Ones of the same shape.
        """

        return torch.ones_like(t)

    def sigma(self, t: Tensor) -> Tensor:
        """Return ``1 - t``.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            The complement.
        """

        return 1.0 - t

    def sigma_dot(self, t: Tensor) -> Tensor:
        """Return a constant minus one.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            Negative ones of the same shape.
        """

        return -torch.ones_like(t)


class TrigonometricPath(ProbabilityPath):
    """Quarter-circle interpolation: the variance-preserving, diffusion-style path."""

    @property
    def name(self) -> str:
        """Identifier recorded with results."""

        return "trigonometric"

    def alpha(self, t: Tensor) -> Tensor:
        """Return ``sin(pi t / 2)``.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            Coefficients of the same shape.
        """

        return torch.sin(0.5 * math.pi * t)

    def alpha_dot(self, t: Tensor) -> Tensor:
        """Return ``(pi / 2) cos(pi t / 2)``.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            Derivatives of the same shape.
        """

        return 0.5 * math.pi * torch.cos(0.5 * math.pi * t)

    def sigma(self, t: Tensor) -> Tensor:
        """Return ``cos(pi t / 2)``.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            Coefficients of the same shape.
        """

        return torch.cos(0.5 * math.pi * t)

    def sigma_dot(self, t: Tensor) -> Tensor:
        """Return ``-(pi / 2) sin(pi t / 2)``.

        Args:
            t: Times in ``[0, 1]``.

        Returns:
            Derivatives of the same shape.
        """

        return -0.5 * math.pi * torch.sin(0.5 * math.pi * t)


PATHS: dict[str, type[ProbabilityPath]] = {
    "linear": LinearPath,
    "trigonometric": TrigonometricPath,
}
"""Registry used by the experiment suite to build paths by name."""


def build_path(name: str) -> ProbabilityPath:
    """Construct a probability path by name.

    Args:
        name: One of ``"linear"`` or ``"trigonometric"``.

    Returns:
        The path.

    Raises:
        KeyError: If the name is unknown.
    """

    if name not in PATHS:
        raise KeyError(f"unknown path {name!r}; available: {sorted(PATHS)}")
    return PATHS[name]()


def _column(t: Tensor) -> Tensor:
    """Reshape a time tensor to broadcast against ``(N, d)`` points.

    Args:
        t: Shape ``(N,)`` or ``(N, 1)``.

    Returns:
        Shape ``(N, 1)``.

    Raises:
        ValueError: If ``t`` has more than two dimensions.
    """

    if t.ndim == 1:
        return t.unsqueeze(-1)
    if t.ndim == 2 and t.shape[-1] == 1:
        return t
    raise ValueError(f"expected times of shape (N,) or (N, 1), got {tuple(t.shape)}")
