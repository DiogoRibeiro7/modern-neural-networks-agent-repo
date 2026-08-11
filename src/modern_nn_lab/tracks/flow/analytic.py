r"""Closed forms for the Gaussian-to-Gaussian case, which is why this track can be audited.

The acceptance criterion is to separate **vector-field approximation error** from **ODE
discretization error**. Those two are hopelessly confounded if the only observable is
sample quality: a bad sample set could mean the network learned the wrong field, or that
the solver took too few steps, and no amount of staring at a scatter plot distinguishes
them.

When both endpoints are isotropic Gaussians, the *marginal* velocity field has a closed
form. That gives a ground truth against which each error can be measured on its own:

- integrate the **exact** field and the only error left is the solver's;
- compare the **learned** field to the exact one pointwise and the only error left is the
  network's.

## The marginal field

With :math:`x_0 \sim \mathcal{N}(0, I)`, :math:`x_1 \sim \mathcal{N}(m, s^2 I)`
independent, and an affine path :math:`x_t = \alpha_t x_1 + \sigma_t x_0`, the triple
:math:`(x_0, x_1, x_t)` is jointly Gaussian. The marginal velocity is the conditional
expectation of the conditional velocity,

.. math::

    u_t(x) = \mathbb{E}\!\left[\dot\alpha_t x_1 + \dot\sigma_t x_0 \;\middle|\; x_t = x\right]

and for jointly Gaussian variables that expectation is affine in :math:`x`:

.. math::

    \mu_t = \alpha_t m, \qquad
    \tau_t^2 = \alpha_t^2 s^2 + \sigma_t^2, \qquad
    c_t = \dot\alpha_t \alpha_t s^2 + \dot\sigma_t \sigma_t

.. math::

    u_t(x) = \dot\alpha_t m + \frac{c_t}{\tau_t^2}\,(x - \mu_t)

Both terms matter: the first moves the mean, the second contracts or expands around it.
:func:`marginal_velocity` implements exactly this, and it is verified two independent ways
rather than asserted: :func:`projection_residual` checks the defining orthogonality property
of a conditional expectation, and ``tests/test_flow.py`` integrates the field to confirm it
actually transports the source onto the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from modern_nn_lab.tracks.flow.paths import ProbabilityPath


@dataclass(frozen=True, slots=True)
class GaussianEndpoints:
    """An isotropic Gaussian target, with a standard-normal source.

    Attributes:
        mean: Target mean, applied to every coordinate.
        scale: Target standard deviation, applied to every coordinate.
    """

    mean: float = 2.0
    scale: float = 0.5

    def __post_init__(self) -> None:
        """Validate the endpoints.

        Raises:
            ValueError: If the scale is not positive.
        """

        if self.scale <= 0:
            raise ValueError("scale must be positive")

    def sample_target(self, n: int, dim: int, generator: torch.Generator) -> Tensor:
        """Draw samples from the target distribution.

        Args:
            n: Number of samples.
            dim: Dimensionality.
            generator: Seeded generator.

        Returns:
            Shape ``(n, dim)``.
        """

        return self.mean + self.scale * torch.randn((n, dim), generator=generator)

    def marginal_moments(self, path: ProbabilityPath, t: Tensor) -> tuple[Tensor, Tensor]:
        """Return the mean and standard deviation of the marginal at time ``t``.

        Args:
            path: The probability path.
            t: Shape ``(N, 1)`` times.

        Returns:
            ``(mean, std)``, each of shape ``(N, 1)``.
        """

        alpha = path.alpha(t)
        sigma = path.sigma(t)
        mean = alpha * self.mean
        variance = alpha.pow(2) * self.scale**2 + sigma.pow(2)
        return mean, variance.sqrt()


def marginal_velocity(
    x: Tensor, t: Tensor, path: ProbabilityPath, endpoints: GaussianEndpoints
) -> Tensor:
    """Return the exact marginal velocity field at ``(x, t)``.

    Args:
        x: Shape ``(N, d)`` points.
        t: Shape ``(N,)`` or ``(N, 1)`` times in ``[0, 1]``.
        path: The probability path.
        endpoints: The Gaussian endpoints.

    Returns:
        Shape ``(N, d)`` velocities.
    """

    time = t.unsqueeze(-1) if t.ndim == 1 else t
    alpha = path.alpha(time)
    sigma = path.sigma(time)
    alpha_dot = path.alpha_dot(time)
    sigma_dot = path.sigma_dot(time)

    mean = alpha * endpoints.mean
    variance = alpha.pow(2) * endpoints.scale**2 + sigma.pow(2)
    covariance = alpha_dot * alpha * endpoints.scale**2 + sigma_dot * sigma

    return alpha_dot * endpoints.mean + (covariance / variance) * (x - mean)


def projection_residual(
    path: ProbabilityPath,
    endpoints: GaussianEndpoints,
    *,
    n_samples: int,
    generator: torch.Generator,
    dim: int = 2,
) -> dict[str, float]:
    r"""Check :func:`marginal_velocity` against the definition of conditional expectation.

    The marginal field is *defined* as :math:`u_t(x) = \mathbb{E}[u \mid x_t = x, t]`, and a
    conditional expectation is characterized by its residual being orthogonal to every
    function of what was conditioned on:

    .. math::

        \mathbb{E}\big[(u - u_t(x_t))\, f(x_t, t)\big] = 0 \quad \text{for all } f

    So the closed form can be verified by plain Monte Carlo over the *joint* distribution,
    with no kernel and no bandwidth — each statistic below is an unbiased estimate of zero
    whose error falls as :math:`1/\sqrt{N}`.

    (An earlier version of this check estimated the conditional expectation directly with a
    narrow Gaussian kernel. That was a mistake: at any bandwidth wide enough to have samples
    in it, the estimator is biased by more than the quantity being verified, and it reported
    a large disagreement against a field that is in fact correct. Testing the defining
    property avoids estimating the conditional expectation at all.)

    Args:
        path: The probability path.
        endpoints: The Gaussian endpoints.
        n_samples: Pairs to draw.
        generator: Seeded generator.
        dim: Dimensionality.

    Returns:
        One statistic per test function; every entry should be near zero.
    """

    source = torch.randn((n_samples, dim), generator=generator)
    target = endpoints.sample_target(n_samples, dim, generator)
    times = torch.rand((n_samples, 1), generator=generator)

    interpolated = path.interpolate(source, target, times)
    residual = path.conditional_velocity(source, target, times) - marginal_velocity(
        interpolated, times, path, endpoints
    )

    test_functions = {
        "constant": torch.ones_like(times),
        "linear": interpolated,
        "quadratic": interpolated.pow(2),
        "time": times,
    }
    return {
        name: float((residual * probe).mean(dim=0).abs().max())
        for name, probe in test_functions.items()
    }
