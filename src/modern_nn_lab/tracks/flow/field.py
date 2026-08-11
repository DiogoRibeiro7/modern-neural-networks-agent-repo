r"""The velocity-field network, and the flow-matching objective it is trained under.

The network takes a point and a time and predicts a velocity. Time enters through Fourier
features rather than as a raw scalar: a single scalar appended to a two-dimensional input
is easy for an MLP to under-use, and the field genuinely depends on time in a way that
matters near both endpoints.

The training objective is the conditional flow-matching loss

.. math::

    \\mathcal{L}(\\theta) = \\mathbb{E}_{t, x_0, x_1}
    \\left\\| v_\\theta(x_t, t) - u_t(x_t \\mid x_0, x_1) \\right\\|^2

whose minimizer is the *marginal* field even though the target used at each step is a
*conditional* one. That is the identity flow matching rests on, and it is not obvious: the
per-sample regression targets are not the thing being learned, but their conditional
expectation is. This repository does not prove the identity, but it does check its
consequence — that the learned field converges to the analytic marginal field on the
Gaussian case, where that field is known in closed form.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from modern_nn_lab.tracks.flow.config import FlowConfig
from modern_nn_lab.tracks.flow.paths import ProbabilityPath


class TimeEmbedding(nn.Module):
    """Fourier features of the time coordinate.

    Attributes:
        n_frequencies: Number of sinusoidal pairs.
    """

    def __init__(self, n_frequencies: int) -> None:
        """Build the embedding.

        Args:
            n_frequencies: Number of sinusoidal pairs.

        Raises:
            ValueError: If ``n_frequencies`` is not positive.
        """

        super().__init__()
        if n_frequencies <= 0:
            raise ValueError("n_frequencies must be positive")
        self.n_frequencies = n_frequencies
        # Fixed, not learned: the point is to give the network a usable basis, and a
        # learned one would make the time dependence harder to attribute.
        self.register_buffer("frequencies", 2.0 ** torch.arange(n_frequencies).float() * math.pi)

    @property
    def width(self) -> int:
        """Output width of the embedding."""

        return 2 * self.n_frequencies

    def forward(self, t: Tensor) -> Tensor:
        """Embed a batch of times.

        Args:
            t: Shape ``(N, 1)`` times.

        Returns:
            Shape ``(N, 2 * n_frequencies)``.
        """

        # `frequencies` is a registered buffer, which mypy types as Tensor | Module.
        frequencies = self.get_buffer("frequencies")
        scaled = t * frequencies
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)


class VectorField(nn.Module):
    """An MLP predicting a velocity from a point and a time.

    Attributes:
        config: The configuration used to build the network.
    """

    def __init__(self, config: FlowConfig) -> None:
        """Build the network.

        Args:
            config: Track configuration.
        """

        super().__init__()
        self.config = config
        self.time_embedding = TimeEmbedding(config.n_frequencies)

        width = config.d_hidden
        layers: list[nn.Module] = [
            nn.Linear(config.dim + self.time_embedding.width, width),
            nn.SiLU(),
        ]
        for _ in range(config.n_layers - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        layers.append(nn.Linear(width, config.dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Predict the velocity at ``(x, t)``.

        Args:
            x: Shape ``(N, d)`` points.
            t: Shape ``(N, 1)`` or ``(N,)`` times.

        Returns:
            Shape ``(N, d)`` velocities.

        Raises:
            ValueError: If the shapes are inconsistent.
        """

        time = t.unsqueeze(-1) if t.ndim == 1 else t
        if x.ndim != 2 or time.ndim != 2:
            raise ValueError(f"expected (N, d) points and (N, 1) times, got {tuple(x.shape)}")
        if x.shape[0] != time.shape[0]:
            raise ValueError("points and times must agree on batch size")
        if x.shape[-1] != self.config.dim:
            raise ValueError(f"expected {self.config.dim} dimensions, got {x.shape[-1]}")

        features = torch.cat([x, self.time_embedding(time)], dim=-1)
        velocity: Tensor = self.network(features)
        return velocity


def flow_matching_loss(
    field: VectorField,
    path: ProbabilityPath,
    source: Tensor,
    target: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Compute the conditional flow-matching loss for one batch.

    Args:
        field: The network being trained.
        path: The probability path defining the conditional target.
        source: Shape ``(N, d)`` samples from the source distribution.
        target: Shape ``(N, d)`` samples from the target distribution.
        generator: Seeded generator, used to draw the times.

    Returns:
        Scalar loss.
    """

    times = torch.rand((source.shape[0], 1), generator=generator)
    interpolated = path.interpolate(source, target, times)
    conditional = path.conditional_velocity(source, target, times)
    predicted = field(interpolated, times)
    loss: Tensor = (predicted - conditional).pow(2).sum(dim=-1).mean()
    return loss
