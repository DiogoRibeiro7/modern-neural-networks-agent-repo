r"""The Kolmogorov-Arnold layer: learnable functions on edges, summation on nodes.

A conventional linear layer computes ``y_j = sum_i w_ji x_i``: the edge carries a scalar.
A KAN layer replaces the scalar with a univariate function,

.. math:: y_j = \sum_i \phi_{ji}(x_i),

and learns ``phi``. Following the primary source, each edge function is a residual
combination of a fixed nonlinearity and a learnable spline:

.. math:: \phi_{ji}(x) = w^{b}_{ji}\, b(x) + w^{s}_{ji}\, \mathrm{spline}_{ji}(x)

with ``spline_{ji}(x) = sum_p c_{jip} B_p(x)``. Deviations from the paper are recorded in
this track's README.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from modern_nn_lab.tracks.kan.spline import (
    adaptive_grid,
    b_spline_basis,
    build_grid,
    curve_to_coefficients,
)

__all__ = ["KANLayer"]


class KANLayer(nn.Module):
    """One Kolmogorov-Arnold layer.

    Shapes:
        - input: ``(B, in_features)``
        - output: ``(B, out_features)``
        - ``base_weight``: ``(out_features, in_features)``
        - ``spline_weight``: ``(out_features, in_features, G + k)``
        - ``grid`` (buffer): ``(in_features, G + 2k + 1)``

    Attributes:
        in_features: Input width.
        out_features: Output width.
        grid_size: Number of grid intervals ``G``.
        spline_order: Spline degree ``k``.
        n_basis: Basis functions per edge, ``G + k``.
    """

    grid: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        base_scale: float = 1.0,
        spline_noise_scale: float = 0.1,
        learnable_spline: bool = True,
        use_base_branch: bool = True,
    ) -> None:
        """Create a KAN layer.

        Args:
            in_features: Input width. Must be positive.
            out_features: Output width. Must be positive.
            grid_size: Number of grid intervals ``G``.
            spline_order: Spline degree ``k``.
            grid_range: Initial domain of the knot vector.
            base_scale: Gain of the residual ``SiLU`` branch.
            spline_noise_scale: Standard deviation of the noise used to initialize the
                spline coefficients. Zero initializes every edge to the pure base branch.
            learnable_spline: When false, spline coefficients are frozen at their
                initial values. This is the *fixed edge function* ablation: the layer
                becomes a random nonlinear feature map with a learned linear readout.
            use_base_branch: When false, the residual ``SiLU`` branch is removed so the
                edge function is a pure spline.

        Raises:
            ValueError: If a width or spline setting is invalid.
        """

        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if spline_order < 1:
            raise ValueError("spline_order must be at least 1")
        if spline_noise_scale < 0:
            raise ValueError("spline_noise_scale must be non-negative")

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.n_basis = grid_size + spline_order
        self.use_base_branch = use_base_branch
        self.base_activation = nn.SiLU()

        self.register_buffer(
            "grid",
            build_grid(
                in_features,
                grid_size=grid_size,
                spline_order=spline_order,
                grid_range=grid_range,
            ),
        )

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, self.n_basis))
        self.reset_parameters(base_scale=base_scale, spline_noise_scale=spline_noise_scale)

        if not learnable_spline:
            self.spline_weight.requires_grad_(False)
        if not use_base_branch:
            self.base_weight.requires_grad_(False)
            with torch.no_grad():
                self.base_weight.zero_()

    def reset_parameters(self, *, base_scale: float = 1.0, spline_noise_scale: float = 0.1) -> None:
        """Initialize both branches deterministically given the ambient RNG state.

        Spline coefficients are drawn directly, scaled by ``1 / sqrt(G + k)`` so that the
        initial edge function has a magnitude independent of the grid resolution. An
        earlier version fitted smooth noise through
        :func:`~modern_nn_lab.tracks.kan.spline.curve_to_coefficients`; that made
        initialization depend on a LAPACK least-squares solve whose reduction order is
        not bitwise stable across thread-pool states, so two identically seeded runs
        could differ in the last bits. Determinism is worth more here than initial
        smoothness, which the small scale already provides.

        Args:
            base_scale: Gain of the residual branch, applied to Kaiming-uniform weights.
            spline_noise_scale: Scale of the initial spline coefficients.
        """

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.base_weight.mul_(base_scale)
            bound = spline_noise_scale / math.sqrt(self.n_basis)
            nn.init.uniform_(self.spline_weight, -bound, bound)

    def basis(self, inputs: Tensor) -> Tensor:
        """Evaluate the B-spline basis at ``inputs``.

        Args:
            inputs: Shape ``(B, in_features)``.

        Returns:
            Shape ``(B, in_features, G + k)``.
        """

        return b_spline_basis(inputs, self.grid, spline_order=self.spline_order)

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply the layer.

        Args:
            inputs: Shape ``(B, in_features)``.

        Returns:
            Shape ``(B, out_features)``.

        Raises:
            ValueError: If ``inputs`` has the wrong rank or width.
        """

        if inputs.ndim != 2:
            raise ValueError(f"expected shape (B, in_features), got {tuple(inputs.shape)}")
        if inputs.shape[1] != self.in_features:
            raise ValueError(f"expected {self.in_features} input features, got {inputs.shape[1]}")

        spline_basis = self.basis(inputs).reshape(inputs.shape[0], -1)  # (B, in * P)
        spline_out = torch.nn.functional.linear(
            spline_basis, self.spline_weight.reshape(self.out_features, -1)
        )
        if not self.use_base_branch:
            return spline_out

        base_out = torch.nn.functional.linear(self.base_activation(inputs), self.base_weight)
        return base_out + spline_out

    @torch.no_grad()
    def edge_functions(self, samples: Tensor) -> Tensor:
        """Evaluate every learned edge function on a shared 1-D sample grid.

        This is what makes a KAN inspectable: each ``phi_ji`` can be plotted directly.
        Figures are produced from the returned values, never from hard-coded numbers.

        Args:
            samples: Shape ``(S,)`` points at which to evaluate every edge.

        Returns:
            Shape ``(out_features, in_features, S)`` values of ``phi_ji(samples)``.

        Raises:
            ValueError: If ``samples`` is not one-dimensional.
        """

        if samples.ndim != 1:
            raise ValueError(f"samples must be 1-D, got {tuple(samples.shape)}")

        repeated = samples.unsqueeze(-1).expand(-1, self.in_features)  # (S, in)
        basis = self.basis(repeated)  # (S, in, P)
        spline = torch.einsum("sip,oip->ois", basis, self.spline_weight)
        if not self.use_base_branch:
            return spline
        base: Tensor = self.base_activation(samples)  # (S,)
        return spline + self.base_weight.unsqueeze(-1) * base

    @torch.no_grad()
    def update_grid(self, inputs: Tensor, *, uniform_mixture: float = 0.02) -> None:
        """Move the knots towards the empirical input distribution, preserving the function.

        The coefficients are refitted on the new knots against the *current* layer
        output, so a grid update is (up to least-squares error) function preserving.
        Without the refit, adapting the grid would silently change the model.

        Args:
            inputs: Shape ``(B, in_features)`` activations reaching this layer.
            uniform_mixture: Blend weight towards a uniform grid; see
                :func:`~modern_nn_lab.tracks.kan.spline.adaptive_grid`.

        Raises:
            ValueError: If ``inputs`` has the wrong rank or width.
        """

        if inputs.ndim != 2 or inputs.shape[1] != self.in_features:
            raise ValueError(f"expected shape (B, {self.in_features}), got {tuple(inputs.shape)}")

        current = torch.einsum(
            "bip,oip->bio", self.basis(inputs), self.spline_weight
        )  # (B, in, out)
        new_grid = adaptive_grid(
            inputs,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            uniform_mixture=uniform_mixture,
        )
        self.grid.copy_(new_grid)
        self.spline_weight.copy_(
            curve_to_coefficients(inputs, current, self.grid, spline_order=self.spline_order)
        )

    def regularization(self) -> Tensor:
        """Return an L1 surrogate encouraging sparse edge functions.

        The primary source regularizes the L1 norm of *activations*. This implementation
        penalizes the mean absolute spline coefficient instead, which is data
        independent and therefore cheaper and reproducible; the substitution is recorded
        as a deviation in the track README.

        Returns:
            Scalar tensor.
        """

        return self.spline_weight.abs().mean()

    def extra_repr(self) -> str:
        """Return a compact description for ``print(model)``."""

        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}, "
            f"base_branch={self.use_base_branch}"
        )
