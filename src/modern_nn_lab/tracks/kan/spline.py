r"""B-spline machinery for Kolmogorov-Arnold layers.

Everything mathematical about the edge functions lives here, separated from the module
plumbing in :mod:`~modern_nn_lab.tracks.kan.layers` so the recursion can be tested on
hand-computable cases.

Notation follows ``docs/mathematical_notation.md`` with the additions used in the KAN
literature:

- ``G`` — number of grid intervals;
- ``k`` — spline order (degree of the piecewise polynomial);
- ``P = G + k`` — number of basis functions per edge.
"""

from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-8
"""Floor for knot spacings. Repeated knots would otherwise divide by zero."""

MIN_KNOT_SPACING = 1e-3
"""Smallest gap allowed between adjacent knots of an adaptive grid.

``EPS`` keeps the recursion from dividing by zero, but a grid whose cells are that
narrow is numerically useless: the basis becomes a set of spikes and the least-squares
refit is hopelessly ill conditioned. This floor keeps a degenerate feature's grid usable
instead of merely finite.
"""


def build_grid(
    in_features: int,
    *,
    grid_size: int,
    spline_order: int,
    grid_range: tuple[float, float] = (-1.0, 1.0),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build a uniform, ``k``-extended knot vector for every input dimension.

    The interval ``[a, b]`` is divided into ``G`` equal cells. The knot vector is
    extended by ``k`` cells on each side so that the recursion is well defined for every
    basis function whose support touches the domain.

    Args:
        in_features: Number of input dimensions; each gets its own knot vector.
        grid_size: Number of interior grid intervals ``G``. Must be positive.
        spline_order: Spline degree ``k``. Must be non-negative.
        grid_range: Domain ``(a, b)`` with ``a < b``.
        device: Device of the returned tensor.
        dtype: Dtype of the returned tensor.

    Returns:
        Knot tensor of shape ``(in_features, G + 2k + 1)``.

    Raises:
        ValueError: If any argument is outside its permitted range.
    """

    if in_features <= 0:
        raise ValueError("in_features must be positive")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if spline_order < 0:
        raise ValueError("spline_order must be non-negative")

    low, high = grid_range
    if not low < high:
        raise ValueError(f"grid_range must satisfy low < high, got {grid_range}")

    step = (high - low) / grid_size
    knots = torch.arange(-spline_order, grid_size + spline_order + 1, device=device, dtype=dtype)
    knots = knots * step + low
    return knots.unsqueeze(0).expand(in_features, -1).contiguous()


def b_spline_basis(inputs: Tensor, grid: Tensor, *, spline_order: int) -> Tensor:
    r"""Evaluate every B-spline basis function at ``inputs`` by the Cox-de Boor recursion.

    Order 0 is the indicator of a half-open knot cell:

    .. math:: B_{i,0}(x) = \mathbb{1}[t_i \le x < t_{i+1}]

    and the recursion for order ``k`` is

    .. math::

        B_{i,k}(x) = \frac{x - t_i}{t_{i+k} - t_i} B_{i,k-1}(x)
                   + \frac{t_{i+k+1} - x}{t_{i+k+1} - t_{i+1}} B_{i+1,k-1}(x).

    Args:
        inputs: Shape ``(B, in_features)``.
        grid: Knot tensor of shape ``(in_features, G + 2k + 1)`` from :func:`build_grid`.
        spline_order: Spline degree ``k``, matching the grid extension.

    Returns:
        Tensor of shape ``(B, in_features, G + k)``.

    Raises:
        ValueError: If shapes are inconsistent with ``spline_order``.

    Note:
        Because order 0 uses a half-open cell, an input exactly equal to the upper
        domain bound falls outside every basis function and evaluates to zero. Clamp
        inputs into the domain before calling if that matters; see the track README.
    """

    if inputs.ndim != 2:
        raise ValueError(f"inputs must have shape (B, in_features), got {tuple(inputs.shape)}")
    if grid.ndim != 2:
        raise ValueError(f"grid must have shape (in_features, knots), got {tuple(grid.shape)}")
    if inputs.shape[1] != grid.shape[0]:
        raise ValueError(f"inputs has {inputs.shape[1]} features but grid has {grid.shape[0]} rows")
    if grid.shape[1] < 2 * spline_order + 2:
        raise ValueError(f"grid has {grid.shape[1]} knots, too few for spline_order={spline_order}")

    grid = grid.to(inputs.dtype)
    values = inputs.unsqueeze(-1)  # (B, in, 1)

    # Order 0: indicator of each knot cell. Shape (B, in, G + 2k).
    bases = ((values >= grid[:, :-1]) & (values < grid[:, 1:])).to(inputs.dtype)

    for order in range(1, spline_order + 1):
        left_span = (grid[:, order:-1] - grid[:, : -(order + 1)]).clamp_min(EPS)
        right_span = (grid[:, order + 1 :] - grid[:, 1:-order]).clamp_min(EPS)
        left = (values - grid[:, : -(order + 1)]) / left_span * bases[:, :, :-1]
        right = (grid[:, order + 1 :] - values) / right_span * bases[:, :, 1:]
        bases = left + right

    return bases.contiguous()


def curve_to_coefficients(
    inputs: Tensor, outputs: Tensor, grid: Tensor, *, spline_order: int
) -> Tensor:
    """Least-squares fit spline coefficients reproducing ``outputs`` at ``inputs``.

    Used both for initialization and for refitting after a grid update, so that changing
    the grid does not change the function the layer currently represents.

    Args:
        inputs: Shape ``(B, in_features)``.
        outputs: Shape ``(B, in_features, out_features)``; the value each edge function
            should take at the corresponding input.
        grid: Knot tensor of shape ``(in_features, G + 2k + 1)``.
        spline_order: Spline degree ``k``.

    Returns:
        Coefficients of shape ``(out_features, in_features, G + k)``.

    Raises:
        ValueError: If ``inputs`` and ``outputs`` disagree on batch or feature count.
    """

    if outputs.ndim != 3:
        raise ValueError(
            f"outputs must have shape (B, in_features, out_features), got {tuple(outputs.shape)}"
        )
    if inputs.shape[0] != outputs.shape[0] or inputs.shape[1] != outputs.shape[1]:
        raise ValueError("inputs and outputs must agree on batch size and input features")

    basis = b_spline_basis(inputs, grid, spline_order=spline_order)  # (B, in, P)
    # Solve one independent least-squares problem per input dimension.
    design = basis.transpose(0, 1)  # (in, B, P)
    targets = outputs.transpose(0, 1)  # (in, B, out)
    solution: Tensor = torch.linalg.lstsq(design, targets).solution  # (in, P, out)
    return solution.permute(2, 0, 1).contiguous()  # (out, in, P)


def adaptive_grid(
    inputs: Tensor,
    *,
    grid_size: int,
    spline_order: int,
    margin: float = 0.01,
    uniform_mixture: float = 0.02,
) -> Tensor:
    """Recompute knots from the empirical distribution of ``inputs``.

    Grid points are placed at equally spaced sample quantiles, so cells are dense where
    the data is dense. A small uniform component is blended in, because a purely
    quantile-based grid collapses to repeated knots when a feature is nearly constant.

    Args:
        inputs: Shape ``(B, in_features)`` activations reaching this layer.
        grid_size: Number of interior intervals ``G``.
        spline_order: Spline degree ``k``.
        margin: Fraction of the observed range added beyond each extreme knot.
        uniform_mixture: Weight in ``[0, 1]`` of the uniform grid blended into the
            adaptive grid. ``0`` is purely adaptive, ``1`` purely uniform.

    Returns:
        Knot tensor of shape ``(in_features, G + 2k + 1)``.

    Raises:
        ValueError: If ``inputs`` is not two-dimensional, is empty, or the mixture
            weight is outside ``[0, 1]``.
    """

    if inputs.ndim != 2:
        raise ValueError(f"inputs must have shape (B, in_features), got {tuple(inputs.shape)}")
    if inputs.shape[0] == 0:
        raise ValueError("inputs must contain at least one sample")
    if not 0.0 <= uniform_mixture <= 1.0:
        raise ValueError("uniform_mixture must lie in [0, 1]")

    batch = int(inputs.shape[0])
    sorted_inputs = torch.sort(inputs, dim=0).values  # (B, in)

    positions = torch.linspace(0, batch - 1, grid_size + 1, device=inputs.device)
    indices = positions.round().long().clamp(0, batch - 1)
    adaptive = sorted_inputs[indices].transpose(0, 1)  # (in, G + 1)

    low = sorted_inputs[0]
    high = sorted_inputs[-1]
    span = (high - low).clamp_min(EPS)
    low = low - margin * span
    high = high + margin * span
    steps = torch.linspace(0.0, 1.0, grid_size + 1, device=inputs.device)
    uniform = low.unsqueeze(-1) + steps * (high - low).unsqueeze(-1)  # (in, G + 1)

    interior = uniform_mixture * uniform + (1.0 - uniform_mixture) * adaptive

    # Enforce *strictly* increasing knots. Sorting alone is not enough: a constant or
    # near-constant feature produces repeated quantiles, and repeated knots collapse the
    # Cox-de Boor denominators. Each knot is therefore pushed to at least
    # `min_step` above its predecessor.
    interior = torch.cummax(interior, dim=-1).values
    min_step = ((high - low) / grid_size).clamp_min(MIN_KNOT_SPACING)  # (in,)
    offsets = torch.arange(grid_size + 1, device=inputs.device, dtype=inputs.dtype)
    interior = torch.maximum(interior, interior[:, :1] + offsets * min_step.unsqueeze(-1))

    step = ((interior[:, -1] - interior[:, 0]) / grid_size).clamp_min(MIN_KNOT_SPACING)
    step = step.unsqueeze(-1)

    left_offsets = torch.arange(spline_order, 0, -1, device=inputs.device, dtype=inputs.dtype)
    right_offsets = torch.arange(1, spline_order + 1, device=inputs.device, dtype=inputs.dtype)
    left = interior[:, :1] - left_offsets * step
    right = interior[:, -1:] + right_offsets * step
    return torch.cat([left, interior, right], dim=-1).to(inputs.dtype).contiguous()
