r"""ODE integration, with the discretization error made visible rather than hidden.

Sampling from a flow model means solving

.. math:: \frac{dx}{dt} = v_\theta(x, t), \qquad x(0) \sim p_{\text{source}}

from :math:`t = 0` to :math:`t = 1`. Two solvers are provided, and the difference between
them is the whole point of offering both:

**Euler** — one field evaluation per step, local error :math:`O(h^2)`, global error
:math:`O(h)`.

**Midpoint (RK2)** — two evaluations per step, global error :math:`O(h^2)`.

Reporting *steps* alone would therefore be misleading, since a midpoint step costs two
field evaluations. :class:`Trajectory` records ``n_evaluations`` so a comparison can be made
at equal cost rather than at equal step count, which is the honest axis when one method is
twice as expensive per step.

The order claims are not decoration: ``tests/test_flow.py`` measures the empirical
convergence order of each solver on a field with a known exact solution and asserts it
matches, so a solver that was silently first-order would fail rather than merely
underperform.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

VelocityFn = Callable[[Tensor, Tensor], Tensor]
"""Maps ``(x, t)`` with shapes ``(N, d)`` and ``(N, 1)`` to a velocity of shape ``(N, d)``."""

Method = Literal["euler", "midpoint"]
"""Supported integration methods."""

EVALUATIONS_PER_STEP: dict[str, int] = {"euler": 1, "midpoint": 2}
"""Field evaluations each method costs per step, for equal-cost comparisons."""


@dataclass(frozen=True, slots=True)
class Trajectory:
    """The result of integrating an ODE.

    Attributes:
        final: Shape ``(N, d)`` state at the end time.
        states: Shape ``(steps + 1, N, d)`` saved states, when requested; otherwise empty.
        times: Shape ``(steps + 1,)`` times aligned with ``states``.
        method: Which solver produced this.
        steps: Number of steps taken.
        n_evaluations: Field evaluations performed, the honest cost measure.
    """

    final: Tensor
    states: Tensor
    times: Tensor
    method: str
    steps: int
    n_evaluations: int


def integrate(
    velocity: VelocityFn,
    initial: Tensor,
    *,
    steps: int,
    method: Method = "euler",
    t_start: float = 0.0,
    t_end: float = 1.0,
    save_trajectory: bool = False,
) -> Trajectory:
    """Integrate an ODE with a fixed step size.

    Integrating from ``t_end`` back to ``t_start`` is supported by passing them reversed,
    which is what the reversibility test uses: a solver whose forward and backward passes
    do not compose to the identity, in the small-step limit, has a sign or indexing error.

    Args:
        velocity: The field to integrate.
        initial: Shape ``(N, d)`` starting state.
        steps: Number of fixed-size steps.
        method: ``"euler"`` or ``"midpoint"``.
        t_start: Starting time.
        t_end: Ending time.
        save_trajectory: Keep every intermediate state, for the diagnostics artefact.

    Returns:
        The trajectory.

    Raises:
        ValueError: If ``steps`` is not positive or the method is unknown.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if method not in EVALUATIONS_PER_STEP:
        raise ValueError(f"unknown method {method!r}; available: {sorted(EVALUATIONS_PER_STEP)}")

    state = initial.clone()
    step_size = (t_end - t_start) / steps
    saved = [state.clone()] if save_trajectory else []
    times = [t_start]

    for index in range(steps):
        time = t_start + index * step_size
        column = torch.full((state.shape[0], 1), time, dtype=state.dtype, device=state.device)

        if method == "euler":
            state = state + step_size * velocity(state, column)
        else:
            half = column + 0.5 * step_size
            midpoint = state + 0.5 * step_size * velocity(state, column)
            state = state + step_size * velocity(midpoint, half)

        if save_trajectory:
            saved.append(state.clone())
        times.append(t_start + (index + 1) * step_size)

    return Trajectory(
        final=state,
        states=torch.stack(saved) if saved else torch.empty((0, *initial.shape)),
        times=torch.tensor(times),
        method=method,
        steps=steps,
        n_evaluations=steps * EVALUATIONS_PER_STEP[method],
    )
