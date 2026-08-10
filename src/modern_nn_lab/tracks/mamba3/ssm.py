r"""Mamba-3's selective state-space recurrence, implemented from the primary source.

The continuous system is

.. math:: \dot h(t) = A(t) h(t) + B(t) x(t), \qquad y(t) = C(t)^\top h(t)

and everything interesting happens in how it is *discretized* and what structure the
transition is allowed to have. This module implements the three contributions of the
primary source, each behind a flag so that turning it off recovers the prior method
exactly:

**1. Exponential-trapezoidal discretization** (source Proposition 1, equations 5-6).
Approximating the state-input integral with a data-dependent convex combination of both
interval endpoints, rather than only the right endpoint, gives

.. math::

    h_t = \alpha_t h_{t-1} + \beta_t B_{t-1} x_{t-1} + \gamma_t B_t x_t

with :math:`\alpha_t = e^{\Delta_t A_t}`, :math:`\beta_t = (1-\lambda_t)\Delta_t
e^{\Delta_t A_t}`, :math:`\gamma_t = \lambda_t \Delta_t`, and :math:`\lambda_t \in [0,1]`
data dependent. Setting :math:`\lambda_t = 1` kills :math:`\beta_t` and recovers the
exponential-Euler rule used by Mamba-1/2; :math:`\lambda_t = 1/2` is the classical
trapezoid. The extra term is a width-two convolution applied to the *state input* inside
the recurrence.

**2. Complex-valued dynamics as data-dependent rotary embeddings** (Propositions 2-3,
equations 8-10). A complex diagonal SSM of state size ``N/2`` equals a real SSM of state
size ``N`` whose transition is a scalar decay times a block-diagonal of 2x2 rotations
:math:`R(\Delta_t \theta_t[i])`. That in turn equals a *scalar* real SSM in which the
cumulative rotation :math:`\prod_{i \le t} R_i^\top` is applied to :math:`B_t` and
:math:`C_t` — the "RoPE trick". Rotation is what lets the model represent oscillatory
dynamics: a real non-negative transition cannot, which is why real SSMs fail parity while
:math:`h_t = R(\pi x_t) h_{t-1}` solves it exactly.

**3. MIMO** (equations 12-14). Raising the rank of the state update from one to ``R``
replaces the outer product :math:`B_t x_t^\top` with a sum of ``R`` outer products, and
the read-out with a sum of ``R`` projections, leaving the carried state size unchanged.

Deviations from the source are listed in this package's ``README.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

MIN_PAIRED_STATE = 2
"""A rotational state needs an even size, since rotations act on 2-D pairs."""


@dataclass(frozen=True, slots=True)
class SSMState:
    r"""Carried state of :class:`SelectiveSSM`.

    Attributes:
        state: Shape ``(B, heads, N, P)``. The SSM state itself.
        pending: Shape ``(B, heads, N, P)``. The rotated ``B_{t-1} x_{t-1}`` outer product
            from the previous step, which the trapezoidal rule needs and the Euler rule
            does not.
        angle: Shape ``(B, heads, N // 2)``. Cumulative rotation angle, i.e. the phase of
            :math:`\\prod_{i \\le t} R_i`.
    """

    state: Tensor
    pending: Tensor
    angle: Tensor


def rotate_pairs(vectors: Tensor, angle: Tensor) -> Tensor:
    """Apply a block-diagonal 2-D rotation ``R(-angle)`` to paired coordinates.

    The source defines ``R(θ) = [[cos, -sin], [sin, cos]]`` and applies the *transpose*
    of the accumulated rotation, and ``R(θ)ᵀ = R(-θ)``; this function applies that
    transpose directly.

    Args:
        vectors: Shape ``(..., N, R)`` with ``N`` even. Consecutive pairs along the ``N``
            axis form the 2-D coordinates that rotate together.
        angle: Shape ``(..., N // 2)`` rotation angle per pair, broadcast over ``R``.

    Returns:
        Rotated tensor of the same shape as ``vectors``.

    Raises:
        ValueError: If the state size is odd or the angle shape does not match.
    """

    state_size = vectors.shape[-2]
    if state_size % 2 != 0:
        raise ValueError(f"state size must be even to form rotation pairs, got {state_size}")
    if angle.shape[-1] != state_size // 2:
        raise ValueError(
            f"expected {state_size // 2} angles for state size {state_size}, got {angle.shape[-1]}"
        )

    # `reshape` rather than `unflatten`: Torch ships the latter unannotated, and the
    # explicit shape is clearer about the pair layout anyway.
    paired = vectors.reshape(*vectors.shape[:-2], state_size // 2, 2, vectors.shape[-1])
    cos = torch.cos(angle).unsqueeze(-1).unsqueeze(-1)  # (..., N/2, 1, 1)
    sin = torch.sin(angle).unsqueeze(-1).unsqueeze(-1)

    first, second = paired[..., 0, :], paired[..., 1, :]
    rotated_first = cos.squeeze(-2) * first + sin.squeeze(-2) * second
    rotated_second = -sin.squeeze(-2) * first + cos.squeeze(-2) * second
    return torch.stack((rotated_first, rotated_second), dim=-2).flatten(-3, -2)


class SelectiveSSM(nn.Module):
    """One Mamba-3 selective state-space layer, stepped explicitly in time.

    Shapes:
        - input: ``(B, D)`` where ``D = heads * head_dim``
        - output: ``(B, D)``
        - state: ``(B, heads, state_size, head_dim)``

    Attributes:
        d_model: Model width ``D``.
        heads: Number of SSM heads.
        head_dim: Width ``P`` of one head.
        state_size: State size ``N`` per head.
        rank: MIMO rank ``R``. ``1`` is the SISO recurrence.
        trapezoidal: Whether the exponential-trapezoidal rule is active.
        rotary: Whether data-dependent rotations are active.
    """

    def __init__(
        self,
        d_model: int,
        *,
        heads: int = 2,
        state_size: int = 8,
        rank: int = 1,
        trapezoidal: bool = True,
        rotary: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ) -> None:
        """Build the layer.

        Args:
            d_model: Model width, divisible by ``heads``.
            heads: Number of heads.
            state_size: State size ``N`` per head; must be even when ``rotary`` is set.
            rank: MIMO rank ``R``.
            trapezoidal: Enable the exponential-trapezoidal rule. When false the layer
                uses exponential-Euler, i.e. the Mamba-1/2 discretization.
            rotary: Enable data-dependent rotations. When false the transition is a real
                non-negative scalar decay, which cannot represent oscillation.
            dt_min: Lower end of the initial step-size range.
            dt_max: Upper end of the initial step-size range.

        Raises:
            ValueError: If any dimension is invalid.
        """

        super().__init__()
        if d_model <= 0 or heads <= 0 or state_size <= 0 or rank <= 0:
            raise ValueError("d_model, heads, state_size, and rank must be positive")
        if d_model % heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by heads {heads}")
        if rotary and state_size % 2 != 0:
            raise ValueError(f"rotary state_size must be even, got {state_size}")
        if state_size < MIN_PAIRED_STATE:
            raise ValueError(f"state_size must be at least {MIN_PAIRED_STATE}")
        if not 0 < dt_min < dt_max:
            raise ValueError("dt_min and dt_max must satisfy 0 < dt_min < dt_max")

        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.state_size = state_size
        self.rank = rank
        self.trapezoidal = trapezoidal
        self.rotary = rotary

        # Data-dependent SSM parameters. Remark 1 of the source notes that Mamba-3 makes
        # every SSM parameter data dependent, including the state transition A_t.
        self.dt_projection = nn.Linear(d_model, heads)
        self.a_projection = nn.Linear(d_model, heads)
        self.b_projection = nn.Linear(d_model, heads * state_size * rank)
        self.c_projection = nn.Linear(d_model, heads * state_size * rank)
        self.lambda_projection = nn.Linear(d_model, heads) if trapezoidal else None
        self.theta_projection = nn.Linear(d_model, heads * (state_size // 2)) if rotary else None

        # MIMO expands the head input to rank R by an element-wise, *data-independent*
        # scale, which is the source's parameter-efficient instantiation (DP + PR
        # parameters per head rather than DPR).
        self.rank_scale = nn.Parameter(torch.ones(rank, self.head_dim))
        self.skip = nn.Parameter(torch.ones(d_model))
        self.output_projection = nn.Linear(d_model, d_model)

        self._initialize(dt_min, dt_max)

    def _initialize(self, dt_min: float, dt_max: float) -> None:
        """Initialize projections, and bias the step size into ``[dt_min, dt_max]``."""

        for module in (self.b_projection, self.c_projection, self.a_projection):
            nn.init.normal_(module.weight, std=1.0 / math.sqrt(self.d_model))
            nn.init.zeros_(module.bias)
        nn.init.normal_(self.dt_projection.weight, std=1.0 / math.sqrt(self.d_model))

        # softplus(bias) spreads the initial step sizes log-uniformly over the range, the
        # standard Mamba initialization: it sets each head's memory horizon.
        steps = torch.exp(
            torch.rand(self.heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        with torch.no_grad():
            self.dt_projection.bias.copy_(steps + torch.log(-torch.expm1(-steps)))

        if self.lambda_projection is not None:
            nn.init.normal_(self.lambda_projection.weight, std=1.0 / math.sqrt(self.d_model))
            # sigmoid(0) = 1/2, the classical trapezoid, as the starting point.
            nn.init.zeros_(self.lambda_projection.bias)
        if self.theta_projection is not None:
            nn.init.normal_(self.theta_projection.weight, std=1.0 / math.sqrt(self.d_model))
            nn.init.zeros_(self.theta_projection.bias)

    def initial_state(self, batch_size: int, *, device: torch.device | str = "cpu") -> SSMState:
        """Return the zero state.

        Args:
            batch_size: Batch size ``B``.
            device: Device for the state tensors.

        Returns:
            A zeroed :class:`SSMState`.
        """

        shape = (batch_size, self.heads, self.state_size, self.head_dim)
        return SSMState(
            state=torch.zeros(shape, device=device),
            pending=torch.zeros(shape, device=device),
            angle=torch.zeros((batch_size, self.heads, self.state_size // 2), device=device),
        )

    def coefficients(self, inputs: Tensor) -> dict[str, Tensor]:
        """Compute the discretization coefficients for one step.

        Exposed separately so the coefficient formulas can be tested directly against
        Table 1 and Proposition 1 of the source, without running a recurrence.

        Args:
            inputs: Shape ``(B, D)``.

        Returns:
            Mapping with ``delta`` ``(B, heads)``, ``a`` ``(B, heads)``,
            ``alpha`` ``(B, heads)``, ``beta`` ``(B, heads)``, ``gamma`` ``(B, heads)``,
            and ``lam`` ``(B, heads)``.
        """

        delta = torch.nn.functional.softplus(self.dt_projection(inputs))  # (B, heads)
        # A_t < 0 keeps the decay alpha_t = exp(delta_t A_t) inside (0, 1).
        a = -torch.nn.functional.softplus(self.a_projection(inputs))
        alpha = torch.exp(delta * a)

        if self.lambda_projection is None:
            lam = torch.ones_like(delta)
        else:
            lam = torch.sigmoid(self.lambda_projection(inputs))

        gamma = lam * delta
        beta = (1.0 - lam) * delta * alpha
        return {"delta": delta, "a": a, "alpha": alpha, "beta": beta, "gamma": gamma, "lam": lam}

    def step(self, inputs: Tensor, state: SSMState) -> tuple[Tensor, SSMState]:
        """Advance the recurrence by one step.

        Implements source equation (11) — the exponential-trapezoidal recurrence with the
        RoPE trick — generalized to MIMO rank ``R`` by equations (12)-(14).

        Args:
            inputs: Shape ``(B, D)``.
            state: Previous state.

        Returns:
            ``(output, new_state)`` with ``output`` of shape ``(B, D)``.

        Raises:
            ValueError: If ``inputs`` has the wrong rank or width.
        """

        if inputs.ndim != 2 or inputs.shape[1] != self.d_model:
            raise ValueError(f"expected shape (B, {self.d_model}), got {tuple(inputs.shape)}")

        batch = inputs.shape[0]
        coefficients = self.coefficients(inputs)
        delta, alpha = coefficients["delta"], coefficients["alpha"]
        beta, gamma = coefficients["beta"], coefficients["gamma"]

        b_matrix = self.b_projection(inputs).view(batch, self.heads, self.state_size, self.rank)
        c_matrix = self.c_projection(inputs).view(batch, self.heads, self.state_size, self.rank)

        if self.theta_projection is not None:
            # Proposition 2 rotates by Delta_t * theta_t; Proposition 3 accumulates those
            # rotations and applies the product to B and C instead of to the state.
            theta = self.theta_projection(inputs).view(batch, self.heads, self.state_size // 2)
            angle = state.angle + delta.unsqueeze(-1) * theta
            b_matrix = rotate_pairs(b_matrix, angle)
            c_matrix = rotate_pairs(c_matrix, angle)
        else:
            angle = state.angle

        # MIMO: R copies of the head input, each element-wise scaled (equation 12).
        head_inputs = inputs.view(batch, self.heads, self.head_dim)
        expanded = head_inputs.unsqueeze(-1) * self.rank_scale.t()  # (B, heads, P, R)

        # Sum of R outer products, i.e. equation (13) written as one contraction.
        outer = torch.einsum("bhnr,bhpr->bhnp", b_matrix, expanded)

        new_state = alpha[..., None, None] * state.state + gamma[..., None, None] * outer
        if self.trapezoidal:
            # The width-two convolution on the state input: beta_t multiplies the
            # *previous* step's rotated outer product.
            new_state = new_state + beta[..., None, None] * state.pending

        # Equation (14), summed over the R output projections.
        read_out = torch.einsum("bhnr,bhnp->bhp", c_matrix, new_state)
        output = read_out.reshape(batch, self.d_model) + self.skip * inputs

        projected: Tensor = self.output_projection(output)
        return projected, SSMState(state=new_state, pending=outer, angle=angle)

    def forward(self, sequence: Tensor) -> Tensor:
        """Run the recurrence over a whole sequence.

        The scan is sequential and written in Python. That is a transparency choice, not
        a systems one: no throughput number from this module says anything about what a
        fused or chunked kernel would achieve.

        Args:
            sequence: Shape ``(B, T, D)``.

        Returns:
            Shape ``(B, T, D)``.

        Raises:
            ValueError: If ``sequence`` is not three-dimensional.
        """

        if sequence.ndim != 3:
            raise ValueError(f"expected shape (B, T, D), got {tuple(sequence.shape)}")

        state = self.initial_state(sequence.shape[0], device=sequence.device)
        outputs = []
        for index in range(sequence.shape[1]):
            output, state = self.step(sequence[:, index], state)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def extra_repr(self) -> str:
        """Return a compact description for ``print(model)``."""

        return (
            f"d_model={self.d_model}, heads={self.heads}, state_size={self.state_size}, "
            f"rank={self.rank}, trapezoidal={self.trapezoidal}, rotary={self.rotary}"
        )
