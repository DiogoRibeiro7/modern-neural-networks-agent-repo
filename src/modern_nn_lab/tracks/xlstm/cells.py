r"""sLSTM and mLSTM cells with exponential gating and explicit normalization.

The mechanism under study is **exponential gating with a normalizer state**. A
conventional LSTM squashes its input gate through a sigmoid, so the gate cannot exceed
one and a late token can never dominate the accumulated cell state. Replacing the sigmoid
with an exponential removes that ceiling, at the cost of an unbounded state — which is
why a second recurrence tracking the total gate mass is required, and why a running
maximum is needed to keep the exponentials representable.

Notation follows ``docs/mathematical_notation.md``; the equation-to-code mapping and the
deviations from the primary source are in this package's ``README.md``.

**sLSTM** (scalar memory, per unit):

.. math::

    c_t = f_t c_{t-1} + i_t z_t, \qquad
    n_t = f_t n_{t-1} + i_t, \qquad
    h_t = o_t \frac{c_t}{n_t}

**mLSTM** (matrix memory, per head):

.. math::

    C_t = f_t C_{t-1} + i_t v_t k_t^\top, \qquad
    n_t = f_t n_{t-1} + i_t k_t, \qquad
    h_t = o_t \odot \frac{C_t q_t}{\max(|n_t^\top q_t|, 1)}

**Stabilization.** Both cells carry a running maximum

.. math:: m_t = \max(\log f_t + m_{t-1},\ \log i_t)

and use :math:`i'_t = \exp(\log i_t - m_t)` and
:math:`f'_t = \exp(\log f_t + m_{t-1} - m_t)`. This rescales numerator and denominator by
the same factor, so :math:`h_t` is mathematically unchanged while every exponential stays
bounded by one. Without it the exponential input gate overflows within a few dozen steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

GateKind = Literal["exponential", "sigmoid"]
"""``"sigmoid"`` selects the conventional gate and is the ablation of the mechanism."""


@dataclass(frozen=True, slots=True)
class SLSTMState:
    """Recurrent state of :class:`SLSTMCell`.

    Attributes:
        cell: Shape ``(B, H)`` cell state ``c``.
        normalizer: Shape ``(B, H)`` gate-mass state ``n``.
        hidden: Shape ``(B, H)`` previous output ``h``, used for memory mixing.
        stabilizer: Shape ``(B, H)`` running log-maximum ``m``.
    """

    cell: Tensor
    normalizer: Tensor
    hidden: Tensor
    stabilizer: Tensor


@dataclass(frozen=True, slots=True)
class MLSTMState:
    """Recurrent state of :class:`MLSTMCell`.

    Attributes:
        memory: Shape ``(B, heads, d_head, d_head)`` matrix memory ``C``.
        normalizer: Shape ``(B, heads, d_head)`` gate-mass state ``n``.
        stabilizer: Shape ``(B, heads)`` running log-maximum ``m``.
    """

    memory: Tensor
    normalizer: Tensor
    stabilizer: Tensor


def _log_gate(pre_activation: Tensor, kind: GateKind) -> Tensor:
    """Return the gate value in log space.

    Working in log space is what makes the stabilizer possible: both gate kinds are
    reduced to a log value, so the same running-maximum machinery applies to each and the
    ablation changes only this function.

    Args:
        pre_activation: Raw gate pre-activation.
        kind: ``"exponential"`` for ``log exp(a) = a``; ``"sigmoid"`` for
            ``log sigmoid(a) = -softplus(-a)``, which is computed directly rather than as
            ``log(sigmoid(a))`` to avoid underflow for very negative ``a``.

    Returns:
        Elementwise log-gate values.

    Raises:
        ValueError: If ``kind`` is unknown.
    """

    if kind == "exponential":
        return pre_activation
    if kind == "sigmoid":
        return -torch.nn.functional.softplus(-pre_activation)
    raise ValueError(f"unknown gate kind {kind!r}")


class SLSTMCell(nn.Module):
    """One sLSTM step: scalar memory, exponential gating, memory mixing.

    "Memory mixing" means the gates and the candidate see the previous hidden state, so
    the recurrence cannot be unrolled in parallel across time. That is a real cost of the
    sLSTM branch and is reported as such.

    Shapes:
        - input: ``(B, D)``
        - state tensors: ``(B, H)``
        - output: ``(B, H)``

    Attributes:
        input_size: Input width ``D``.
        hidden_size: Hidden width ``H``.
        input_gate: Gate kind for ``i``.
        forget_gate: Gate kind for ``f``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        input_gate: GateKind = "exponential",
        forget_gate: GateKind = "sigmoid",
    ) -> None:
        """Create an sLSTM cell.

        Args:
            input_size: Input width.
            hidden_size: Hidden width.
            input_gate: Gate kind for the input gate. ``"sigmoid"`` is the ablation.
            forget_gate: Gate kind for the forget gate. The primary source permits either;
                sigmoid is the default because an exponential forget gate makes the state
                grow without bound on long sequences.

        Raises:
            ValueError: If a width is not positive.
        """

        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_gate = input_gate
        self.forget_gate = forget_gate

        # One projection producing all four pre-activations: z, i, f, o.
        self.input_projection = nn.Linear(input_size, 4 * hidden_size)
        self.recurrent_projection = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights uniformly by fan-in, and bias the forget gate positive.

        A positive forget bias makes the cell retentive at initialization, which is the
        standard remedy for a recurrent model that otherwise forgets before it has learned
        to remember.
        """

        bound = 1.0 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)
        with torch.no_grad():
            forget_slice = slice(2 * self.hidden_size, 3 * self.hidden_size)
            self.input_projection.bias[forget_slice] = 1.0

    def initial_state(self, batch_size: int, *, device: torch.device | str = "cpu") -> SLSTMState:
        """Return the zero state.

        ``normalizer`` starts at one rather than zero: at ``t = 0`` the hidden state is
        ``c_0 / n_0``, and a zero denominator would make the first step undefined.

        Args:
            batch_size: Batch size ``B``.
            device: Device for the state tensors.

        Returns:
            A zeroed :class:`SLSTMState`.
        """

        shape = (batch_size, self.hidden_size)
        return SLSTMState(
            cell=torch.zeros(shape, device=device),
            normalizer=torch.ones(shape, device=device),
            hidden=torch.zeros(shape, device=device),
            stabilizer=torch.zeros(shape, device=device),
        )

    def forward(self, inputs: Tensor, state: SLSTMState) -> tuple[Tensor, SLSTMState]:
        """Advance the recurrence by one step.

        Args:
            inputs: Shape ``(B, D)``.
            state: Previous state.

        Returns:
            ``(hidden, new_state)`` with ``hidden`` of shape ``(B, H)``.

        Raises:
            ValueError: If ``inputs`` has the wrong rank or width.
        """

        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError(f"expected shape (B, {self.input_size}), got {tuple(inputs.shape)}")

        pre = self.input_projection(inputs) + self.recurrent_projection(state.hidden)
        candidate, input_pre, forget_pre, output_pre = pre.chunk(4, dim=-1)

        log_i = _log_gate(input_pre, self.input_gate)
        log_f = _log_gate(forget_pre, self.forget_gate)

        # Running maximum keeps both exponentials at most one.
        stabilizer = torch.maximum(log_f + state.stabilizer, log_i)
        gate_i = torch.exp(log_i - stabilizer)
        gate_f = torch.exp(log_f + state.stabilizer - stabilizer)

        cell = gate_f * state.cell + gate_i * torch.tanh(candidate)
        normalizer = gate_f * state.normalizer + gate_i
        # The floor guards the transient where every gate mass is negligible.
        hidden = torch.sigmoid(output_pre) * (cell / normalizer.clamp_min(1e-6))

        return hidden, SLSTMState(
            cell=cell, normalizer=normalizer, hidden=hidden, stabilizer=stabilizer
        )

    def extra_repr(self) -> str:
        """Return a compact description for ``print(model)``."""

        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"input_gate={self.input_gate}, forget_gate={self.forget_gate}"
        )


class MLSTMCell(nn.Module):
    """One mLSTM step: matrix memory, exponential gating, no memory mixing.

    Because the gates depend only on the current input, the recurrence has no
    hidden-to-hidden path and could in principle be evaluated in parallel across time.
    This implementation still steps sequentially — the point here is transparency, and
    any throughput claim would be about the kernel rather than the mechanism.

    Shapes:
        - input: ``(B, D)``
        - memory: ``(B, heads, d_head, d_head)``
        - output: ``(B, D)``

    Attributes:
        input_size: Input width ``D``, which must be divisible by ``heads``.
        heads: Number of memory heads.
        head_dim: Width of one head.
    """

    def __init__(
        self,
        input_size: int,
        *,
        heads: int = 2,
        input_gate: GateKind = "exponential",
        forget_gate: GateKind = "sigmoid",
    ) -> None:
        """Create an mLSTM cell.

        Args:
            input_size: Input and output width.
            heads: Number of memory heads.
            input_gate: Gate kind for the input gate. ``"sigmoid"`` is the ablation.
            forget_gate: Gate kind for the forget gate.

        Raises:
            ValueError: If the width is not positive or not divisible by ``heads``.
        """

        super().__init__()
        if input_size <= 0 or heads <= 0:
            raise ValueError("input_size and heads must be positive")
        if input_size % heads != 0:
            raise ValueError(f"input_size {input_size} must be divisible by heads {heads}")

        self.input_size = input_size
        self.heads = heads
        self.head_dim = input_size // heads
        self.input_gate = input_gate
        self.forget_gate = forget_gate

        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.output_gate = nn.Linear(input_size, input_size)
        self.gates = nn.Linear(input_size, 2 * heads)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights uniformly by fan-in and bias the forget gate positive."""

        bound = 1.0 / math.sqrt(self.input_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)
        with torch.no_grad():
            self.gates.bias[self.heads :] = 1.0

    def initial_state(self, batch_size: int, *, device: torch.device | str = "cpu") -> MLSTMState:
        """Return the zero state.

        Args:
            batch_size: Batch size ``B``.
            device: Device for the state tensors.

        Returns:
            A zeroed :class:`MLSTMState`.
        """

        return MLSTMState(
            memory=torch.zeros(
                (batch_size, self.heads, self.head_dim, self.head_dim), device=device
            ),
            normalizer=torch.zeros((batch_size, self.heads, self.head_dim), device=device),
            stabilizer=torch.zeros((batch_size, self.heads), device=device),
        )

    def forward(self, inputs: Tensor, state: MLSTMState) -> tuple[Tensor, MLSTMState]:
        """Advance the recurrence by one step.

        Args:
            inputs: Shape ``(B, D)``.
            state: Previous state.

        Returns:
            ``(hidden, new_state)`` with ``hidden`` of shape ``(B, D)``.

        Raises:
            ValueError: If ``inputs`` has the wrong rank or width.
        """

        if inputs.ndim != 2 or inputs.shape[1] != self.input_size:
            raise ValueError(f"expected shape (B, {self.input_size}), got {tuple(inputs.shape)}")

        batch = inputs.shape[0]
        shape = (batch, self.heads, self.head_dim)
        query = self.query(inputs).view(shape)
        # Scaling the key by 1/sqrt(d) keeps the outer-product update's magnitude
        # independent of head width, exactly as in scaled dot-product attention.
        key = self.key(inputs).view(shape) / math.sqrt(self.head_dim)
        value = self.value(inputs).view(shape)

        input_pre, forget_pre = self.gates(inputs).chunk(2, dim=-1)  # (B, heads) each
        log_i = _log_gate(input_pre, self.input_gate)
        log_f = _log_gate(forget_pre, self.forget_gate)

        stabilizer = torch.maximum(log_f + state.stabilizer, log_i)
        gate_i = torch.exp(log_i - stabilizer).unsqueeze(-1)  # (B, heads, 1)
        gate_f = torch.exp(log_f + state.stabilizer - stabilizer).unsqueeze(-1)

        memory = gate_f.unsqueeze(-1) * state.memory + gate_i.unsqueeze(-1) * (
            value.unsqueeze(-1) * key.unsqueeze(-2)
        )
        normalizer = gate_f * state.normalizer + gate_i * key

        retrieved = torch.einsum("bhij,bhj->bhi", memory, query)
        overlap = torch.einsum("bhi,bhi->bh", normalizer, query).abs()
        # The source's covering rule is max(|n^T q|, 1) on the *unstabilized* state. The
        # stored state here is scaled by exp(-m_t), and a constant floor is not
        # scale-invariant, so the floor must be scaled the same way. Writing the true
        # quantities as n = n~ * exp(m) and C = C~ * exp(m):
        #
        #   h = (C~ q e^m) / max(|n~^T q| e^m, 1) = C~ q / max(|n~^T q|, e^-m)
        #
        # Flooring at a bare 1.0 instead silently changes the function; the hand-computed
        # two-step test in tests/test_xlstm.py is what catches that.
        denominator = torch.maximum(overlap, torch.exp(-stabilizer))
        hidden = retrieved / denominator.unsqueeze(-1)

        output = torch.sigmoid(self.output_gate(inputs)) * hidden.reshape(batch, self.input_size)
        return output, MLSTMState(memory=memory, normalizer=normalizer, stabilizer=stabilizer)

    def extra_repr(self) -> str:
        """Return a compact description for ``print(model)``."""

        return (
            f"input_size={self.input_size}, heads={self.heads}, "
            f"input_gate={self.input_gate}, forget_gate={self.forget_gate}"
        )
