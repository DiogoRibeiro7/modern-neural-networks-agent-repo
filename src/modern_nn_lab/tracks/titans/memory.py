r"""Titans-style neural long-term memory: a module that learns to memorize at test time.

The memory is an MLP whose *weights* are the recurrent state. Reading a token writes to
that memory by gradient descent on an associative loss, and retrieval is a plain forward
pass. Transcribed from the primary source:

.. math::

    k_t = x_t W_K, \quad v_t = x_t W_V, \quad q_t = x_t W_Q      \qquad\text{(eq. 11)}

    \ell(M_{t-1}; x_t) = \lVert M_{t-1}(k_t) - v_t \rVert_2^2     \qquad\text{(eq. 12)}

    M_t = (1 - \alpha_t) M_{t-1} + S_t                            \qquad\text{(eq. 13)}

    S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(M_{t-1}; x_t)     \qquad\text{(eq. 14)}

    y_t = M^*_t(q_t)                                              \qquad\text{(eq. 15)}

Three data-dependent gates control it, and each does something a plain gradient step
cannot:

- :math:`\theta_t` — how much of the *momentary* surprise to absorb;
- :math:`\eta_t` — how fast *past* surprise decays. Without it, a run of unsurprising
  tokens after a surprising one drives the gradient to zero and the memory stops
  recording, which is the failure the source's momentum term exists to fix;
- :math:`\alpha_t` — weight decay, i.e. adaptive forgetting. :math:`\alpha_t \to 1`
  clears the memory outright, :math:`\alpha_t \to 0` preserves it.

Everything about this module is instrumented, because this track's acceptance criterion
is explicit memory-write/read diagnostics rather than task accuracy alone. See
:class:`MemoryTrace`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class MemoryState:
    """Recurrent state of :class:`NeuralMemory`.

    Attributes:
        weights: The memory MLP's weight tensors, each batched over ``B``. This *is* the
            memory; it is a hidden state, not a parameter.
        momentum: The past-surprise term ``S_t``, one tensor per weight tensor.
    """

    weights: tuple[Tensor, ...]
    momentum: tuple[Tensor, ...]


@dataclass(slots=True)
class MemoryTrace:
    """Per-token record of what the memory did.

    Task accuracy cannot distinguish a memory that stored the right thing from one that
    got lucky downstream, so every quantity that describes a write or a read is collected
    here and serialized as a diagnostic artefact.

    Attributes:
        loss: ``l(M_{t-1}; x_t)`` before the write — the associative reconstruction error,
            which *is* the surprise signal.
        surprise_norm: ``||grad l||``, the momentary surprise magnitude.
        momentum_norm: ``||S_t||`` after the update.
        write_norm: ``||M_t - M_{t-1}||``, how much the memory actually moved.
        forget_gate: ``alpha_t``, the fraction of memory decayed away this step.
        learning_rate: ``theta_t``.
        momentum_gate: ``eta_t``.
    """

    loss: list[float] = field(default_factory=list)
    surprise_norm: list[float] = field(default_factory=list)
    momentum_norm: list[float] = field(default_factory=list)
    write_norm: list[float] = field(default_factory=list)
    forget_gate: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    momentum_gate: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, list[float]]:
        """Return the trace as JSON-serializable lists."""

        return {
            "loss": self.loss,
            "surprise_norm": self.surprise_norm,
            "momentum_norm": self.momentum_norm,
            "write_norm": self.write_norm,
            "forget_gate": self.forget_gate,
            "learning_rate": self.learning_rate,
            "momentum_gate": self.momentum_gate,
        }


class NeuralMemory(nn.Module):
    """Neural long-term memory with surprise, momentum, and adaptive forgetting.

    Shapes:
        - input: ``(B, T, D)``
        - output: ``(B, T, D)``
        - memory weights: ``(B, d_key, d_hidden)`` and ``(B, d_hidden, d_value)``

    Attributes:
        d_model: Model width.
        d_memory: Width of the key/value/query projections.
        depth: Number of memory MLP layers. The source argues ``>= 2`` is strictly more
            expressive than a matrix-valued memory.
        updates_enabled: When false the memory never writes — the frozen ablation.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_memory: int | None = None,
        d_hidden: int | None = None,
        depth: int = 2,
        updates_enabled: bool = True,
        learning_rate_scale: float = 1.0,
        learning_rate_base: float = 0.1,
        use_momentum: bool = True,
        use_forgetting: bool = True,
    ) -> None:
        """Build the memory module.

        Args:
            d_model: Model width.
            d_memory: Key/value/query width. Defaults to ``d_model // 2``.
            d_hidden: Hidden width of the memory MLP. Defaults to ``2 * d_memory``.
            depth: Memory MLP layers; 1 gives a matrix-valued (linear) memory.
            updates_enabled: Enable writes. ``False`` is the frozen-memory ablation.
            learning_rate_scale: Multiplies ``theta_t``. The update-rate ablation.
            learning_rate_base: Upper bound on ``theta_t``. The source specifies a
                learnable ``theta_t`` but no numeric range; an unbounded one diverges
                here within four tokens. See the README's stability section.
            use_momentum: Keep the past-surprise term. ``False`` reduces eq. 14 to a bare
                gradient step, the source's equation 8.
            use_forgetting: Keep the ``(1 - alpha_t)`` decay. ``False`` makes writes purely
                additive, which is the pre-Titans behaviour the source criticizes.

        Raises:
            ValueError: If a width or depth is invalid.
        """

        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if depth < 1:
            raise ValueError("depth must be at least 1")

        memory_width = d_model // 2 if d_memory is None else d_memory
        hidden_width = 2 * memory_width if d_hidden is None else d_hidden
        if memory_width <= 0 or hidden_width <= 0:
            raise ValueError("d_memory and d_hidden must be positive")
        if learning_rate_scale < 0:
            raise ValueError("learning_rate_scale must be non-negative")

        self.d_model = d_model
        self.d_memory = memory_width
        self.d_hidden = hidden_width
        self.depth = depth
        self.updates_enabled = updates_enabled
        self.learning_rate_scale = learning_rate_scale
        self.learning_rate_base = learning_rate_base
        self.use_momentum = use_momentum
        self.use_forgetting = use_forgetting

        # Outer-loop parameters (eq. 11). These are "hyper-parameters" of the inner loss
        # in the source's terminology: the inner loop optimizes the memory, never these.
        self.key = nn.Linear(d_model, memory_width, bias=False)
        self.value = nn.Linear(d_model, memory_width, bias=False)
        self.query = nn.Linear(d_model, memory_width, bias=False)
        self.output = nn.Linear(memory_width, d_model)

        # Data-dependent gates: theta (learning rate), eta (momentum), alpha (forget).
        self.gate_projection = nn.Linear(d_model, 3)

        widths = [memory_width] + [hidden_width] * (depth - 1) + [memory_width]
        self.initial_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(widths[index], widths[index + 1])) for index in range(depth)]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize outer-loop parameters and the initial memory contents."""

        for module in (self.key, self.value, self.query):
            nn.init.normal_(module.weight, std=1.0 / self.d_model**0.5)
        nn.init.zeros_(self.gate_projection.weight)
        # sigmoid(bias): theta ~ 0.27, eta ~ 0.73, alpha ~ 0.05. A retentive start —
        # decaying quickly before anything has been learned destroys the memory.
        with torch.no_grad():
            self.gate_projection.bias.copy_(torch.tensor([-1.0, 1.0, -3.0]))
        for weight in self.initial_weights:
            nn.init.normal_(weight, std=1.0 / weight.shape[0] ** 0.5)

    def initial_state(self, batch_size: int) -> MemoryState:
        """Return the memory at ``t = 0``, one copy per example.

        Args:
            batch_size: Batch size ``B``.

        Returns:
            A :class:`MemoryState` with zero momentum.
        """

        weights = tuple(
            weight.unsqueeze(0).expand(batch_size, *weight.shape).contiguous()
            for weight in self.initial_weights
        )
        return MemoryState(weights=weights, momentum=tuple(torch.zeros_like(w) for w in weights))

    def read(self, queries: Tensor, weights: Sequence[Tensor]) -> Tensor:
        """Retrieve from the memory: a forward pass with no weight update (eq. 15).

        Args:
            queries: Shape ``(B, d_memory)``.
            weights: Memory weight tensors.

        Returns:
            Shape ``(B, d_memory)``.
        """

        # Unit-normalized queries and keys. The write is an outer product scaled by the
        # key, so an unnormalized key of norm r scales every write by r and the induced
        # loss by r^2 — the term that drives the divergence documented in the README.
        activations: Tensor = torch.nn.functional.normalize(queries, dim=-1)
        for index, weight in enumerate(weights):
            activations = torch.einsum("bij,bi->bj", weight, activations)
            if index < len(weights) - 1:
                activations = torch.nn.functional.silu(activations)
        return activations

    def associative_loss(self, step_input: Tensor, weights: Sequence[Tensor]) -> Tensor:
        """Associative memory loss of source equation (12).

        Args:
            step_input: Shape ``(B, D)``.
            weights: Current memory weights.

        Returns:
            Scalar loss summed over the batch, so each example's gradient is independent.
        """

        predicted = self.read(self.key(step_input), weights)
        target: Tensor = self.value(step_input)
        # Mean over features, sum over the batch: the mean keeps the gradient scale
        # independent of memory width, the sum keeps each example's gradient its own.
        return ((predicted - target) ** 2).mean(dim=-1).sum()

    def gates(self, step_input: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return the data-dependent ``(theta, eta, alpha)`` gates.

        Args:
            step_input: Shape ``(B, D)``.

        Returns:
            Three tensors of shape ``(B, 1)``: learning rate, momentum decay, forget rate.
        """

        raw = torch.sigmoid(self.gate_projection(step_input))
        theta = raw[:, 0:1] * self.learning_rate_base * self.learning_rate_scale
        eta = raw[:, 1:2] if self.use_momentum else torch.zeros_like(raw[:, 1:2])
        alpha = raw[:, 2:3] if self.use_forgetting else torch.zeros_like(raw[:, 2:3])
        return theta, eta, alpha

    def _surprise(self, step_input: Tensor, weights: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """Return ``grad l(M; x_t)``, differentiably and regardless of ambient grad mode."""

        grad_enabled = torch.is_grad_enabled()
        create_graph = grad_enabled and self.training
        with torch.enable_grad():
            if grad_enabled:
                targets = tuple(weights)
            else:
                targets = tuple(w.detach().clone().requires_grad_(True) for w in weights)
            loss = self.associative_loss(step_input, targets)
            return (loss.detach(), *torch.autograd.grad(loss, targets, create_graph=create_graph))

    def step(
        self, step_input: Tensor, state: MemoryState, trace: MemoryTrace | None = None
    ) -> tuple[Tensor, MemoryState]:
        """Write the token into memory, then read from the updated memory.

        Args:
            step_input: Shape ``(B, D)``.
            state: Previous memory state.
            trace: Optional recorder for the write/read diagnostics.

        Returns:
            ``(retrieved, new_state)`` with ``retrieved`` of shape ``(B, D)``.

        Raises:
            ValueError: If ``step_input`` has the wrong rank or width.
        """

        if step_input.ndim != 2 or step_input.shape[1] != self.d_model:
            raise ValueError(f"expected shape (B, {self.d_model}), got {tuple(step_input.shape)}")

        if not self.updates_enabled:
            retrieved = self.read(self.query(step_input), state.weights)
            if trace is not None:
                zero = 0.0
                trace.loss.append(float(self.associative_loss(step_input, state.weights).detach()))
                for series in (trace.surprise_norm, trace.momentum_norm, trace.write_norm):
                    series.append(zero)
                for series in (trace.forget_gate, trace.learning_rate, trace.momentum_gate):
                    series.append(zero)
            return self.output(retrieved), state

        theta, eta, alpha = self.gates(step_input)
        loss, *gradients = self._surprise(step_input, state.weights)

        # Equation 14 then equation 13, in that order.
        momentum = tuple(
            eta.view(-1, 1, 1) * previous - theta.view(-1, 1, 1) * gradient
            for previous, gradient in zip(state.momentum, gradients, strict=True)
        )
        weights = tuple(
            (1.0 - alpha.view(-1, 1, 1)) * previous + surprise
            for previous, surprise in zip(state.weights, momentum, strict=True)
        )

        if trace is not None:
            trace.loss.append(float(loss))
            trace.surprise_norm.append(_mean_norm(gradients))
            trace.momentum_norm.append(_mean_norm(momentum))
            trace.write_norm.append(
                _mean_norm(
                    tuple(new - old for new, old in zip(weights, state.weights, strict=True))
                )
            )
            trace.forget_gate.append(float(alpha.mean()))
            trace.learning_rate.append(float(theta.mean()))
            trace.momentum_gate.append(float(eta.mean()))

        retrieved = self.read(self.query(step_input), weights)
        return self.output(retrieved), MemoryState(weights=weights, momentum=momentum)

    def forward(self, sequence: Tensor, trace: MemoryTrace | None = None) -> Tensor:
        """Run the memory over a sequence.

        Args:
            sequence: Shape ``(B, T, D)``.
            trace: Optional recorder for per-token diagnostics.

        Returns:
            Shape ``(B, T, D)``.

        Raises:
            ValueError: If ``sequence`` is not three-dimensional.
        """

        if sequence.ndim != 3:
            raise ValueError(f"expected shape (B, T, D), got {tuple(sequence.shape)}")

        state = self.initial_state(sequence.shape[0])
        outputs = []
        for index in range(sequence.shape[1]):
            output, state = self.step(sequence[:, index], state, trace)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def extra_repr(self) -> str:
        """Return a compact description for ``print(model)``."""

        return (
            f"d_model={self.d_model}, d_memory={self.d_memory}, depth={self.depth}, "
            f"updates_enabled={self.updates_enabled}, momentum={self.use_momentum}, "
            f"forgetting={self.use_forgetting}"
        )


def _mean_norm(tensors: Sequence[Tensor]) -> float:
    """Return the mean Frobenius norm across a batch and a group of tensors."""

    total = torch.zeros(())
    for tensor in tensors:
        flat = tensor.detach().reshape(tensor.shape[0], -1)
        total = total + flat.norm(dim=-1).mean()
    return float(total / max(len(tensors), 1))
