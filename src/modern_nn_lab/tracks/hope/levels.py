r"""Optimization levels as explicit, separately testable objects.

The Nested Learning claim this track audits is structural: a learning system is a set of
optimization problems, each with **its own state, its own objective, and its own update
frequency**. That claim is only meaningful if those three things are separable in code, so
here each level is an object that owns exactly one state tensor group and declares how
often it updates.

The derivation behind every rule below is in
[`docs/nested_learning_audit.md`](../../../../docs/nested_learning_audit.md); equation
numbers refer to the primary source.

The level table this module implements:

===============  ==================  ==========================================  ==========
level            state               objective                                   updates
===============  ==================  ==========================================  ==========
L0 data memory   ``W``               maps inputs to their Local Surprise Signal  every step
L1 gradient mem  ``m``               maps inputs to their LSS-value (eq. 13)     every ``k``
===============  ==================  ==========================================  ==========

Setting L1 aside recovers plain gradient descent exactly, which is the point: "add a
level" and "use momentum" are the same operation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor

EPS = 1e-12
"""Floor for norms used in normalized update rules."""


@dataclass(slots=True)
class LevelState:
    """State owned by one optimization level.

    Attributes:
        tensors: The level's state. For ``L0`` this is the weight matrix; for ``L1`` the
            momentum buffer.
        steps: How many times this level has *updated*, which is not the same as how many
            samples it has seen when its period is greater than one.
        samples: How many samples this level has been offered.
    """

    tensors: dict[str, Tensor]
    steps: int = 0
    samples: int = 0


class Level(ABC):
    """One optimization level: a state, an objective, and an update frequency.

    Attributes:
        name: Identifier used in traces and records.
        period: Update every ``period`` samples. ``1`` is the classical case; larger
            values make a genuinely slower level, which is what "nested timescales"
            means operationally.
    """

    def __init__(self, name: str, *, period: int = 1) -> None:
        """Create a level.

        Args:
            name: Identifier.
            period: Update period in samples. Must be positive.

        Raises:
            ValueError: If ``period`` is not positive.
        """

        if period <= 0:
            raise ValueError("period must be positive")
        self.name = name
        self.period = period

    @abstractmethod
    def initial_state(self, shape: tuple[int, ...]) -> LevelState:
        """Return the level's zero state for a weight matrix of ``shape``."""

    @abstractmethod
    def update(self, state: LevelState, signal: Tensor) -> tuple[LevelState, Tensor]:
        """Consume an incoming signal and emit this level's contribution.

        Args:
            state: The level's current state.
            signal: What this level compresses. For ``L1`` it is the gradient.

        Returns:
            ``(new_state, output)``. ``output`` is what the *next* level consumes.
        """

    def due(self, state: LevelState) -> bool:
        """Whether this level updates on the upcoming sample.

        Args:
            state: The level's current state.

        Returns:
            ``True`` when the sample index is a multiple of the period.
        """

        return state.samples % self.period == 0


class GradientMemory(Level):
    """L1: the momentum state, read as an associative memory over gradients.

    Implements source equations 10-11: ``m_{t+1} = m_t + eta grad``, with the slow weights
    then updated by ``W_{t+1} = W_t - m_{t+1}``. A decay factor generalizes it to the
    familiar exponential form; ``decay = 1`` is the paper's accumulate-everything version.

    Attributes:
        learning_rate: ``eta`` in equation 11.
        decay: Retention of past gradient information.
    """

    def __init__(self, *, learning_rate: float = 0.1, decay: float = 0.9, period: int = 1) -> None:
        """Create the gradient-memory level.

        Args:
            learning_rate: Step size applied to the incoming gradient.
            decay: Multiplier on the retained state.
            period: Update period in samples.

        Raises:
            ValueError: If the learning rate is not positive or the decay is outside [0, 1].
        """

        super().__init__("gradient_memory", period=period)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must lie in [0, 1]")
        self.learning_rate = learning_rate
        self.decay = decay

    def initial_state(self, shape: tuple[int, ...]) -> LevelState:
        """Return zero momentum."""

        return LevelState(tensors={"momentum": torch.zeros(shape)})

    def update(self, state: LevelState, signal: Tensor) -> tuple[LevelState, Tensor]:
        """Compress the gradient into the momentum state and emit it.

        Args:
            state: Current momentum state.
            signal: The incoming gradient.

        Returns:
            ``(new_state, momentum)``; the emitted momentum is what updates the weights.
        """

        state.samples += 1
        if not self.due(LevelState(tensors=state.tensors, samples=state.samples - 1)):
            # Not this level's turn: pass the retained state through unchanged, so a
            # slower level genuinely holds its value between updates.
            return state, state.tensors["momentum"]

        momentum = self.decay * state.tensors["momentum"] + self.learning_rate * signal
        state.tensors["momentum"] = momentum
        state.steps += 1
        return state, momentum


@dataclass(slots=True)
class DataMemoryConfig:
    """Settings for the L0 data-memory level.

    Attributes:
        learning_rate: Step size when the level consumes a raw gradient.
        rule: ``"gradient"`` for equations 1/8, ``"hebbian"`` for equation 64, ``"delta"``
            for equation 65.
        decay: Retention factor ``alpha`` used by the Hebbian and delta rules.
    """

    learning_rate: float = 0.1
    rule: str = "gradient"
    decay: float = 1.0


class DataMemory(Level):
    """L0: the weight matrix, read as an associative memory over the data stream.

    Three interchangeable learning rules, all shown by the source to be gradient steps on
    an associative objective:

    - ``gradient`` — equations 1 and 8: ``W <- W - eta * grad``.
    - ``hebbian`` — equation 64: ``M <- alpha M + eta v k^T``, purely additive writes.
    - ``delta`` — equation 65: ``M <- (I - eta k k^T) M + eta v k^T``, which *removes* the
      previously stored value for this key before writing the new one.

    Attributes:
        config: The level's settings.
    """

    def __init__(self, config: DataMemoryConfig | None = None, *, period: int = 1) -> None:
        """Create the data-memory level.

        Args:
            config: Settings; defaults to plain gradient descent.
            period: Update period in samples.

        Raises:
            ValueError: If the rule is unknown or the settings are invalid.
        """

        super().__init__("data_memory", period=period)
        self.config = config or DataMemoryConfig()
        if self.config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.config.rule not in ("gradient", "hebbian", "delta"):
            raise ValueError(f"unknown rule {self.config.rule!r}")
        if not 0.0 <= self.config.decay <= 1.0:
            raise ValueError("decay must lie in [0, 1]")

    def initial_state(self, shape: tuple[int, ...]) -> LevelState:
        """Return zero weights."""

        return LevelState(tensors={"weights": torch.zeros(shape)})

    def update(self, state: LevelState, signal: Tensor) -> tuple[LevelState, Tensor]:
        """Apply the gradient rule: subtract the incoming update from the weights.

        Args:
            state: Current weights.
            signal: The update to subtract — either a raw gradient or, in a two-level
                system, the momentum emitted by :class:`GradientMemory`.

        Returns:
            ``(new_state, weights)``.
        """

        state.samples += 1
        state.tensors["weights"] = state.tensors["weights"] - signal
        state.steps += 1
        return state, state.tensors["weights"]

    def associative_update(self, state: LevelState, key: Tensor, value: Tensor) -> LevelState:
        """Apply the Hebbian or delta associative rule directly.

        Args:
            state: Current weights, shape ``(d_value, d_key)``.
            key: Shape ``(d_key,)``.
            value: Shape ``(d_value,)``.

        Returns:
            The updated state.

        Raises:
            ValueError: If called on a level configured for the plain gradient rule.
        """

        if self.config.rule == "gradient":
            raise ValueError("associative_update requires the hebbian or delta rule")

        memory = state.tensors["weights"]
        rate = self.config.learning_rate
        outer = torch.outer(value, key)

        if self.config.rule == "hebbian":
            # Equation 64: purely additive write with decay.
            memory = self.config.decay * memory + rate * outer
        else:
            # Equation 65: remove what this key already retrieves, then write.
            memory = memory - rate * torch.outer(memory @ key, key) + rate * outer

        state.tensors["weights"] = memory
        state.samples += 1
        state.steps += 1
        return state


@dataclass(slots=True)
class NestedTrace:
    """Per-step record of which levels updated and what they held.

    Attributes:
        level_steps: Update counts per level, so a slower level's frequency is auditable
            rather than assumed.
        weight_norm: Norm of the L0 state.
        signal_norm: Norm of what L0 consumed.
        loss: The task loss before the update.
    """

    level_steps: dict[str, list[int]] = field(default_factory=dict)
    weight_norm: list[float] = field(default_factory=list)
    signal_norm: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        """Return the trace as JSON-serializable lists."""

        return {
            "level_steps": self.level_steps,
            "weight_norm": self.weight_norm,
            "signal_norm": self.signal_norm,
            "loss": self.loss,
        }


def local_surprise_signal(weights: Tensor, inputs: Tensor, targets: Tensor) -> Tensor:
    r"""Return ``grad_y L`` for a linear map under squared error — the LSS of equation 8.

    For ``y = W x`` and ``L = 1/2 ||y - t||^2`` this is simply ``y - t``. Keeping it as a
    named function matters: equation 8 factors the weight gradient into *this* signal
    times the input, and the self-referential rule of equation 58 is defined by generating
    it from the memory's own state.

    Args:
        weights: Shape ``(d_out, d_in)``.
        inputs: Shape ``(d_in,)``.
        targets: Shape ``(d_out,)``.

    Returns:
        Shape ``(d_out,)``.
    """

    return weights @ inputs - targets


def weight_gradient(weights: Tensor, inputs: Tensor, targets: Tensor) -> Tensor:
    """Return the weight gradient as the outer product of equation 8.

    Args:
        weights: Shape ``(d_out, d_in)``.
        inputs: Shape ``(d_in,)``.
        targets: Shape ``(d_out,)``.

    Returns:
        Shape ``(d_out, d_in)``, equal to ``LSS (x) input``.
    """

    return torch.outer(local_surprise_signal(weights, inputs, targets), inputs)
