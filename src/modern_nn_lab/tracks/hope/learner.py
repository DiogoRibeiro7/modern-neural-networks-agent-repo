r"""Nested learners: one, two, and self-referential levels over a stream.

Each learner below is a *composition of levels* from
:mod:`modern_nn_lab.tracks.hope.levels`, and the only difference between them is how many
levels there are and how the signal flows between them. That is the whole point of the
audit: "use momentum" and "add an optimization level" must be the same edit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from modern_nn_lab.tracks.hope.levels import (
    DataMemory,
    DataMemoryConfig,
    GradientMemory,
    LevelState,
    NestedTrace,
    local_surprise_signal,
    weight_gradient,
)

ValueFn = Callable[[Tensor, Tensor], Tensor]
"""Maps ``(weights, inputs)`` to a self-generated value, per source equation 60."""


@dataclass(slots=True)
class LearnerConfig:
    """Configuration of a nested learner.

    Attributes:
        d_in: Input width.
        d_out: Output width.
        learning_rate: Step size of the data-memory level.
        use_gradient_memory: Add the L1 momentum level. ``False`` is one-level SGD.
        momentum_decay: Retention of the gradient memory.
        momentum_period: Update period of L1 in samples; ``> 1`` makes it a slower level.
        rule: Learning rule of the data memory.
        init_scale: Standard deviation of the initial L0 weights. Zero starts from an
            exactly zero matrix, which makes the learner fully deterministic and would
            make every seed produce an identical run.
    """

    d_in: int = 8
    d_out: int = 4
    learning_rate: float = 0.05
    use_gradient_memory: bool = True
    momentum_decay: float = 0.9
    momentum_period: int = 1
    rule: str = "gradient"
    init_scale: float = 0.01

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            ValueError: If a width or period is invalid.
        """

        if self.d_in <= 0 or self.d_out <= 0:
            raise ValueError("d_in and d_out must be positive")
        if self.momentum_period <= 0:
            raise ValueError("momentum_period must be positive")
        if self.init_scale < 0:
            raise ValueError("init_scale must be non-negative")

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot for the experiment record."""

        return {
            "d_in": self.d_in,
            "d_out": self.d_out,
            "learning_rate": self.learning_rate,
            "use_gradient_memory": self.use_gradient_memory,
            "momentum_decay": self.momentum_decay,
            "momentum_period": self.momentum_period,
            "rule": self.rule,
            "init_scale": self.init_scale,
            "levels": 2 if self.use_gradient_memory else 1,
        }


class NestedLearner:
    """A learner built from explicit optimization levels.

    Signal flow, for the two-level case::

        gradient --> [L1 gradient memory] --> momentum --> [L0 data memory] --> weights

    Removing L1 sends the gradient straight into L0, which is plain gradient descent.

    Attributes:
        config: The learner's settings.
        data_memory: The L0 level.
        gradient_memory: The L1 level, or ``None`` for a one-level learner.
    """

    def __init__(self, config: LearnerConfig) -> None:
        """Build the learner.

        Args:
            config: Learner settings.
        """

        self.config = config
        self.data_memory = DataMemory(
            DataMemoryConfig(learning_rate=config.learning_rate, rule=config.rule)
        )
        self.gradient_memory = (
            GradientMemory(
                learning_rate=config.learning_rate,
                decay=config.momentum_decay,
                period=config.momentum_period,
            )
            if config.use_gradient_memory
            else None
        )
        self.data_state: LevelState = self.data_memory.initial_state((config.d_out, config.d_in))
        if config.init_scale > 0:
            # Seeded random initialization, so different seeds are different runs. With
            # a zero start the learner is deterministic and five seeds would report an
            # interval of exactly zero width, overstating the evidence.
            self.data_state.tensors["weights"] = (
                torch.randn(config.d_out, config.d_in) * config.init_scale
            )
        self.gradient_state: LevelState | None = (
            self.gradient_memory.initial_state((config.d_out, config.d_in))
            if self.gradient_memory is not None
            else None
        )

    @property
    def weights(self) -> Tensor:
        """The L0 state."""

        return self.data_state.tensors["weights"]

    def reset_gradient_memory(self) -> None:
        """Discard the L1 state, keeping L0.

        The source argues that knowledge about the loss landscape lives in the momentum
        state, so discarding it between tasks should be measurable. This method exists to
        make that manipulation available, not to assert that it matters.
        """

        if self.gradient_memory is not None:
            self.gradient_state = self.gradient_memory.initial_state(
                (self.config.d_out, self.config.d_in)
            )

    def step(self, inputs: Tensor, targets: Tensor, trace: NestedTrace | None = None) -> float:
        """Consume one sample and update every level that is due.

        Args:
            inputs: Shape ``(d_in,)``.
            targets: Shape ``(d_out,)``.
            trace: Optional recorder.

        Returns:
            The squared-error loss *before* the update.

        Raises:
            ValueError: If the shapes do not match the configuration.
        """

        if inputs.shape != (self.config.d_in,) or targets.shape != (self.config.d_out,):
            raise ValueError(
                f"expected inputs {(self.config.d_in,)} and targets {(self.config.d_out,)}, "
                f"got {tuple(inputs.shape)} and {tuple(targets.shape)}"
            )

        weights = self.weights
        error = local_surprise_signal(weights, inputs, targets)
        loss = float(0.5 * (error @ error))
        gradient = weight_gradient(weights, inputs, targets)

        if self.gradient_memory is not None and self.gradient_state is not None:
            self.gradient_state, signal = self.gradient_memory.update(self.gradient_state, gradient)
        else:
            signal = self.config.learning_rate * gradient

        self.data_state, _ = self.data_memory.update(self.data_state, signal)

        if trace is not None:
            trace.loss.append(loss)
            trace.weight_norm.append(float(self.weights.norm()))
            trace.signal_norm.append(float(signal.norm()))
            trace.level_steps.setdefault("data_memory", []).append(self.data_state.steps)
            if self.gradient_state is not None:
                trace.level_steps.setdefault("gradient_memory", []).append(
                    self.gradient_state.steps
                )
        return loss

    def predict(self, inputs: Tensor) -> Tensor:
        """Return ``W x`` for a batch.

        Args:
            inputs: Shape ``(N, d_in)``.

        Returns:
            Shape ``(N, d_out)``.
        """

        return inputs @ self.weights.T


class SelfReferentialLearner(NestedLearner):
    """A learner whose target values are generated by its own state (eqs. 58-60).

    Ordinary supervised learning takes the value from outside. Generalized Gradient
    Descent takes ``u_t = f_{W_t}(x_t)``: the memory produces the thing it then learns
    from. With the default ``value_fn`` this reduces *exactly* to equation 58, which
    ``test_self_generated_value_reduces_to_equation_58`` asserts; supplying a different
    ``value_fn`` is the self-modifying generalization of Definition 5.

    Attributes:
        value_fn: Maps ``(weights, inputs)`` to the self-generated value.
    """

    def __init__(self, config: LearnerConfig, value_fn: ValueFn | None = None) -> None:
        """Build the self-referential learner.

        Args:
            config: Learner settings.
            value_fn: Value generator. ``None`` uses the equation-58 instance, in which
                the generated value is the negated local surprise signal.
        """

        super().__init__(config)
        self.value_fn = value_fn

    def self_generated_value(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Return ``u_t = f_{W_t}(x_t)`` (equation 60).

        Args:
            inputs: Shape ``(d_in,)``.
            targets: Shape ``(d_out,)``. Used only by the default equation-58 instance.

        Returns:
            Shape ``(d_out,)``.
        """

        if self.value_fn is not None:
            return self.value_fn(self.weights, inputs)
        return -local_surprise_signal(self.weights, inputs, targets)

    def step(self, inputs: Tensor, targets: Tensor, trace: NestedTrace | None = None) -> float:
        """Take one self-referential step: ``W <- W + eta u_t (x) x_t``.

        Args:
            inputs: Shape ``(d_in,)``.
            targets: Shape ``(d_out,)``.
            trace: Optional recorder.

        Returns:
            The squared-error loss before the update.
        """

        weights = self.weights
        error = local_surprise_signal(weights, inputs, targets)
        loss = float(0.5 * (error @ error))

        value = self.self_generated_value(inputs, targets)
        # Equation 58 adds eta * u (x) x; the data-memory level subtracts what it is
        # given, so the sign is flipped once here rather than special-casing the level.
        signal = -self.config.learning_rate * torch.outer(value, inputs)

        if self.gradient_memory is not None and self.gradient_state is not None:
            self.gradient_state, signal = self.gradient_memory.update(
                self.gradient_state, signal / self.config.learning_rate
            )

        self.data_state, _ = self.data_memory.update(self.data_state, signal)

        if trace is not None:
            trace.loss.append(loss)
            trace.weight_norm.append(float(self.weights.norm()))
            trace.signal_norm.append(float(signal.norm()))
            trace.level_steps.setdefault("data_memory", []).append(self.data_state.steps)
        return loss
