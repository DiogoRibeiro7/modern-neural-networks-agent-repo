r"""Test-Time Training layers: the hidden state is a learner, not a vector.

The mechanism, from the primary source: a TTT layer's recurrent state is the *weights*
``W`` of an inner model ``f``, and the "recurrence" is a gradient step on a self-supervised
reconstruction loss computed from the current token.

.. math::

    \ell(W; x_t) = \lVert f(\theta_K x_t; W) - \theta_V x_t \rVert^2   \qquad\text{(eq. 4)}

    W_t = W_{t-1} - \eta_t \nabla \ell(W_{t-1}; x_t)                   \qquad\text{(eq. 6)}

    z_t = f(\theta_Q x_t; W_t)                                          \qquad\text{(eq. 5)}

``θ_K`` is the *training view*, ``θ_V`` the *label view*, ``θ_Q`` the *test view*. All three,
plus the initial state ``W_0`` and the inner learning rate, are ordinary outer-loop
parameters trained by backpropagation. ``W`` is **not** a parameter — it is a hidden state
that is recomputed from scratch for every sequence.

**The distinction that defines this track.** Two different learning problems are running
at once:

===================  ==========================================  ==========================
                     inner loop                                  outer loop
===================  ==========================================  ==========================
what is optimized    ``W``, the inner model's weights            ``θ_K, θ_V, θ_Q, W_0, η``
when                 during the forward pass, every token       during training only
by what objective    self-supervised reconstruction (eq. 4)     the task loss
persists after?      no — discarded at the end of the sequence  yes
===================  ==========================================  ==========================

A forward pass therefore performs gradient descent *without changing a single*
``nn.Parameter``. ``test_forward_pass_never_mutates_outer_parameters`` asserts exactly
that, and it is the operational meaning of "test-time training of the hidden state".

Instantiations follow the source's Subsection 2.7: ``f(x) = x + LN(f_res(x))`` with
``f_res`` either linear or a two-layer GELU MLP with 4x hidden width, a learnable ``W_0``,
and a token-dependent inner learning rate ``eta(x) = eta_base * sigmoid(theta_lr . x)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

InnerModel = Literal["linear", "mlp"]
UpdateRule = Literal["online", "batch"]

MLP_WIDTH_MULTIPLIER = 4
"""Source Subsection 2.7: the inner MLP's hidden dimension is 4x its input dimension."""


@dataclass(frozen=True, slots=True)
class LearnerState:
    """The hidden state of a TTT layer: the inner model's weights.

    Attributes:
        weights: One tensor per inner-model weight matrix, each shaped ``(B, ...)`` —
            every example in the batch carries its *own* learner.
    """

    weights: tuple[Tensor, ...]


class TTTLayer(nn.Module):
    """A TTT layer whose hidden state is an inner model trained during the forward pass.

    Shapes:
        - input: ``(B, T, D)``
        - output: ``(B, T, D)``
        - inner state (linear): one tensor ``(B, d_inner, d_inner)``
        - inner state (mlp): ``(B, d_inner, 4·d_inner)`` and ``(B, 4·d_inner, d_inner)``

    Attributes:
        d_model: Model width ``D``.
        d_inner: Width of the reconstruction views, ``≤ D``.
        inner_model: ``"linear"`` for TTT-Linear, ``"mlp"`` for TTT-MLP.
        update_rule: ``"online"`` takes the gradient at ``W_{t-1}``; ``"batch"`` takes it
            at ``W_0`` for every step, which the source proves is linear attention.
        learner_updates: When false, the inner loop is disabled and ``W_t = W_0`` for all
            ``t``. This is the required ablation.
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_inner: int | None = None,
        inner_model: InnerModel = "linear",
        update_rule: UpdateRule = "online",
        learner_updates: bool = True,
        layernorm_residual: bool = True,
        eta_base: float | None = None,
    ) -> None:
        """Build the layer.

        Args:
            d_model: Model width.
            d_inner: Width of the training/label/test views. Defaults to ``d_model // 2``,
                reflecting the source's note that the training view is a *low-rank*
                projection with fewer dimensions than the token.
            inner_model: Inner model family.
            update_rule: Gradient-descent variant; see the attribute documentation.
            learner_updates: Enable the inner loop. ``False`` is the frozen ablation.
            layernorm_residual: Use ``f(x) = x + LN(f_res(x))`` from Subsection 2.7.
                Disabling it leaves the bare ``f(x) = f_res(x)``, which is what the
                source's Theorem 1 assumes and what the hand-computed tests use.
            eta_base: Base inner learning rate. Defaults to the source's values: ``1.0``
                for TTT-Linear and ``0.1`` for TTT-MLP.

        Raises:
            ValueError: If a width is invalid.
        """

        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")

        resolved_inner = d_model // 2 if d_inner is None else d_inner
        if resolved_inner <= 0:
            raise ValueError("d_inner must be positive")

        self.d_model = d_model
        self.d_inner = resolved_inner
        self.inner_model = inner_model
        self.update_rule = update_rule
        self.learner_updates = learner_updates
        self.layernorm_residual = layernorm_residual
        self.eta_base = (1.0 if inner_model == "linear" else 0.1) if eta_base is None else eta_base

        # Outer-loop parameters. Every one of these is trained by backpropagation and
        # none of them is touched by the inner loop.
        self.train_view = nn.Linear(d_model, resolved_inner, bias=False)
        self.label_view = nn.Linear(d_model, resolved_inner, bias=False)
        self.test_view = nn.Linear(d_model, resolved_inner, bias=False)
        self.output_projection = nn.Linear(resolved_inner, d_model)

        # eta(x) = eta_base * sigmoid(theta_lr . x), the source's learnable inner rate.
        self.learning_rate_projection = nn.Linear(d_model, 1)

        # W_0, the learnable initial hidden state ("theta_init" in the source).
        hidden = MLP_WIDTH_MULTIPLIER * resolved_inner
        if inner_model == "linear":
            self.initial_weights = nn.ParameterList(
                [nn.Parameter(torch.zeros(resolved_inner, resolved_inner))]
            )
        else:
            self.initial_weights = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(resolved_inner, hidden)),
                    nn.Parameter(torch.zeros(hidden, resolved_inner)),
                ]
            )
        self.inner_norm = nn.LayerNorm(resolved_inner)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the outer-loop parameters.

        ``W_0`` is initialized small rather than at zero: the source reports that a
        learnable ``W_0`` improves training stability, and a zero start would make the
        first reconstruction loss depend only on the label view.
        """

        for module in (self.train_view, self.label_view, self.test_view):
            nn.init.normal_(module.weight, std=1.0 / self.d_model**0.5)
        nn.init.zeros_(self.learning_rate_projection.weight)
        nn.init.zeros_(self.learning_rate_projection.bias)
        for weight in self.initial_weights:
            nn.init.normal_(weight, std=1.0 / weight.shape[0] ** 0.5)

    def initial_state(self, batch_size: int) -> LearnerState:
        """Return the learner state at ``t = 0``, one copy per example.

        Args:
            batch_size: Batch size ``B``.

        Returns:
            A :class:`LearnerState` holding ``W_0`` broadcast across the batch.
        """

        return LearnerState(
            weights=tuple(
                weight.unsqueeze(0).expand(batch_size, *weight.shape)
                for weight in self.initial_weights
            )
        )

    def apply_inner_model(self, values: Tensor, weights: Sequence[Tensor]) -> Tensor:
        """Evaluate ``f(values; W)``.

        Args:
            values: Shape ``(B, d_inner)``.
            weights: The inner model's weight tensors, each batched over ``B``.

        Returns:
            Shape ``(B, d_inner)``.
        """

        if self.inner_model == "linear":
            hidden = torch.einsum("bij,bi->bj", weights[0], values)
        else:
            first = torch.einsum("bij,bi->bj", weights[0], values)
            hidden = torch.einsum("bij,bi->bj", weights[1], torch.nn.functional.gelu(first))

        if not self.layernorm_residual:
            return hidden
        # Subsection 2.7: f(x) = x + LN(f_res(x)).
        normalized: Tensor = self.inner_norm(hidden)
        return values + normalized

    def inner_loss(self, step_input: Tensor, weights: Sequence[Tensor]) -> Tensor:
        """Self-supervised reconstruction loss of source equation (4).

        Args:
            step_input: Shape ``(B, D)`` token representations.
            weights: Current inner weights.

        Returns:
            Scalar loss, summed over the batch so each example's gradient is independent.
        """

        prediction = self.apply_inner_model(self.train_view(step_input), weights)
        target: Tensor = self.label_view(step_input)
        # Summed, not averaged: every example carries its own learner, and averaging
        # would scale each example's inner gradient by 1/B.
        return ((prediction - target) ** 2).sum()

    def inner_learning_rate(self, step_input: Tensor) -> Tensor:
        """Token-dependent inner learning rate ``eta(x) = eta_base * sigmoid(theta_lr . x)``.

        Args:
            step_input: Shape ``(B, D)``.

        Returns:
            Shape ``(B, 1)``.
        """

        rate: Tensor = self.eta_base * torch.sigmoid(self.learning_rate_projection(step_input))
        return rate

    def inner_gradient(self, step_input: Tensor, weights: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """Return ``grad_W loss(W; x_t)``, differentiably.

        ``create_graph=True`` is what lets the *outer* loop backpropagate through the
        inner update — the "gradients of gradients" the source describes. Without it the
        views and ``W_0`` would receive no gradient signal from the inner loop at all.

        Args:
            step_input: Shape ``(B, D)``.
            weights: Current inner weights; each must participate in the graph.

        Returns:
            One gradient tensor per weight tensor.
        """

        # The inner loop must run even inside `torch.no_grad()`. Evaluation and
        # profiling wrap the forward pass in it, and a TTT layer that stopped learning
        # under evaluation would not be doing test-time training at all. `create_graph`
        # is decided from the *ambient* mode, captured before enabling grad.
        grad_enabled = torch.is_grad_enabled()
        create_graph = grad_enabled and self.training

        with torch.enable_grad():
            if grad_enabled:
                # Training: differentiate through the existing graph so the outer loop
                # can reach the views and W_0 through the inner update.
                differentiable = tuple(weights)
            else:
                # Inference: the incoming state carries no graph, even though a view of a
                # parameter still advertises `requires_grad=True`. Materialize real leaves
                # instead; `clone` matters because `initial_state` broadcasts W_0 with
                # `expand`, and a stride-0 view is not a usable differentiation target.
                differentiable = tuple(
                    weight.detach().clone().requires_grad_(True) for weight in weights
                )
            loss = self.inner_loss(step_input, differentiable)
            return torch.autograd.grad(loss, differentiable, create_graph=create_graph)

    def step(
        self, step_input: Tensor, state: LearnerState, base: LearnerState
    ) -> tuple[Tensor, LearnerState]:
        """Advance the learner by one token and emit the output.

        Args:
            step_input: Shape ``(B, D)``.
            state: Learner state after the previous token.
            base: The ``t = 0`` state, needed by the ``"batch"`` update rule, which always
                differentiates at ``W_0``.

        Returns:
            ``(output, new_state)`` with ``output`` of shape ``(B, D)``.
        """

        if not self.learner_updates:
            # Frozen ablation: no inner loop at all, so W_t = W_0 for every t.
            output = self.apply_inner_model(self.test_view(step_input), state.weights)
            return self.output_projection(output), state

        gradient_at = base.weights if self.update_rule == "batch" else state.weights
        gradients = self.inner_gradient(step_input, gradient_at)
        eta = self.inner_learning_rate(step_input)

        updated = tuple(
            weight - eta.view(-1, *([1] * (weight.ndim - 1))) * gradient
            for weight, gradient in zip(state.weights, gradients, strict=True)
        )
        new_state = LearnerState(weights=updated)

        # Equation (5): the output uses the *updated* weights.
        output = self.apply_inner_model(self.test_view(step_input), updated)
        return self.output_projection(output), new_state

    def forward(self, sequence: Tensor) -> Tensor:
        """Run the layer over a sequence, training the inner model as it goes.

        Args:
            sequence: Shape ``(B, T, D)``.

        Returns:
            Shape ``(B, T, D)``.

        Raises:
            ValueError: If ``sequence`` is not three-dimensional or has the wrong width.
        """

        if sequence.ndim != 3 or sequence.shape[2] != self.d_model:
            raise ValueError(f"expected shape (B, T, {self.d_model}), got {tuple(sequence.shape)}")

        base = self.initial_state(sequence.shape[0])
        state = base
        outputs = []
        for index in range(sequence.shape[1]):
            output, state = self.step(sequence[:, index], state, base)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def extra_repr(self) -> str:
        """Return a compact description for ``print(model)``."""

        return (
            f"d_model={self.d_model}, d_inner={self.d_inner}, inner_model={self.inner_model}, "
            f"update_rule={self.update_rule}, learner_updates={self.learner_updates}"
        )
