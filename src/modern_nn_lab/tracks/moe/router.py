r"""Top-k routing with explicit capacity, and the diagnostics that make it auditable.

A sparse mixture of experts is not simply a cheaper layer. It is a layer whose behaviour
depends on a *discrete assignment* that the loss only shapes indirectly, so a model can
reach a good task metric while routing pathologically — every token to one expert, or a
third of the tokens dropped on the floor. The track prompt is explicit that accuracy alone
is insufficient, so this module returns its routing statistics as a first-class result
rather than hiding them behind a debug flag.

## The routing rule

Gates come from a softmax over experts, and each token keeps its top :math:`k`:

.. math::

    p(e \mid x) = \operatorname{softmax}(W_r x)_e, \qquad
    \mathcal{T}(x) = \operatorname{top-}k\, p(\cdot \mid x)

The kept gates are **renormalized to sum to one** over the selected experts, so a token's
output is a convex combination and does not silently shrink because its top-k mass was
small. That is a choice, not a necessity — the alternative leaves the raw gates and lets
the norm carry routing confidence — and it is asserted in the tests either way.

.. warning::

    **Renormalizing with** :math:`k = 1` **makes the router untrainable by the task loss.**
    With one expert kept, the renormalized gate is :math:`g / g \equiv 1`: a constant, whose
    derivative with respect to the router's parameters is exactly zero. The expert output is
    then multiplied by a literal ``1.0``, the task loss never reaches the gate, and the only
    thing left shaping routing is the auxiliary balancing loss — which pushes towards
    *uniform*, the opposite of specialization. This is measured rather than asserted: the
    router's gradient norm is ~1e-9 under ``top_k=1, renormalize=True`` against ~3.5e-2
    without renormalization, and the effect on specialization is the headline result of this
    track's report. Use ``renormalize=False`` with top-1 routing.

## Capacity, and what happens when it is exceeded

Each expert accepts at most

.. math:: C = \left\lceil \text{capacity factor} \cdot \frac{N k}{E} \right\rceil

assignments. Beyond that, assignments are **dropped**: the token simply does not visit that
expert, and its contribution is lost rather than being rerouted elsewhere. Overflow is
therefore visible in the output, not merely in a counter, and
:attr:`RoutingInfo.dropped_fraction` reports how much of it happened.

Priority is **slot-major**: every token's first choice is offered capacity before any
token's second choice. Ordering token-major instead would let one token's second preference
displace another token's first, which is the wrong trade at equal cost.

## The load-balancing loss

The Switch-Transformer auxiliary loss, with :math:`f_e` the fraction of tokens dispatched to
expert :math:`e` and :math:`P_e` the mean gate mass it received:

.. math:: \mathcal{L}_{\text{balance}} = E \sum_{e=1}^{E} f_e P_e

It equals :math:`1` under uniform routing and :math:`E` under total collapse, which is a
falsifiable pair of endpoints rather than a vague "encourages balance" — both are asserted
in the tests. Note that it is computed from the routing *before* capacity is applied: it
should penalize the imbalance that causes drops, not be confounded by the drops themselves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class RoutingInfo:
    """Everything about one routing decision that is not the output tensor.

    Attributes:
        load_balancing_loss: Auxiliary loss, ``1.0`` under uniform routing and ``n_experts``
            under collapse.
        entropy: Mean per-token routing entropy, in nats.
        normalized_entropy: ``entropy / log(n_experts)``, so ``1.0`` is uniform and ``0.0``
            is a one-hot router, comparable across expert counts.
        utilization: Shape ``(n_experts,)`` fraction of dispatched assignments per expert.
        gate_mass: Shape ``(n_experts,)`` mean routing probability per expert.
        dropped_fraction: Fraction of top-k assignments discarded for lack of capacity.
        capacity: Assignments each expert accepted.
    """

    load_balancing_loss: Tensor
    entropy: Tensor
    normalized_entropy: Tensor
    utilization: Tensor
    gate_mass: Tensor
    dropped_fraction: float
    capacity: int

    def as_metrics(self) -> dict[str, float]:
        """Return the scalar diagnostics for an experiment record.

        Returns:
            JSON-serializable routing statistics, including the utilization of the
            most- and least-used expert — the pair that reveals collapse.
        """

        return {
            "load_balancing_loss": float(self.load_balancing_loss),
            "routing_entropy": float(self.entropy),
            "routing_entropy_normalized": float(self.normalized_entropy),
            "expert_utilization_max": float(self.utilization.max()),
            "expert_utilization_min": float(self.utilization.min()),
            "dropped_fraction": self.dropped_fraction,
        }


class TopKRouter(nn.Module):
    """Route each token to its top-k experts, subject to a capacity limit.

    Attributes:
        n_experts: Number of experts.
        top_k: Experts each token is sent to.
        capacity_factor: Multiplier on the even share that sets each expert's capacity.
        renormalize: Whether kept gates are rescaled to sum to one.
    """

    def __init__(
        self,
        d_model: int,
        n_experts: int,
        *,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        renormalize: bool = True,
    ) -> None:
        """Build the router.

        Args:
            d_model: Input width.
            n_experts: Number of experts.
            top_k: Experts per token.
            capacity_factor: Capacity multiplier; ``1.0`` allows exactly the even share.
            renormalize: Rescale kept gates to sum to one.

        Raises:
            ValueError: If any argument is outside its permitted range.
        """

        super().__init__()
        if n_experts < 2:
            raise ValueError("n_experts must be at least 2")
        if not 1 <= top_k <= n_experts:
            raise ValueError(f"top_k must lie in [1, {n_experts}], got {top_k}")
        if capacity_factor <= 0:
            raise ValueError("capacity_factor must be positive")

        self.n_experts = n_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.renormalize = renormalize
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def capacity_for(self, n_tokens: int) -> int:
        """Return the per-expert capacity for a batch of a given size.

        Args:
            n_tokens: Tokens being routed.

        Returns:
            At least one, so a batch is never routed to nothing.
        """

        even_share = n_tokens * self.top_k / self.n_experts
        return max(1, math.ceil(self.capacity_factor * even_share))

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor, Tensor, RoutingInfo]:
        """Assign tokens to experts and report how it went.

        Args:
            tokens: Shape ``(N, d_model)`` flattened tokens.

        Returns:
            ``(expert_index, slot_gate, keep, info)`` where ``expert_index`` and
            ``slot_gate`` have shape ``(N, top_k)``, ``keep`` is a boolean mask of the same
            shape marking assignments that fit within capacity, and ``info`` carries the
            diagnostics.

        Raises:
            ValueError: If ``tokens`` is not two-dimensional.
        """

        if tokens.ndim != 2:
            raise ValueError(f"expected tokens of shape (N, d_model), got {tuple(tokens.shape)}")

        probabilities = torch.softmax(self.gate(tokens), dim=-1)
        top_gate, expert_index = probabilities.topk(self.top_k, dim=-1)
        if self.renormalize:
            top_gate = top_gate / top_gate.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        keep, capacity = self._apply_capacity(expert_index, tokens.shape[0])
        info = self._diagnostics(probabilities, expert_index, keep, capacity)
        return expert_index, top_gate, keep, info

    def _apply_capacity(self, expert_index: Tensor, n_tokens: int) -> tuple[Tensor, int]:
        """Mark which assignments fit within each expert's capacity.

        Args:
            expert_index: Shape ``(N, top_k)`` chosen experts.
            n_tokens: Number of tokens.

        Returns:
            ``(keep, capacity)`` with ``keep`` of shape ``(N, top_k)``.
        """

        capacity = self.capacity_for(n_tokens)

        # Slot-major order: every token's first choice is offered capacity before any
        # token's second choice.
        slot_major = expert_index.transpose(0, 1).reshape(-1)
        one_hot = torch.nn.functional.one_hot(slot_major, self.n_experts)
        rank_within_expert = (one_hot.cumsum(dim=0) - 1)[
            torch.arange(slot_major.shape[0], device=expert_index.device), slot_major
        ]

        keep_flat = rank_within_expert < capacity
        keep = keep_flat.reshape(self.top_k, n_tokens).transpose(0, 1).contiguous()
        return keep, capacity

    def _diagnostics(
        self, probabilities: Tensor, expert_index: Tensor, keep: Tensor, capacity: int
    ) -> RoutingInfo:
        """Compute the load-balancing loss, entropy, and utilization.

        Args:
            probabilities: Shape ``(N, n_experts)`` full routing distribution.
            expert_index: Shape ``(N, top_k)`` chosen experts.
            keep: Shape ``(N, top_k)`` capacity mask.
            capacity: Per-expert capacity.

        Returns:
            The diagnostics.
        """

        n_tokens = probabilities.shape[0]
        assignments = torch.nn.functional.one_hot(expert_index.reshape(-1), self.n_experts).float()

        # Computed before capacity is applied: the loss should penalize the imbalance that
        # causes drops, not be confounded by the drops it caused.
        fraction = assignments.sum(dim=0) / max(n_tokens * self.top_k, 1)
        gate_mass = probabilities.mean(dim=0)
        balance = self.n_experts * (fraction * gate_mass).sum()

        entropy = -(probabilities.clamp_min(1e-9).log() * probabilities).sum(dim=-1).mean()
        normalized = entropy / math.log(self.n_experts)

        kept = assignments * keep.reshape(-1, 1).float()
        dispatched = kept.sum()
        utilization = kept.sum(dim=0) / dispatched.clamp_min(1.0)
        dropped = 1.0 - float(dispatched) / max(n_tokens * self.top_k, 1)

        return RoutingInfo(
            load_balancing_loss=balance,
            entropy=entropy,
            normalized_entropy=normalized,
            utilization=utilization,
            gate_mass=gate_mass,
            dropped_fraction=dropped,
            capacity=capacity,
        )
