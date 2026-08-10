r"""Expert banks, and the three layers this track compares.

All three hold the *same* expert bank shape, so the comparison is about how experts are
used rather than how many parameters exist:

:class:`DenseFFN`
    One feed-forward network. No routing, no experts. The conventional baseline whose cost
    the sparse layer is trying to beat.
:class:`DenseMoELayer`
    Milestone 1 of the track prompt: every expert runs on every token and the outputs are
    combined by the full gate distribution. This is the *reference* a sparse layer
    approximates, and it is the right upper bound on routing quality — it pays the full
    cost of consulting everyone and never drops a token.
:class:`SparseMoELayer`
    Top-k dispatch with capacity. Cheaper than the dense ensemble, and the question is how
    much quality that costs.

The dense ensemble is what makes a sparse result interpretable. If sparse matches dense,
routing is choosing well; if it falls short, the gap is the price of sparsity and can be
attributed to a specific cause — a low top-k, a tight capacity, or a collapsed router — by
reading the diagnostics beside the metric.

Dispatch is implemented as a loop over experts rather than a single batched matmul. With a
handful of experts that is not the bottleneck, and it keeps the correspondence between the
code and the routing rule visible, which matters more here than throughput: this track's
throughput numbers are reported as *measured on CPU* and are not a claim about what an
optimized kernel would do.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from modern_nn_lab.tracks.moe.router import RoutingInfo, TopKRouter


def build_expert(d_model: int, d_hidden: int) -> nn.Module:
    """Build one expert: a two-layer feed-forward network.

    Args:
        d_model: Input and output width.
        d_hidden: Hidden width.

    Returns:
        The expert.
    """

    return nn.Sequential(
        nn.Linear(d_model, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_model),
    )


def expert_flops(d_model: int, d_hidden: int) -> int:
    """Return the multiply-accumulate FLOPs one expert costs for one token.

    Counted as two multiply-accumulates per parameter of the two matmuls, which is the
    usual convention; biases and the activation are not counted because they do not scale
    with width and would only blur the dense-versus-sparse ratio this is used for.

    Args:
        d_model: Input and output width.
        d_hidden: Hidden width.

    Returns:
        FLOPs per token.
    """

    return 2 * (2 * d_model * d_hidden)


class DenseFFN(nn.Module):
    """A single feed-forward network: the conventional baseline.

    Attributes:
        d_model: Input and output width.
        d_hidden: Hidden width.
    """

    def __init__(self, d_model: int, d_hidden: int) -> None:
        """Build the layer.

        Args:
            d_model: Input and output width.
            d_hidden: Hidden width.
        """

        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.network = build_expert(d_model, d_hidden)

    def forward(self, tokens: Tensor) -> tuple[Tensor, RoutingInfo | None]:
        """Apply the network.

        Args:
            tokens: Shape ``(..., d_model)``.

        Returns:
            ``(output, None)``; the second element exists so every layer in this track has
            one signature, and ``None`` states plainly that there is no routing to report.
        """

        output: Tensor = self.network(tokens)
        return output, None

    def flops_per_token(self) -> int:
        """Return FLOPs per token."""

        return expert_flops(self.d_model, self.d_hidden)

    def activated_parameters(self) -> int:
        """Return parameters used per token, which here is all of them."""

        return sum(parameter.numel() for parameter in self.parameters())


class DenseMoELayer(nn.Module):
    """Every expert runs on every token; outputs are combined by the full gate.

    Attributes:
        n_experts: Number of experts.
        d_model: Input and output width.
        d_hidden: Expert hidden width.
    """

    def __init__(self, d_model: int, d_hidden: int, n_experts: int) -> None:
        """Build the layer.

        Args:
            d_model: Input and output width.
            d_hidden: Expert hidden width.
            n_experts: Number of experts.

        Raises:
            ValueError: If ``n_experts`` is below two.
        """

        super().__init__()
        if n_experts < 2:
            raise ValueError("n_experts must be at least 2")
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_experts = n_experts
        self.experts = nn.ModuleList(build_expert(d_model, d_hidden) for _ in range(n_experts))
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, tokens: Tensor) -> tuple[Tensor, RoutingInfo]:
        """Combine every expert's output by its gate.

        Args:
            tokens: Shape ``(..., d_model)``.

        Returns:
            ``(output, info)``. The diagnostics are reported on the same footing as the
            sparse layer's so the two are directly comparable, with a dropped fraction of
            zero because a dense ensemble cannot drop a token.
        """

        shape = tokens.shape
        flat = tokens.reshape(-1, shape[-1])
        probabilities = torch.softmax(self.gate(flat), dim=-1)

        stacked = torch.stack([expert(flat) for expert in self.experts], dim=1)
        combined = (stacked * probabilities.unsqueeze(-1)).sum(dim=1)

        entropy = -(probabilities.clamp_min(1e-9).log() * probabilities).sum(dim=-1).mean()
        info = RoutingInfo(
            load_balancing_loss=torch.zeros((), device=flat.device),
            entropy=entropy,
            normalized_entropy=entropy / torch.log(torch.tensor(float(self.n_experts))),
            utilization=probabilities.mean(dim=0),
            gate_mass=probabilities.mean(dim=0),
            dropped_fraction=0.0,
            capacity=flat.shape[0],
        )
        return combined.reshape(shape), info

    def flops_per_token(self) -> int:
        """Return FLOPs per token: every expert, every time."""

        return self.n_experts * expert_flops(self.d_model, self.d_hidden) + (
            self.d_model * self.n_experts
        )

    def activated_parameters(self) -> int:
        """Return parameters used per token, which here is all of them."""

        return sum(parameter.numel() for parameter in self.parameters())


class SparseMoELayer(nn.Module):
    """Top-k dispatch to a bank of experts, with capacity and a balancing loss.

    Attributes:
        n_experts: Number of experts.
        top_k: Experts each token visits.
        d_model: Input and output width.
        d_hidden: Expert hidden width.
        router: The routing module.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        n_experts: int,
        *,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        renormalize: bool = True,
    ) -> None:
        """Build the layer.

        Args:
            d_model: Input and output width.
            d_hidden: Expert hidden width.
            n_experts: Number of experts.
            top_k: Experts each token visits.
            capacity_factor: Capacity multiplier.
            renormalize: Rescale kept gates to sum to one.
        """

        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(build_expert(d_model, d_hidden) for _ in range(n_experts))
        self.router = TopKRouter(
            d_model,
            n_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
            renormalize=renormalize,
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, RoutingInfo]:
        """Dispatch, apply experts, and combine.

        A token whose every assignment was dropped for lack of capacity receives an output
        of exactly zero from this layer. That is the explicit overflow behaviour: the loss
        is visible in the output rather than absorbed silently, and in a residual network
        it degrades to passing the input through unchanged.

        Args:
            tokens: Shape ``(..., d_model)``.

        Returns:
            ``(output, info)``.
        """

        shape = tokens.shape
        flat = tokens.reshape(-1, shape[-1])
        expert_index, gate, keep, info = self.router(flat)

        output = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            selected = (expert_index == expert_id) & keep
            if not bool(selected.any()):
                continue
            token_positions, slots = selected.nonzero(as_tuple=True)
            expert_output = expert(flat[token_positions])
            weights = gate[token_positions, slots].unsqueeze(-1)
            output.index_add_(0, token_positions, expert_output * weights)

        return output.reshape(shape), info

    def flops_per_token(self) -> int:
        """Return FLOPs per token: the router, plus ``top_k`` experts."""

        return self.top_k * expert_flops(self.d_model, self.d_hidden) + (
            self.d_model * self.n_experts
        )

    def activated_parameters(self) -> int:
        """Return parameters used per token.

        This is the number the track prompt requires alongside the total, and the two are
        very different for a sparse layer: the router always runs, but only ``top_k`` of
        the experts do.

        Returns:
            Router parameters plus ``top_k`` experts' worth.
        """

        router = sum(parameter.numel() for parameter in self.router.parameters())
        per_expert = sum(parameter.numel() for parameter in self.experts[0].parameters())
        return router + self.top_k * per_expert
