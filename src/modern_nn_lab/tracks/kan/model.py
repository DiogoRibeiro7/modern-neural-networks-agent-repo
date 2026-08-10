"""Kolmogorov-Arnold network and its matched-budget MLP baseline."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from modern_nn_lab.tracks.kan.config import KANConfig
from modern_nn_lab.tracks.kan.layers import KANLayer


class KAN(nn.Module):
    """A stack of :class:`~modern_nn_lab.tracks.kan.layers.KANLayer` modules.

    No activation function sits between layers: the nonlinearity lives on the edges, so
    inserting one would confound the mechanism under study.

    Shapes:
        - input: ``(B, layer_widths[0])``
        - output: ``(B, layer_widths[-1])``

    Attributes:
        layers: The KAN layers, in order.
        config: The configuration used to build the network.
    """

    layers: nn.ModuleList

    def __init__(self, config: KANConfig) -> None:
        """Build a KAN from a validated configuration.

        Args:
            config: Track configuration.
        """

        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            KANLayer(
                in_features,
                out_features,
                grid_size=config.grid_size,
                spline_order=config.spline_order,
                grid_range=config.grid_range,
                base_scale=config.base_scale,
                spline_noise_scale=config.spline_noise_scale,
                learnable_spline=config.learnable_spline,
                use_base_branch=config.use_base_branch,
            )
            for in_features, out_features in zip(
                config.layer_widths[:-1], config.layer_widths[1:], strict=True
            )
        )

    @property
    def kan_layers(self) -> tuple[KANLayer, ...]:
        """Typed view of the registered layers.

        ``nn.ModuleList`` erases the element type, so methods that need the KAN-specific
        interface go through this accessor rather than casting at each call site.

        Returns:
            The layers, in order.
        """

        layers: list[KANLayer] = []
        for layer in self.layers:
            if not isinstance(layer, KANLayer):  # pragma: no cover - construction invariant
                raise TypeError(f"unexpected layer type {type(layer).__name__}")
            layers.append(layer)
        return tuple(layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the network.

        Args:
            inputs: Shape ``(B, layer_widths[0])``.

        Returns:
            Shape ``(B, layer_widths[-1])``.
        """

        activations: Tensor = inputs
        for layer in self.layers:
            activations = layer(activations)
        return activations

    @torch.no_grad()
    def update_grids(self, inputs: Tensor, *, uniform_mixture: float = 0.02) -> None:
        """Adapt every layer's grid to the activations it actually receives.

        Layers are updated front to back, each using the activations produced by the
        already-updated layers below it.

        Args:
            inputs: Shape ``(B, layer_widths[0])`` batch of network inputs.
            uniform_mixture: Blend weight towards a uniform grid.
        """

        activations: Tensor = inputs
        for layer in self.kan_layers:
            layer.update_grid(activations, uniform_mixture=uniform_mixture)
            activations = layer(activations)

    def regularization(self) -> Tensor:
        """Sum the per-layer L1 surrogate over the network.

        Returns:
            Scalar tensor.
        """

        total = torch.zeros((), device=next(self.parameters()).device)
        for layer in self.kan_layers:
            total = total + layer.regularization()
        return total

    @torch.no_grad()
    def first_layer_edges(self, samples: Tensor) -> Tensor:
        """Evaluate the first layer's edge functions, the ones with a direct input meaning.

        Args:
            samples: Shape ``(S,)`` evaluation points.

        Returns:
            Shape ``(out_features, in_features, S)``.
        """

        return self.kan_layers[0].edge_functions(samples)


class MLP(nn.Module):
    """Fully connected baseline with the same input and output widths as a KAN.

    Kept in the track package rather than a shared module because the comparison is
    only meaningful with the width-matching logic in :func:`match_parameter_budget`.

    Shapes:
        - input: ``(B, in_features)``
        - output: ``(B, out_features)``
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        hidden_widths: Sequence[int],
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        """Build the baseline.

        Args:
            in_features: Input width.
            out_features: Output width.
            hidden_widths: Hidden layer widths; may be empty for a linear model.
            activation: Activation module inserted between linear layers.

        Raises:
            ValueError: If any width is not positive.
        """

        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if any(width <= 0 for width in hidden_widths):
            raise ValueError("hidden widths must be positive")

        widths = [in_features, *hidden_widths, out_features]
        modules: list[nn.Module] = []
        for index, (fan_in, fan_out) in enumerate(itertools.pairwise(widths)):
            modules.append(nn.Linear(fan_in, fan_out))
            if index < len(widths) - 2:
                modules.append(activation())
        self.net = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the baseline.

        Args:
            inputs: Shape ``(B, in_features)``.

        Returns:
            Shape ``(B, out_features)``.
        """

        result: Tensor = self.net(inputs)
        return result


def count_parameters(model: nn.Module) -> int:
    """Return the total number of parameters in ``model``.

    Args:
        model: Module to inspect.

    Returns:
        Parameter count, including frozen parameters, because capacity comparisons are
        about representational size, not about which weights happen to receive updates.
    """

    return sum(parameter.numel() for parameter in model.parameters())


def match_parameter_budget(
    target_parameters: int,
    in_features: int,
    out_features: int,
    *,
    depth: int = 1,
    max_width: int = 4096,
) -> list[int]:
    """Find hidden widths whose MLP parameter count is closest to ``target_parameters``.

    A KAN edge holds ``G + k + 1`` parameters where a linear edge holds one, so an
    equal-width MLP is far smaller. Comparing them without this correction would report
    a capacity difference as an architecture difference.

    Args:
        target_parameters: Parameter count to match, typically a KAN's.
        in_features: Input width.
        out_features: Output width.
        depth: Number of hidden layers, all of equal width.
        max_width: Upper bound of the search.

    Returns:
        A list of ``depth`` equal hidden widths.

    Raises:
        ValueError: If ``depth`` is negative or ``max_width`` is not positive.
    """

    if depth < 0:
        raise ValueError("depth must be non-negative")
    if max_width <= 0:
        raise ValueError("max_width must be positive")
    if depth == 0:
        return []

    best_width = 1
    best_gap = float("inf")
    for width in range(1, max_width + 1):
        candidate = MLP(in_features, out_features, hidden_widths=[width] * depth)
        gap = abs(count_parameters(candidate) - target_parameters)
        if gap < best_gap:
            best_gap, best_width = gap, width
        elif gap > best_gap:
            # Parameter count is monotone in width, so the gap is unimodal: stop early.
            break
    return [best_width] * depth
