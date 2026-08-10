"""The token model that wraps whichever mixture layer is under test.

Everything outside the layer is identical across the comparison — the same input
projection, the same residual connection, the same readout — so a difference in the task
metric is attributable to the layer and not to the scaffolding around it.

The residual connection matters for more than optimization here. A token whose every
expert assignment was dropped for lack of capacity receives exactly zero from a sparse
layer, so the residual is what that token falls back to: its own projected input, passed
through unchanged. Capacity overflow therefore degrades a token to "not processed by this
layer" rather than to "erased", which is the behaviour the prompt asks to be explicit
about.
"""

from __future__ import annotations

from torch import Tensor, nn

from modern_nn_lab.tracks.moe.config import MoEConfig
from modern_nn_lab.tracks.moe.layer import DenseFFN, DenseMoELayer, SparseMoELayer
from modern_nn_lab.tracks.moe.router import RoutingInfo

LayerVariant = DenseFFN | DenseMoELayer | SparseMoELayer


def build_layer(config: MoEConfig) -> LayerVariant:
    """Construct the mixture layer named by the configuration.

    Args:
        config: Track configuration.

    Returns:
        The layer.

    Raises:
        ValueError: If the layer kind is unknown.
    """

    if config.layer == "dense-ffn":
        # Widened so its parameter count is comparable to the expert bank it replaces;
        # see MoEConfig.dense_hidden.
        return DenseFFN(config.d_model, config.dense_hidden)
    if config.layer == "dense-moe":
        return DenseMoELayer(config.d_model, config.d_hidden, config.n_experts)
    if config.layer == "sparse-moe":
        return SparseMoELayer(
            config.d_model,
            config.d_hidden,
            config.n_experts,
            top_k=config.top_k,
            capacity_factor=config.capacity_factor,
            renormalize=config.renormalize,
        )
    raise ValueError(f"unknown layer {config.layer!r}")


class MixtureModel(nn.Module):
    """Project, apply one mixture layer with a residual, and read out.

    Attributes:
        config: The configuration used to build the model.
        layer: The mixture layer under test.
    """

    def __init__(self, config: MoEConfig) -> None:
        """Build the model.

        Args:
            config: Track configuration.
        """

        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.d_in, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.layer = build_layer(config)
        self.readout = nn.Linear(config.d_model, config.d_out)
        self.last_routing: RoutingInfo | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        """Predict a value per token.

        The routing diagnostics are stashed on ``last_routing`` rather than returned,
        because the shared training loop expects a model to return predictions alone. The
        auxiliary loss is read from there by the track's loss function immediately after
        the forward pass.

        Args:
            inputs: Shape ``(B, T, d_in)``.

        Returns:
            Shape ``(B, T, d_out)``.

        Raises:
            ValueError: If the input is not three-dimensional.
        """

        if inputs.ndim != 3:
            raise ValueError(f"expected inputs of shape (B, T, d_in), got {tuple(inputs.shape)}")

        hidden = self.norm(self.input_projection(inputs))
        delta, info = self.layer(hidden)
        self.last_routing = info
        predictions: Tensor = self.readout(hidden + delta)
        return predictions

    def flops_per_token(self) -> int:
        """Return the layer's FLOPs per token, excluding the shared scaffolding.

        The projection and readout are identical across every variant, so excluding them
        keeps the ratio between variants meaningful rather than diluted by a shared cost.

        Returns:
            FLOPs per token.
        """

        return self.layer.flops_per_token()

    def activated_parameters(self) -> int:
        """Return parameters used per token, across the whole model.

        Returns:
            Scaffolding parameters plus the layer's activated share.
        """

        shared = (
            sum(parameter.numel() for parameter in self.input_projection.parameters())
            + sum(parameter.numel() for parameter in self.norm.parameters())
            + sum(parameter.numel() for parameter in self.readout.parameters())
        )
        return shared + self.layer.activated_parameters()
