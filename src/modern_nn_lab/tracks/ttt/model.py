"""The TTT model: a stack of pre-norm residual Test-Time Training blocks."""

from __future__ import annotations

from torch import Tensor, nn

from modern_nn_lab.models.sequence import TokenSequenceModel
from modern_nn_lab.tracks.ttt.config import TTTConfig
from modern_nn_lab.tracks.ttt.layer import TTTLayer


class TTT(TokenSequenceModel):
    """A stack of pre-norm residual :class:`TTTLayer` blocks.

    Shapes:
        - input: ``(B, T)`` integer token ids
        - output: ``(B, T, V)`` logits

    Attributes:
        config: The configuration used to build the model.
    """

    blocks: nn.ModuleList
    block_norms: nn.ModuleList

    def __init__(self, vocab_size: int, config: TTTConfig) -> None:
        """Build the model.

        Args:
            vocab_size: Number of token ids.
            config: Track configuration.
        """

        super().__init__(vocab_size, config.d_model)
        self.config = config
        self.block_norms = nn.ModuleList(
            nn.LayerNorm(config.d_model) for _ in range(config.n_blocks)
        )
        self.blocks = nn.ModuleList(
            TTTLayer(
                config.d_model,
                d_inner=config.d_inner,
                inner_model=config.inner_model,
                update_rule=config.update_rule,
                learner_updates=config.learner_updates,
                layernorm_residual=config.layernorm_residual,
            )
            for _ in range(config.n_blocks)
        )

    @property
    def layers(self) -> tuple[tuple[TTTLayer, nn.LayerNorm], ...]:
        """Typed view of the ``(ttt_layer, norm)`` pairs.

        Returns:
            One pair per block, in order.

        Raises:
            TypeError: If a registered module is not of the expected type.
        """

        pairs: list[tuple[TTTLayer, nn.LayerNorm]] = []
        for block, norm in zip(self.blocks, self.block_norms, strict=True):
            if not isinstance(block, TTTLayer) or not isinstance(norm, nn.LayerNorm):
                raise TypeError("unexpected module registered in a TTT block")
            pairs.append((block, norm))
        return tuple(pairs)

    def forward(self, tokens: Tensor) -> Tensor:
        """Run the stack.

        Args:
            tokens: Shape ``(B, T)``.

        Returns:
            Shape ``(B, T, V)`` logits.
        """

        hidden: Tensor = self.encode(tokens)
        for block, norm in self.layers:
            hidden = hidden + block(norm(hidden))
        return self.decode(hidden)
