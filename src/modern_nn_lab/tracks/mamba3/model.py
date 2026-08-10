"""The Mamba-3 model: a stack of pre-norm residual selective-SSM blocks.

Baselines and the shared token-sequence contract live in
:mod:`modern_nn_lab.models.sequence`, so this track and the xLSTM track are compared
inside identical scaffolding.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from modern_nn_lab.models.sequence import TokenSequenceModel
from modern_nn_lab.tracks.mamba3.config import Mamba3Config
from modern_nn_lab.tracks.mamba3.ssm import SelectiveSSM


class Mamba3(TokenSequenceModel):
    """A stack of pre-norm residual :class:`SelectiveSSM` blocks.

    Shapes:
        - input: ``(B, T)`` integer token ids
        - output: ``(B, T, V)`` logits

    Attributes:
        config: The configuration used to build the model.
    """

    blocks: nn.ModuleList
    block_norms: nn.ModuleList

    def __init__(self, vocab_size: int, config: Mamba3Config) -> None:
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
            SelectiveSSM(
                config.d_model,
                heads=config.heads,
                state_size=config.state_size,
                rank=config.rank,
                trapezoidal=config.trapezoidal,
                rotary=config.rotary,
            )
            for _ in range(config.n_blocks)
        )

    @property
    def layers(self) -> tuple[tuple[SelectiveSSM, nn.LayerNorm], ...]:
        """Typed view of the ``(ssm, norm)`` pairs.

        Returns:
            One pair per block, in order.

        Raises:
            TypeError: If a registered module is not of the expected type.
        """

        pairs: list[tuple[SelectiveSSM, nn.LayerNorm]] = []
        for block, norm in zip(self.blocks, self.block_norms, strict=True):
            if not isinstance(block, SelectiveSSM) or not isinstance(norm, nn.LayerNorm):
                raise TypeError("unexpected module registered in a Mamba-3 block")
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


def parity_reference(tokens: Tensor) -> Tensor:
    """Running parity computed by an explicit rotation, as the source describes it.

    The source motivates complex-valued states by noting that parity on binary inputs is
    solved by ``h_t = R(pi * x_t) h_{t-1}``, which real non-negative transitions cannot
    express. This function is that construction, used in the tests as ground truth that
    the *mechanism* — not the learned model — can represent the task.

    Args:
        tokens: Shape ``(B, T)`` binary tokens.

    Returns:
        Shape ``(B, T)`` running parity, computed by accumulating rotations.
    """

    batch, length = tokens.shape
    state = torch.zeros(batch, 2)
    state[:, 0] = 1.0  # unit vector at angle 0

    outputs = []
    for index in range(length):
        angle = torch.pi * tokens[:, index].to(state.dtype)
        cos, sin = torch.cos(angle), torch.sin(angle)
        rotated = torch.stack(
            (cos * state[:, 0] - sin * state[:, 1], sin * state[:, 0] + cos * state[:, 1]),
            dim=-1,
        )
        state = rotated
        # angle 0 -> parity 0, angle pi -> parity 1
        outputs.append((state[:, 0] < 0).long())
    return torch.stack(outputs, dim=1)
