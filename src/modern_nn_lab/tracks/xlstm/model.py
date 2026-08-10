"""The xLSTM model: a stack of pre-norm residual xLSTM blocks.

Baselines and the shared token-sequence contract live in
:mod:`modern_nn_lab.models.sequence`, because a second track needs them too.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from modern_nn_lab.models.sequence import TokenSequenceModel
from modern_nn_lab.tracks.xlstm.cells import GateKind, MLSTMCell, SLSTMCell

BlockKind = Literal["slstm", "mlstm"]


class XLSTM(TokenSequenceModel):
    """A stack of pre-norm residual xLSTM blocks.

    Each block is ``x + cell(norm(x))``. The residual path is what lets a deeper stack
    train at all at this scale; it is present identically in the Transformer baseline, so
    it is not part of what is being compared.

    Attributes:
        block_kinds: The cell type of each block, in order.
    """

    block_norms: nn.ModuleList
    cells: nn.ModuleList

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 64,
        n_blocks: int = 2,
        block_kinds: tuple[BlockKind, ...] | None = None,
        heads: int = 2,
        input_gate: GateKind = "exponential",
        forget_gate: GateKind = "sigmoid",
    ) -> None:
        """Build an xLSTM.

        Args:
            vocab_size: Number of token ids.
            d_model: Model width.
            n_blocks: Number of blocks.
            block_kinds: Cell type per block. Defaults to all ``"mlstm"``.
            heads: Memory heads for ``mlstm`` blocks.
            input_gate: Input-gate kind. ``"sigmoid"`` is the gating ablation.
            forget_gate: Forget-gate kind.

        Raises:
            ValueError: If ``n_blocks`` is not positive or ``block_kinds`` has the wrong
                length.
        """

        super().__init__(vocab_size, d_model)
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")

        kinds: tuple[BlockKind, ...] = block_kinds or tuple(["mlstm"] * n_blocks)
        if len(kinds) != n_blocks:
            raise ValueError(f"block_kinds has length {len(kinds)}, expected {n_blocks}")

        self.block_kinds = kinds
        self.block_norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_blocks))
        self.cells = nn.ModuleList(
            SLSTMCell(d_model, d_model, input_gate=input_gate, forget_gate=forget_gate)
            if kind == "slstm"
            else MLSTMCell(d_model, heads=heads, input_gate=input_gate, forget_gate=forget_gate)
            for kind in kinds
        )

    @property
    def blocks(self) -> tuple[tuple[SLSTMCell | MLSTMCell, nn.LayerNorm], ...]:
        """Typed view of the ``(cell, norm)`` pairs.

        ``nn.ModuleList`` erases the element type, so the cell-specific interface
        (``initial_state``) is reached through this accessor rather than a cast at each
        call site.

        Returns:
            One ``(cell, norm)`` pair per block, in order.

        Raises:
            TypeError: If a registered module is not of the expected type.
        """

        pairs: list[tuple[SLSTMCell | MLSTMCell, nn.LayerNorm]] = []
        for cell, norm in zip(self.cells, self.block_norms, strict=True):
            if not isinstance(cell, SLSTMCell | MLSTMCell) or not isinstance(norm, nn.LayerNorm):
                raise TypeError("unexpected module registered in an xLSTM block")
            pairs.append((cell, norm))
        return tuple(pairs)

    def forward(self, tokens: Tensor) -> Tensor:
        """Run the stack over a full sequence.

        Args:
            tokens: Shape ``(B, T)``.

        Returns:
            Shape ``(B, T, V)`` logits.
        """

        hidden = self.encode(tokens)
        batch, seq_len, _ = hidden.shape

        for cell, norm in self.blocks:
            state = cell.initial_state(batch, device=hidden.device)
            normed = norm(hidden)
            outputs = []
            for step in range(seq_len):
                # Each step sees only positions <= step: causality is structural here,
                # not enforced by a mask that could be forgotten.
                output, state = cell(normed[:, step], state)
                outputs.append(output)
            hidden = hidden + torch.stack(outputs, dim=1)

        return self.decode(hidden)
