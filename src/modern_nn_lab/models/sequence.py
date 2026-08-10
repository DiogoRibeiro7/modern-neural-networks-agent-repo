"""Sequence-model scaffolding shared by more than one track.

Track packages must not import from each other, so the moment a second track needed the
same embedding/read-out contract and the same baselines, they moved here. Every model in
the repository that consumes token sequences shares one interface — ``(B, T)`` integer
tokens in, ``(B, T, V)`` logits out — so the shared training loop drives all of them
without special cases, and an apparent quality difference cannot come from a difference
in the surrounding scaffolding.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor, nn

BaselineKind = Literal["lstm", "gru"]

Builder = Callable[[int], nn.Module]
"""Maps a model width to a constructed model, for parameter-budget matching."""


class TokenSequenceModel(nn.Module):
    """Embedding, a recurrent or attentive body, and a tied-width readout.

    Shapes:
        - input: ``(B, T)`` integer token ids
        - output: ``(B, T, V)`` logits

    Attributes:
        vocab_size: Number of token ids ``V``.
        d_model: Model width ``D``.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """Set up the shared embedding and readout.

        Args:
            vocab_size: Number of token ids.
            d_model: Model width.

        Raises:
            ValueError: If either argument is not positive.
        """

        super().__init__()
        if vocab_size <= 0 or d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, vocab_size)

    def encode(self, tokens: Tensor) -> Tensor:
        """Map token ids to the body's input representation.

        Args:
            tokens: Shape ``(B, T)``.

        Returns:
            Shape ``(B, T, D)``.

        Raises:
            ValueError: If ``tokens`` is not two-dimensional.
        """

        if tokens.ndim != 2:
            raise ValueError(f"expected shape (B, T), got {tuple(tokens.shape)}")
        embedded: Tensor = self.embedding(tokens)
        return embedded

    def decode(self, hidden: Tensor) -> Tensor:
        """Map body outputs to logits.

        Args:
            hidden: Shape ``(B, T, D)``.

        Returns:
            Shape ``(B, T, V)``.
        """

        logits: Tensor = self.readout(self.norm(hidden))
        return logits


class RecurrentBaseline(TokenSequenceModel):
    """LSTM or GRU baseline using Torch's own fused implementation.

    Using the library implementation rather than a hand-written one is deliberate: the
    baseline should be the strong, conventional option, not a weakened re-implementation.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        kind: BaselineKind = "lstm",
    ) -> None:
        """Build the baseline.

        Args:
            vocab_size: Number of token ids.
            d_model: Model width.
            n_layers: Recurrent layers.
            kind: ``"lstm"`` or ``"gru"``.

        Raises:
            ValueError: If ``n_layers`` is not positive or ``kind`` is unknown.
        """

        super().__init__(vocab_size, d_model)
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if kind not in ("lstm", "gru"):
            raise ValueError(f"unknown baseline kind {kind!r}")

        factory = nn.LSTM if kind == "lstm" else nn.GRU
        self.kind = kind
        self.rnn = factory(d_model, d_model, num_layers=n_layers, batch_first=True)

    def forward(self, tokens: Tensor) -> Tensor:
        """Run the baseline.

        Args:
            tokens: Shape ``(B, T)``.

        Returns:
            Shape ``(B, T, V)`` logits.
        """

        hidden, _ = self.rnn(self.encode(tokens))
        return self.decode(hidden)


class CausalTransformer(TokenSequenceModel):
    """Small pre-norm causal Transformer baseline.

    Attention is masked so position ``t`` sees only positions ``<= t``. The mask is
    verified by the same causality test that covers the recurrent models, because a
    silently wrong mask is the classic way a sequence benchmark becomes meaningless.
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = 512,
        feedforward_multiplier: int = 2,
    ) -> None:
        """Build the baseline.

        Args:
            vocab_size: Number of token ids.
            d_model: Model width.
            n_layers: Transformer blocks.
            n_heads: Attention heads.
            max_len: Longest sequence the learned positional embedding supports.
            feedforward_multiplier: Hidden width of the feed-forward block, as a multiple
                of ``d_model``. Kept small so the parameter budget can be matched.

        Raises:
            ValueError: If any dimension is invalid.
        """

        super().__init__(vocab_size, d_model)
        if n_layers <= 0 or n_heads <= 0 or max_len <= 0:
            raise ValueError("n_layers, n_heads, and max_len must be positive")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")

        self.position = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_multiplier * d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, tokens: Tensor) -> Tensor:
        """Run the baseline with a causal mask.

        Args:
            tokens: Shape ``(B, T)``.

        Returns:
            Shape ``(B, T, V)`` logits.

        Raises:
            ValueError: If the sequence is longer than ``max_len``.
        """

        seq_len = tokens.shape[1]
        if seq_len > self.max_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_len {self.max_len}")

        positions = torch.arange(seq_len, device=tokens.device)
        hidden = self.encode(tokens) + self.position(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=tokens.device)
        return self.decode(self.encoder(hidden, mask=mask, is_causal=True))


def count_parameters(model: nn.Module) -> int:
    """Return the total number of parameters in ``model``.

    Args:
        model: Module to inspect.

    Returns:
        Parameter count, including frozen parameters.
    """

    return sum(parameter.numel() for parameter in model.parameters())


def match_width_to_budget(
    target_parameters: int,
    build: Builder,
    *,
    candidate_widths: tuple[int, ...] = tuple(range(8, 257, 4)),
) -> int:
    """Choose the model width whose parameter count is closest to ``target_parameters``.

    Recurrent and attentive bodies have very different parameter counts at equal width, so
    comparing them at equal width compares capacity rather than mechanism.

    Args:
        target_parameters: Parameter count to match.
        build: Maps a width to a constructed model.
        candidate_widths: Widths to consider.

    Returns:
        The best width from ``candidate_widths``.

    Raises:
        ValueError: If no candidate width can be constructed.
    """

    best_width: int | None = None
    best_gap = math.inf
    for width in candidate_widths:
        try:
            model = build(width)
        except ValueError:
            continue  # e.g. width not divisible by the head count
        gap = abs(count_parameters(model) - target_parameters)
        if gap < best_gap:
            best_gap, best_width = gap, width

    if best_width is None:
        raise ValueError("no candidate width produced a valid model")
    return best_width
