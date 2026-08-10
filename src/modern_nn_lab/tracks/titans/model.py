r"""The Memory-as-Gate (MAG) Titans variant, plus its short-term attention branch.

The source proposes three ways to incorporate the neural memory. This track implements
**Memory as a Gate** (source Section 4.2, equations 26-28):

.. math::

    \tilde{x} = [p_1 \ldots p_{N_p}] \Vert x, \qquad
    y = \mathrm{SWAttn}^*(\tilde{x}), \qquad
    o = y \otimes M(\tilde{x})

Sliding-window attention acts as a precise short-term memory; the neural memory acts as a
fading long-term memory; a gate combines them.

**Why this variant.** Memory as Context is the source's strongest performer, but it
segments the sequence and updates the memory *from the attention output*, so the two
memory systems are entangled and a per-token write/read diagnostic would not attribute
cleanly. Memory as a Layer stacks them, so the memory's contribution is only observable
after attention has processed it. MAG keeps both systems running over the same tokens and
combines them at the very end, which makes each branch separately removable and each
write separately measurable — exactly what this track's acceptance criterion (explicit
memory-write/read diagnostics) requires. The cost of the choice is that no result here
speaks to MAC, which is the variant the source recommends.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from modern_nn_lab.models.sequence import TokenSequenceModel
from modern_nn_lab.tracks.titans.config import TitansConfig
from modern_nn_lab.tracks.titans.memory import MemoryTrace, NeuralMemory


class SlidingWindowAttention(nn.Module):
    """Causal attention restricted to a fixed window of recent tokens.

    This is the *short-term* memory of the MAG design: exact within its window, blind
    beyond it. Its blindness is the point — it is what makes the long-term memory's
    contribution measurable on a task whose answer lies outside the window.

    Shapes:
        - input: ``(B, T, D)``
        - output: ``(B, T, D)``
    """

    def __init__(self, d_model: int, *, n_heads: int = 2, window: int = 8) -> None:
        """Build the attention branch.

        Args:
            d_model: Model width.
            n_heads: Attention heads.
            window: Number of past tokens visible, including the current one.

        Raises:
            ValueError: If the width is not divisible by the head count, or the window is
                not positive.
        """

        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        if window <= 0:
            raise ValueError("window must be positive")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.projection = nn.Linear(d_model, d_model)

    def band_mask(self, length: int, device: torch.device) -> Tensor:
        """Return the additive mask allowing only the last ``window`` positions.

        Args:
            length: Sequence length.
            device: Device for the mask.

        Returns:
            Shape ``(length, length)`` additive mask with ``-inf`` outside the band.
        """

        positions = torch.arange(length, device=device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        allowed = (distance >= 0) & (distance < self.window)
        # A large finite penalty, not -inf: softmax backward through -inf can produce
        # 0 * inf = NaN, and every row here has at least its own position unmasked.
        return torch.where(allowed, 0.0, torch.finfo(torch.float32).min / 4)

    def forward(self, sequence: Tensor) -> Tensor:
        """Apply sliding-window causal attention.

        Args:
            sequence: Shape ``(B, T, D)``.

        Returns:
            Shape ``(B, T, D)``.
        """

        batch, length, _ = sequence.shape
        qkv = self.qkv(sequence).view(batch, length, 3, self.n_heads, self.head_dim)
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4)

        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores + self.band_mask(length, sequence.device)
        attended = torch.softmax(scores, dim=-1) @ values

        merged = attended.transpose(1, 2).reshape(batch, length, self.d_model)
        result: Tensor = self.projection(merged)
        return result


class TitansMAG(TokenSequenceModel):
    """Titans Memory-as-Gate: sliding-window attention gated with a neural long-term memory.

    Shapes:
        - input: ``(B, T)`` integer token ids
        - output: ``(B, T, V)`` logits

    Attributes:
        config: The configuration used to build the model.
        memory: The neural long-term memory, or ``None`` in the short-term-only ablation.
        attention: The short-term branch, or ``None`` in the long-term-only ablation.
    """

    def __init__(self, vocab_size: int, config: TitansConfig) -> None:
        """Build the model.

        Args:
            vocab_size: Number of token ids.
            config: Track configuration.

        Raises:
            ValueError: If both branches are disabled.
        """

        super().__init__(vocab_size, config.d_model)
        if not config.use_short_term and not config.use_long_term:
            raise ValueError("at least one of the short-term and long-term branches must be on")

        self.config = config
        self.input_norm = nn.LayerNorm(config.d_model)

        # Persistent memory (eq. 19): learnable, data-independent tokens prepended to the
        # sequence. They carry task knowledge rather than context, and are never written.
        self.persistent = (
            nn.Parameter(torch.zeros(config.persistent_tokens, config.d_model))
            if config.persistent_tokens > 0
            else None
        )
        if self.persistent is not None:
            nn.init.normal_(self.persistent, std=1.0 / config.d_model**0.5)

        self.attention = (
            SlidingWindowAttention(config.d_model, n_heads=config.n_heads, window=config.window)
            if config.use_short_term
            else None
        )
        self.memory = (
            NeuralMemory(
                config.d_model,
                d_memory=config.d_memory,
                depth=config.memory_depth,
                updates_enabled=config.memory_updates,
                learning_rate_scale=config.learning_rate_scale,
                use_momentum=config.use_momentum,
                use_forgetting=config.use_forgetting,
            )
            if config.use_long_term
            else None
        )

        self.short_gain = nn.Parameter(torch.ones(config.d_model))
        self.long_gain = nn.Parameter(torch.ones(config.d_model))
        self.short_norm = nn.LayerNorm(config.d_model)
        self.long_norm = nn.LayerNorm(config.d_model)

    def forward(self, tokens: Tensor, trace: MemoryTrace | None = None) -> Tensor:
        """Run the model.

        Args:
            tokens: Shape ``(B, T)``.
            trace: Optional recorder for the memory's write/read diagnostics.

        Returns:
            Shape ``(B, T, V)`` logits.
        """

        hidden = self.input_norm(self.encode(tokens))
        prefix = 0
        if self.persistent is not None:
            prefix = self.persistent.shape[0]
            hidden = torch.cat(
                [self.persistent.unsqueeze(0).expand(hidden.shape[0], -1, -1), hidden], dim=1
            )

        # Equation 28: o = y (x) M(x~). A disabled branch contributes the identity, so the
        # two ablations are exactly "remove one factor" rather than "change the design".
        combined = torch.ones_like(hidden)
        if self.attention is not None:
            combined = combined * torch.nn.functional.silu(
                self.short_gain * self.short_norm(self.attention(hidden))
            )
        if self.memory is not None:
            combined = combined * torch.nn.functional.silu(
                self.long_gain * self.long_norm(self.memory(hidden, trace))
            )

        return self.decode(combined[:, prefix:] + hidden[:, prefix:])

    def memory_trace(self, tokens: Tensor) -> MemoryTrace:
        """Run the model once and return the memory's per-token diagnostics.

        Args:
            tokens: Shape ``(B, T)``.

        Returns:
            The populated :class:`MemoryTrace`.

        Raises:
            ValueError: If the model has no long-term memory.
        """

        if self.memory is None:
            raise ValueError("this configuration has no long-term memory to trace")

        trace = MemoryTrace()
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                self(tokens, trace)
        finally:
            self.train(was_training)
        return trace
