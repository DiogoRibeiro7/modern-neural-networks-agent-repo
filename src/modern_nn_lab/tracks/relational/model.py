r"""A compact relational encoder, and the homogeneous baseline it is compared against.

The prototype keeps rows as rows. It embeds each one with a projection chosen by its
*table*, passes messages along foreign keys towards the target entity, and reads out at the
entity. Nothing is pre-aggregated, so any sum or count the task needs must be learned from
the row set rather than handed over.

Message passing runs from children towards parents, which is the direction foreign keys
point:

.. math::

    h_p \leftarrow h_p + \phi\!\left(h_p,\;
    \frac{1}{|C(p)|}\sum_{c \in C(p)} g(\Delta t_c) \, \psi(h_c)\right)

Two rounds suffice for this schema, and that is not an arbitrary depth: one round moves a
linked row's attributes onto the event that references it, the second moves events onto the
entity. A one-round model can therefore express ``one_hop`` but not ``multi_hop``, which is
what makes the regime comparison diagnostic rather than decorative.

The time gate :math:`g(\Delta t)` is a learned function of elapsed time only. It cannot
recover a row the sampler withheld — gating happens before the model sees anything — so it
controls *how much* recent history counts, never *whether* future rows are visible.

Three flags remove one mechanism each, so every comparison in the report varies one thing:

``use_types``
    Off: one shared projection for every table. Tests whether typing rows matters.
``use_links``
    Off: mean-pool every row straight into the entity, ignoring foreign keys. This is the
    homogeneous-GNN baseline the prompt asks for.
``use_time``
    Off: the elapsed-time channel is zeroed and the gate is disabled. Tests whether the
    model is using time at all.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from modern_nn_lab.tracks.relational.config import RelationalConfig
from modern_nn_lab.tracks.relational.sampler import (
    CATEGORY_SLICE,
    DT_INDEX,
    MASK_INDEX,
    N_TYPES,
    NUMERIC_SLICE,
    PARENT_INDEX,
    ROW_WIDTH,
    TYPE_SLICE,
)

FEATURE_WIDTH = (
    1 + (NUMERIC_SLICE.stop - NUMERIC_SLICE.start) + (CATEGORY_SLICE.stop - CATEGORY_SLICE.start)
)
"""Elapsed time, numeric slots, and category slots — what a row encoder actually reads."""


def split_row_channels(rows: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Unpack the encoded row layout.

    Args:
        rows: Shape ``(B, R, ROW_WIDTH)``.

    Returns:
        ``(features, type_onehot, mask, parent)`` with shapes ``(B, R, FEATURE_WIDTH)``,
        ``(B, R, N_TYPES)``, ``(B, R)`` and ``(B, R)``.

    Raises:
        ValueError: If the final dimension is not the documented width.
    """

    if rows.shape[-1] != ROW_WIDTH:
        raise ValueError(f"expected rows of width {ROW_WIDTH}, got {rows.shape[-1]}")

    features = torch.cat(
        [rows[..., DT_INDEX : DT_INDEX + 1], rows[..., NUMERIC_SLICE], rows[..., CATEGORY_SLICE]],
        dim=-1,
    )
    return features, rows[..., TYPE_SLICE], rows[..., MASK_INDEX], rows[..., PARENT_INDEX]


class RelationalEncoder(nn.Module):
    """Encode a typed row set and predict the target entity's label.

    Attributes:
        config: The configuration used to build the model.
    """

    def __init__(self, config: RelationalConfig) -> None:
        """Build the model.

        Args:
            config: Track configuration.
        """

        super().__init__()
        self.config = config
        width = config.d_model

        n_encoders = N_TYPES if config.use_types else 1
        self.row_encoders = nn.ModuleList(
            nn.Sequential(nn.Linear(FEATURE_WIDTH, width), nn.GELU(), nn.Linear(width, width))
            for _ in range(n_encoders)
        )
        self.message = nn.Linear(width, width)
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)
        # A scalar gate on elapsed time: one slope, one offset. Small enough that a
        # measured effect is attributable to time and not to added capacity.
        self.time_gate = nn.Linear(1, 1)
        self.readout = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def encode_rows(self, features: Tensor, type_onehot: Tensor) -> Tensor:
        """Embed every row with the projection belonging to its table.

        Args:
            features: Shape ``(B, R, FEATURE_WIDTH)``.
            type_onehot: Shape ``(B, R, N_TYPES)``.

        Returns:
            Shape ``(B, R, d_model)``.
        """

        if not self.config.use_types:
            encoded: Tensor = self.row_encoders[0](features)
            return encoded

        stacked = torch.stack([encoder(features) for encoder in self.row_encoders], dim=2)
        return (stacked * type_onehot.unsqueeze(-1)).sum(dim=2)

    def propagate(self, hidden: Tensor, parent: Tensor, mask: Tensor, elapsed: Tensor) -> Tensor:
        """Send one round of messages from each row to the row it references.

        Args:
            hidden: Shape ``(B, R, d_model)``.
            parent: Shape ``(B, R)`` destination slot per row; negative means no parent.
            mask: Shape ``(B, R)`` row presence.
            elapsed: Shape ``(B, R)`` encoded elapsed time.

        Returns:
            Shape ``(B, R, d_model)`` updated states.
        """

        sends = (parent >= 0) & (mask > 0)
        destination = parent.clamp_min(0).to(torch.long)

        messages = self.message(hidden)
        if self.config.use_time:
            messages = messages * torch.sigmoid(self.time_gate(elapsed.unsqueeze(-1)))
        messages = messages * sends.unsqueeze(-1)

        index = destination.unsqueeze(-1).expand_as(messages)
        gathered = torch.zeros_like(hidden).scatter_add_(1, index, messages)
        counts = torch.zeros_like(mask).scatter_add_(1, destination, sends.float())
        pooled = gathered / counts.clamp_min(1.0).unsqueeze(-1)

        updated: Tensor = self.norm(hidden + self.update(torch.cat([hidden, pooled], dim=-1)))
        return updated

    def pool_flat(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """Mean-pool every present row into the entity slot, ignoring foreign keys.

        This is the homogeneous baseline: the rows are still there, but which row points at
        which is discarded, so a multi-hop path is indistinguishable from two unrelated
        rows sitting in the same bag.

        Args:
            hidden: Shape ``(B, R, d_model)``.
            mask: Shape ``(B, R)``.

        Returns:
            Shape ``(B, d_model)``.
        """

        weights = mask.unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        root = hidden[:, 0]
        merged: Tensor = self.norm(root + self.update(torch.cat([root, pooled], dim=-1)))
        return merged

    def forward(self, rows: Tensor) -> Tensor:
        """Predict a logit per prediction point.

        Args:
            rows: Shape ``(B, R, ROW_WIDTH)`` encoded neighbourhoods.

        Returns:
            Shape ``(B,)`` logits.
        """

        features, type_onehot, mask, parent = split_row_channels(rows)
        elapsed = features[..., 0]
        if not self.config.use_time:
            features = torch.cat([torch.zeros_like(features[..., :1]), features[..., 1:]], dim=-1)
            elapsed = torch.zeros_like(elapsed)

        hidden = self.encode_rows(features, type_onehot) * mask.unsqueeze(-1)

        if self.config.use_links:
            for _ in range(self.config.n_rounds):
                hidden = self.propagate(hidden, parent, mask, elapsed)
            entity = hidden[:, 0]
        else:
            entity = self.pool_flat(hidden, mask)

        logits: Tensor = self.readout(entity).squeeze(-1)
        return logits


class TargetOnlyModel(nn.Module):
    """Predict from the target entity's own columns, ignoring every related table.

    The floor. Any relational claim has to clear this, and on ``cold_start`` it is close to
    the ceiling as well, because there is no history to read.
    """

    def __init__(self, config: RelationalConfig) -> None:
        """Build the model.

        Args:
            config: Track configuration.
        """

        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(FEATURE_WIDTH, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 1),
        )

    def forward(self, rows: Tensor) -> Tensor:
        """Predict a logit per prediction point.

        Args:
            rows: Shape ``(B, R, ROW_WIDTH)``.

        Returns:
            Shape ``(B,)`` logits.
        """

        features, _, _, _ = split_row_channels(rows)
        logits: Tensor = self.network(features[:, 0]).squeeze(-1)
        return logits
