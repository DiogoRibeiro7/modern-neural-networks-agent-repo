r"""A Prior-Fitted Network: in-context prediction with no per-dataset fitting.

The model reads a labelled context and a set of unlabelled queries as **one sequence**, and
predicts every query's label in a single forward pass. Nothing is fitted per dataset — the
only training that ever happens is prior fitting, done once, over sampled tasks.

Two structural properties define the architecture, and both are asserted in the tests
rather than assumed:

1. **Queries cannot see each other.** Each query attends to the context only. Otherwise a
   query could read another query's position and the model would be doing transduction
   over the whole test set rather than answering each question independently.
2. **Order does not matter.** There is no positional encoding, so permuting the context
   leaves predictions unchanged. A dataset is a *set*, and a model that depended on row
   order would be exploiting an artefact.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from modern_nn_lab.tracks.pfn.config import PFNConfig


class PriorFittedNetwork(nn.Module):
    """A compact PFN over binary-classification tasks.

    Shapes:
        - context inputs: ``(B, n_context, d)``; context labels: ``(B, n_context)``
        - query inputs: ``(B, n_query, d)``
        - output: ``(B, n_query, 2)`` logits

    Attributes:
        config: The configuration used to build the model.
    """

    def __init__(self, config: PFNConfig) -> None:
        """Build the model.

        Args:
            config: Track configuration.
        """

        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.n_features, config.d_model)
        # Labels enter as an embedding, with a dedicated row for "unknown" so a query is
        # represented as a point whose label has not been revealed.
        self.label_embedding = nn.Embedding(config.n_classes + 1, config.d_model)
        self.unknown_label = config.n_classes

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward * config.d_model,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.readout = nn.Linear(config.d_model, config.n_classes)

    def attention_mask(self, n_context: int, n_query: int, device: torch.device) -> Tensor:
        """Build the PFN attention mask.

        Context positions attend to context positions. Query positions attend to context
        positions **and to themselves**, but never to another query.

        Args:
            n_context: Number of context positions.
            n_query: Number of query positions.
            device: Device for the mask.

        Returns:
            Shape ``(n_context + n_query, n_context + n_query)`` additive float mask.
        """

        total = n_context + n_query
        allowed = torch.zeros((total, total), dtype=torch.bool, device=device)
        allowed[:, :n_context] = True  # everything may read the context
        queries = torch.arange(n_context, total, device=device)
        allowed[queries, queries] = True  # and each query may read itself
        allowed[:n_context, n_context:] = False  # context never reads queries

        # A large finite penalty rather than -inf: softmax backward through -inf can
        # produce NaN, and every row here has at least one permitted position.
        return torch.where(allowed, 0.0, torch.finfo(torch.float32).min / 4)

    def forward(
        self, context_inputs: Tensor, context_labels: Tensor, query_inputs: Tensor
    ) -> Tensor:
        """Predict query labels in one pass, conditioned on the context.

        Args:
            context_inputs: Shape ``(B, n_context, d)``.
            context_labels: Shape ``(B, n_context)`` integer labels.
            query_inputs: Shape ``(B, n_query, d)``.

        Returns:
            Shape ``(B, n_query, n_classes)`` logits.

        Raises:
            ValueError: If the shapes are inconsistent with the configuration.
        """

        if context_inputs.ndim != 3 or query_inputs.ndim != 3:
            raise ValueError("context and query inputs must have shape (B, n, d)")
        if context_inputs.shape[-1] != self.config.n_features:
            raise ValueError(
                f"expected {self.config.n_features} features, got {context_inputs.shape[-1]}"
            )
        if context_inputs.shape[:2] != context_labels.shape:
            raise ValueError("context inputs and labels must agree on batch and length")

        n_context = context_inputs.shape[1]
        n_query = query_inputs.shape[1]

        context_tokens = self.input_projection(context_inputs) + self.label_embedding(
            context_labels
        )
        unknown = torch.full(
            query_inputs.shape[:2], self.unknown_label, dtype=torch.long, device=query_inputs.device
        )
        query_tokens = self.input_projection(query_inputs) + self.label_embedding(unknown)

        sequence = torch.cat([context_tokens, query_tokens], dim=1)
        mask = self.attention_mask(n_context, n_query, sequence.device)
        encoded = self.encoder(sequence, mask=mask)

        logits: Tensor = self.readout(self.norm(encoded[:, n_context:]))
        return logits

    @torch.no_grad()
    def predict_proba(
        self, context_inputs: Tensor, context_labels: Tensor, query_inputs: Tensor
    ) -> Tensor:
        """Return class probabilities without any gradient step.

        This is the whole point of the architecture: prediction on a new dataset is a
        forward pass, not an optimization. ``test_prediction_does_not_change_parameters``
        asserts that no parameter moves.

        Args:
            context_inputs: Shape ``(B, n_context, d)``.
            context_labels: Shape ``(B, n_context)``.
            query_inputs: Shape ``(B, n_query, d)``.

        Returns:
            Shape ``(B, n_query, n_classes)`` probabilities.
        """

        was_training = self.training
        self.eval()
        try:
            logits = self(context_inputs, context_labels, query_inputs)
            return torch.softmax(logits, dim=-1)
        finally:
            self.train(was_training)

    def extra_repr(self) -> str:
        """Return a compact description for ``print(model)``."""

        return (
            f"n_features={self.config.n_features}, d_model={self.config.d_model}, "
            f"n_layers={self.config.n_layers}, n_classes={self.config.n_classes}"
        )
