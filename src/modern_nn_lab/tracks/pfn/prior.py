r"""Task priors: the distribution a Prior-Fitted Network is fitted to.

A PFN is not trained on a dataset. It is trained on a **distribution over datasets**, and
at inference it is handed a small labelled context and asked about queries without any
gradient step. So the prior *is* the training data, and changing it changes what the model
believes before it sees anything — which is the property this track measures.

Each prior samples a fresh task, then draws a context set and a query set from that task:

.. math::

    f \sim p(\mathcal{T}), \qquad
    (x_i, y_i)_{i \le n} \sim f, \qquad
    (x_q, y_q)_{q \le m} \sim f

The three priors here are deliberately nested so that "in-prior" and "out-of-prior" are
structural rather than a matter of degree:

- :class:`LinearPrior` — labels are a linear separator's sign.
- :class:`MLPPrior` — labels come from a random two-layer network, so decision boundaries
  are curved and a linear-fitted PFN is genuinely out of prior.
- :class:`XORPrior` — labels are a parity of thresholded coordinates, which no linear
  boundary can express at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class TaskBatch:
    """A batch of sampled tasks, each with its own context and query set.

    Attributes:
        context_inputs: Shape ``(B, n_context, d)``.
        context_labels: Shape ``(B, n_context)`` integer labels.
        query_inputs: Shape ``(B, n_query, d)``.
        query_labels: Shape ``(B, n_query)`` integer labels.
    """

    context_inputs: Tensor
    context_labels: Tensor
    query_inputs: Tensor
    query_labels: Tensor

    @property
    def n_features(self) -> int:
        """Input width."""

        return int(self.context_inputs.shape[-1])

    @property
    def n_context(self) -> int:
        """Context set size."""

        return int(self.context_inputs.shape[1])


class TaskPrior(ABC):
    """A distribution over supervised tasks.

    Attributes:
        n_features: Input width.
        label_noise: Probability of flipping each label after it is generated.
    """

    def __init__(self, *, n_features: int = 4, label_noise: float = 0.0) -> None:
        """Create a prior.

        Args:
            n_features: Input width.
            label_noise: Probability of flipping a label. Applied to context *and* query
                labels, so it is irreducible error rather than a train/test mismatch.

        Raises:
            ValueError: If ``n_features`` is not positive or the noise is outside [0, 1].
        """

        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if not 0.0 <= label_noise <= 1.0:
            raise ValueError("label_noise must lie in [0, 1]")
        self.n_features = n_features
        self.label_noise = label_noise

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier recorded with results."""

    @abstractmethod
    def label(self, inputs: Tensor, generator: torch.Generator) -> Tensor:
        """Sample one task's labelling function and apply it.

        Args:
            inputs: Shape ``(N, d)`` points from a single task.
            generator: Seeded generator.

        Returns:
            Shape ``(N,)`` integer labels in ``{0, 1}``.
        """

    def sample(
        self,
        *,
        batch_size: int,
        n_context: int,
        n_query: int,
        generator: torch.Generator,
    ) -> TaskBatch:
        """Draw a batch of independent tasks.

        Args:
            batch_size: Number of tasks.
            n_context: Labelled examples given to the model.
            n_query: Points the model must label.
            generator: Seeded generator.

        Returns:
            A :class:`TaskBatch`.

        Raises:
            ValueError: If any size is not positive.
        """

        if min(batch_size, n_context, n_query) <= 0:
            raise ValueError("batch_size, n_context, and n_query must be positive")

        contexts, context_labels, queries, query_labels = [], [], [], []
        for _ in range(batch_size):
            points = torch.randn((n_context + n_query, self.n_features), generator=generator)
            labels = self.label(points, generator)

            if self.label_noise > 0:
                flips = torch.rand(labels.shape, generator=generator) < self.label_noise
                labels = torch.where(flips, 1 - labels, labels)

            contexts.append(points[:n_context])
            context_labels.append(labels[:n_context])
            queries.append(points[n_context:])
            query_labels.append(labels[n_context:])

        return TaskBatch(
            context_inputs=torch.stack(contexts),
            context_labels=torch.stack(context_labels),
            query_inputs=torch.stack(queries),
            query_labels=torch.stack(query_labels),
        )


class LinearPrior(TaskPrior):
    """Labels are the sign of a random linear function."""

    @property
    def name(self) -> str:
        """Identifier recorded with results."""

        return "linear"

    def label(self, inputs: Tensor, generator: torch.Generator) -> Tensor:
        """Apply a random linear separator through a random offset.

        Args:
            inputs: Shape ``(N, d)``.
            generator: Seeded generator.

        Returns:
            Shape ``(N,)`` labels.
        """

        weights = torch.randn((self.n_features,), generator=generator)
        bias = torch.randn((), generator=generator) * 0.5
        return ((inputs @ weights + bias) > 0).long()


class ImbalancedPrior(LinearPrior):
    """A linear prior whose classes are skewed to a fixed positive rate.

    The separator is the same object as in :class:`LinearPrior`; only the threshold moves.
    Thresholding at a quantile rather than adding a fixed bias makes the imbalance exact
    per task, so the study varies class balance and nothing else.

    Attributes:
        positive_rate: Target fraction of positive labels.
    """

    def __init__(
        self, *, n_features: int = 4, label_noise: float = 0.0, positive_rate: float = 0.2
    ) -> None:
        """Create the prior.

        Args:
            n_features: Input width.
            label_noise: Label-flip probability.
            positive_rate: Target fraction of positive labels, in ``(0, 1)``.

        Raises:
            ValueError: If ``positive_rate`` is not strictly between 0 and 1.
        """

        super().__init__(n_features=n_features, label_noise=label_noise)
        if not 0.0 < positive_rate < 1.0:
            raise ValueError("positive_rate must lie strictly between 0 and 1")
        self.positive_rate = positive_rate

    @property
    def name(self) -> str:
        """Identifier recorded with results."""

        return f"imbalanced{self.positive_rate:g}"

    def label(self, inputs: Tensor, generator: torch.Generator) -> Tensor:
        """Apply a random separator thresholded at the requested quantile.

        Args:
            inputs: Shape ``(N, d)``.
            generator: Seeded generator.

        Returns:
            Shape ``(N,)`` labels with roughly ``positive_rate`` positives.
        """

        weights = torch.randn((self.n_features,), generator=generator)
        scores = inputs @ weights
        threshold = torch.quantile(scores, 1.0 - self.positive_rate)
        return (scores > threshold).long()


class MLPPrior(TaskPrior):
    """Labels come from a random two-layer network, so boundaries are curved.

    Attributes:
        hidden: Width of the sampled network's hidden layer.
    """

    def __init__(self, *, n_features: int = 4, label_noise: float = 0.0, hidden: int = 8) -> None:
        """Create the prior.

        Args:
            n_features: Input width.
            label_noise: Label-flip probability.
            hidden: Hidden width of the sampled labelling network.

        Raises:
            ValueError: If ``hidden`` is not positive.
        """

        super().__init__(n_features=n_features, label_noise=label_noise)
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        self.hidden = hidden

    @property
    def name(self) -> str:
        """Identifier recorded with results."""

        return "mlp"

    def label(self, inputs: Tensor, generator: torch.Generator) -> Tensor:
        """Apply a random tanh network's sign.

        Args:
            inputs: Shape ``(N, d)``.
            generator: Seeded generator.

        Returns:
            Shape ``(N,)`` labels.
        """

        first = torch.randn((self.n_features, self.hidden), generator=generator)
        second = torch.randn((self.hidden,), generator=generator)
        hidden = torch.tanh(inputs @ first)
        # Centre the logit so the classes stay roughly balanced across sampled tasks.
        logits = hidden @ second
        return (logits > logits.median()).long()


class XORPrior(TaskPrior):
    """Labels are the parity of thresholded coordinates: no linear boundary can express it."""

    @property
    def name(self) -> str:
        """Identifier recorded with results."""

        return "xor"

    def label(self, inputs: Tensor, generator: torch.Generator) -> Tensor:
        """Apply the parity of two randomly chosen coordinates' signs.

        Args:
            inputs: Shape ``(N, d)``.
            generator: Seeded generator.

        Returns:
            Shape ``(N,)`` labels.

        Raises:
            ValueError: If the prior has fewer than two features.
        """

        if self.n_features < 2:
            raise ValueError("XORPrior needs at least two features")
        picks = torch.randperm(self.n_features, generator=generator)[:2]
        signs = (inputs[:, picks] > 0).long()
        return (signs[:, 0] ^ signs[:, 1]).long()


PRIORS: dict[str, type[TaskPrior]] = {
    "linear": LinearPrior,
    "imbalanced": ImbalancedPrior,
    "mlp": MLPPrior,
    "xor": XORPrior,
}
"""Registry used by the experiment suite to build priors by name."""


def build_prior(
    name: str, *, n_features: int, label_noise: float = 0.0, **options: float
) -> TaskPrior:
    """Construct a prior by name.

    Args:
        name: One of ``"linear"``, ``"imbalanced"``, ``"mlp"``, ``"xor"``.
        n_features: Input width.
        label_noise: Label-flip probability.
        **options: Prior-specific settings, such as ``positive_rate`` for
            ``"imbalanced"`` or ``hidden`` for ``"mlp"``.

    Returns:
        The prior.

    Raises:
        KeyError: If the name is unknown.
        TypeError: If an option does not apply to the named prior.
    """

    if name not in PRIORS:
        raise KeyError(f"unknown prior {name!r}; available: {sorted(PRIORS)}")
    return PRIORS[name](n_features=n_features, label_noise=label_noise, **options)
