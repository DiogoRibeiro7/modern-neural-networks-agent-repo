r"""The JEPA, its anti-collapse mechanism, and the baselines it is compared against.

## What exactly is predicted

Patches are split into a visible **context** set and a hidden **target** set. The context
encoder embeds the visible patches and pools them into one context vector; the predictor is
then asked, for each hidden patch, to produce that patch's *representation* — not its pixels:

.. math::

    \hat z_p = g_\phi\big(\bar h_{\text{context}}\big),
    \qquad z_p = \operatorname{sg}\big[f_\xi(x_p)\big],
    \qquad \mathcal{L} = \frac{1}{|T|}\sum_{p \in T} \lVert \hat z_p - z_p \rVert^2

The target :math:`z_p` comes from a *separate* target encoder :math:`f_\xi`, and
:math:`\operatorname{sg}` is a stop-gradient. Because content is shared across a sample's
patches and nuisance is not, the only predictable part of a target patch is its content —
so the objective pushes the representation towards content and away from nuisance without
anyone writing that down as a penalty.

## Why trivial collapse is or is not prevented

**The loss alone does not prevent it.** A constant encoder :math:`f(x) = k` makes the
prediction problem trivial: the predictor outputs :math:`k` and the loss reaches zero. This
is not a hypothetical — :class:`JEPA` with ``anti_collapse="none"`` is exactly that failure
mode, kept as an ablation so the report can show the collapse rather than describe it.

**What prevents it here** is the pairing of a stop-gradient with a target encoder whose
weights are an exponential moving average of the context encoder's:

.. math:: \xi \leftarrow \tau\,\xi + (1 - \tau)\,\theta

No gradient flows into :math:`\xi`, so the loss cannot reduce itself by moving the target.
The target is a slowly-moving copy of the online encoder, which makes the objective a
moving one: the online encoder chases a target that is itself a lagged version of the
online encoder. This does not make collapse impossible — it is an empirical stabilizer, not
a proof — and the report treats "collapse did not occur" as a measurement rather than a
guarantee.

``anti_collapse="variance"`` offers a different mechanism for contrast: an explicit hinge
penalty that keeps each representation dimension's standard deviation above a floor. It is
a direct constraint rather than an optimization-dynamics argument, and comparing the two is
one of the ablations.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

from modern_nn_lab.tracks.jepa.config import JEPAConfig


def build_encoder(d_in: int, d_hidden: int, d_out: int, n_layers: int) -> nn.Module:
    """Build an MLP encoder applied to each patch independently.

    Args:
        d_in: Patch width.
        d_hidden: Hidden width.
        d_out: Representation width.
        n_layers: Hidden layers.

    Returns:
        The encoder.
    """

    layers: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.GELU()]
    for _ in range(max(n_layers - 1, 0)):
        layers += [nn.Linear(d_hidden, d_hidden), nn.GELU()]
    layers.append(nn.Linear(d_hidden, d_out))
    return nn.Sequential(*layers)


class JEPA(nn.Module):
    """Predict masked patch representations from visible ones.

    Attributes:
        config: The configuration used to build the model.
        encoder: The online context encoder, whose representation is probed.
        target_encoder: The encoder producing prediction targets, or ``None`` when the
            online encoder is reused.
        predictor: Maps a pooled context to a predicted target representation.
    """

    def __init__(self, config: JEPAConfig) -> None:
        """Build the model.

        Args:
            config: Track configuration.
        """

        super().__init__()
        self.config = config
        self.encoder = build_encoder(
            config.d_patch, config.d_hidden, config.d_representation, config.n_encoder_layers
        )

        if config.anti_collapse == "ema":
            # A detached copy: it is updated by assignment, never by gradient.
            self.target_encoder: nn.Module | None = copy.deepcopy(self.encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        else:
            self.target_encoder = None

        self.predictor = self._build_predictor(config)

    @staticmethod
    def _build_predictor(config: JEPAConfig) -> nn.Module:
        """Build the predictor at the configured capacity.

        A depth of zero gives the identity, which is the sharpest capacity ablation: the
        context representation must then *be* the target representation, with nothing in
        between to absorb the difference.

        Args:
            config: Track configuration.

        Returns:
            The predictor.
        """

        if config.n_predictor_layers == 0:
            return nn.Identity()

        width = config.d_predictor
        layers: list[nn.Module] = [nn.Linear(config.d_representation, width), nn.GELU()]
        for _ in range(config.n_predictor_layers - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers.append(nn.Linear(width, config.d_representation))
        return nn.Sequential(*layers)

    def encode(self, patches: Tensor) -> Tensor:
        """Embed every patch with the online encoder.

        Args:
            patches: Shape ``(N, P, d_patch)``.

        Returns:
            Shape ``(N, P, d_representation)``.
        """

        encoded: Tensor = self.encoder(patches)
        return encoded

    def represent(self, patches: Tensor) -> Tensor:
        """Return one representation per sample, by pooling over all patches.

        This is what the linear probes see. Pooling over *all* patches rather than a
        context subset keeps the probe independent of the masking used during training.

        Args:
            patches: Shape ``(N, P, d_patch)``.

        Returns:
            Shape ``(N, d_representation)``.
        """

        return self.encode(patches).mean(dim=1)

    def encode_targets(self, patches: Tensor) -> Tensor:
        """Embed patches with the target encoder, under a stop-gradient.

        Args:
            patches: Shape ``(N, P, d_patch)``.

        Returns:
            Shape ``(N, P, d_representation)``, detached when an EMA target is used.
        """

        if self.target_encoder is None:
            # No separate target: the online encoder supplies the targets, and gradient
            # flows through them. This is the configuration that collapses.
            online: Tensor = self.encoder(patches)
            return online

        with torch.no_grad():
            target: Tensor = self.target_encoder(patches)
        return target

    @torch.no_grad()
    def update_target(self) -> None:
        """Move the target encoder towards the online encoder by one EMA step."""

        if self.target_encoder is None:
            return
        decay = self.config.ema_decay
        for target, online in zip(
            self.target_encoder.parameters(), self.encoder.parameters(), strict=True
        ):
            target.mul_(decay).add_(online.detach(), alpha=1.0 - decay)

    def forward(self, patches: Tensor, context_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Predict the representations of every patch from the visible context.

        The target mask is not needed here and is deliberately not accepted: predictions
        are produced for all patches and :func:`jepa_loss` selects the masked ones. Keeping
        the selection in one place means the model cannot accidentally see which patches it
        will be scored on.

        Args:
            patches: Shape ``(N, P, d_patch)``.
            context_mask: Shape ``(N, P)`` boolean, true for visible patches.

        Returns:
            ``(predicted, targets)``, both shape ``(N, P, d_representation)``.

        Raises:
            ValueError: If a sample has no visible context patch.
        """

        visible = context_mask.unsqueeze(-1).float()
        counts = visible.sum(dim=1)
        if bool((counts == 0).any()):
            raise ValueError("every sample needs at least one context patch")

        encoded = self.encode(patches)
        pooled = (encoded * visible).sum(dim=1) / counts
        predicted = self.predictor(pooled).unsqueeze(1).expand_as(encoded)
        targets = self.encode_targets(patches)
        return predicted, targets


def jepa_loss(
    predicted: Tensor,
    targets: Tensor,
    target_mask: Tensor,
    *,
    variance_weight: float = 0.0,
    variance_floor: float = 1.0,
) -> Tensor:
    """Mean squared error over masked patches, with an optional variance floor.

    Args:
        predicted: Shape ``(N, P, d)`` predictions.
        targets: Shape ``(N, P, d)`` targets.
        target_mask: Shape ``(N, P)`` boolean, true where a prediction is scored.
        variance_weight: Weight on the anti-collapse hinge; ``0`` disables it.
        variance_floor: Standard deviation each dimension is pushed above.

    Returns:
        Scalar loss.
    """

    mask = target_mask.unsqueeze(-1).float()
    error = ((predicted - targets).pow(2) * mask).sum() / mask.sum().clamp_min(1.0)

    if variance_weight > 0:
        flat = targets.reshape(-1, targets.shape[-1])
        deviation = flat.var(dim=0).clamp_min(1e-8).sqrt()
        error = error + variance_weight * torch.relu(variance_floor - deviation).mean()

    return error


class Autoencoder(nn.Module):
    """Reconstruct raw patches — the baseline that cannot collapse.

    An autoencoder has no trivial solution: a constant code reconstructs nothing, so the
    loss itself keeps the representation informative. That makes it the right comparison
    for whether predicting *representations* buys anything over predicting *observations*,
    and it is expected to retain nuisance factors that a JEPA should discard.

    Attributes:
        encoder: Patch encoder.
        decoder: Patch decoder.
    """

    def __init__(self, config: JEPAConfig) -> None:
        """Build the model.

        Args:
            config: Track configuration.
        """

        super().__init__()
        self.config = config
        self.encoder = build_encoder(
            config.d_patch, config.d_hidden, config.d_representation, config.n_encoder_layers
        )
        self.decoder = build_encoder(
            config.d_representation, config.d_hidden, config.d_patch, config.n_encoder_layers
        )

    def represent(self, patches: Tensor) -> Tensor:
        """Return one representation per sample.

        Args:
            patches: Shape ``(N, P, d_patch)``.

        Returns:
            Shape ``(N, d_representation)``.
        """

        encoded: Tensor = self.encoder(patches)
        return encoded.mean(dim=1)

    def forward(self, patches: Tensor) -> Tensor:
        """Reconstruct the patches.

        Args:
            patches: Shape ``(N, P, d_patch)``.

        Returns:
            Shape ``(N, P, d_patch)``.
        """

        reconstructed: Tensor = self.decoder(self.encoder(patches))
        return reconstructed


class ContrastiveLearner(nn.Module):
    """Pull patches of a sample together and push different samples apart.

    The other standard answer to collapse: an explicit repulsion term. Two patches of the
    same sample share content and differ in nuisance, so treating them as a positive pair
    targets the same structure the JEPA does, by a different route.

    Attributes:
        encoder: Patch encoder.
        temperature: Softmax temperature of the InfoNCE objective.
    """

    def __init__(self, config: JEPAConfig) -> None:
        """Build the model.

        Args:
            config: Track configuration.
        """

        super().__init__()
        self.config = config
        self.encoder = build_encoder(
            config.d_patch, config.d_hidden, config.d_representation, config.n_encoder_layers
        )
        self.temperature = config.temperature

    def represent(self, patches: Tensor) -> Tensor:
        """Return one representation per sample.

        Args:
            patches: Shape ``(N, P, d_patch)``.

        Returns:
            Shape ``(N, d_representation)``.
        """

        encoded: Tensor = self.encoder(patches)
        return encoded.mean(dim=1)

    def forward(self, patches: Tensor) -> Tensor:
        """Compute the InfoNCE loss over two patches per sample.

        Args:
            patches: Shape ``(N, P, d_patch)`` with at least two patches.

        Returns:
            Scalar loss.

        Raises:
            ValueError: If there are fewer than two patches.
        """

        if patches.shape[1] < 2:
            raise ValueError("contrastive learning needs at least two patches per sample")

        encoded = self.encoder(patches)
        first = nn.functional.normalize(encoded[:, 0], dim=-1)
        second = nn.functional.normalize(encoded[:, 1], dim=-1)

        logits = first @ second.T / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        forward_loss = nn.functional.cross_entropy(logits, labels)
        backward_loss = nn.functional.cross_entropy(logits.T, labels)
        return 0.5 * (forward_loss + backward_loss)


class RawFeatures(nn.Module):
    """Pool the raw patches, with no learning at all.

    The floor. Any claim that a learned representation is useful has to clear the same
    numbers computed on the observations themselves, and on a task this simple that is a
    higher bar than it sounds.
    """

    def __init__(self, config: JEPAConfig) -> None:
        """Build the model.

        Args:
            config: Track configuration.
        """

        super().__init__()
        self.config = config

    def represent(self, patches: Tensor) -> Tensor:
        """Return the mean patch.

        Args:
            patches: Shape ``(N, P, d_patch)``.

        Returns:
            Shape ``(N, d_patch)``.
        """

        return patches.mean(dim=1)
