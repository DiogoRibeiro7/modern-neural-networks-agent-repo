r"""A patch dataset with known latent factors, split into content and nuisance.

Representation quality is usually inferred from downstream accuracy, which conflates the
representation with the probe and with the task. Here the generative factors are known, so
the question becomes direct: *does the representation contain the content factors, and has
it discarded the nuisance factors?* Both are measurable, and they are different questions —
a representation can score well on the first while failing the second, and only a nuisance
probe reveals it.

Each observation is a set of patches sharing one content vector:

.. math::

    x_p = \tanh\!\big(W_c\, c + W_n\, n_p\big) + \varepsilon_p,
    \qquad c \sim \mathcal{N}(0, I), \quad n_p \sim \mathcal{N}(0, I)

The content :math:`c` is **shared across every patch of a sample**; the nuisance
:math:`n_p` is **drawn independently per patch**. That asymmetry is the entire design, and
it is what makes the task well posed for a predictive architecture: content is exactly the
part of one patch that another patch can predict, and nuisance is exactly the part it
cannot. A model that predicts masked patch representations from visible ones is therefore
being pushed towards content and away from nuisance by the structure of the data, not by a
hand-written regularizer.

It also gives the invariance analysis a ground truth. "Invariance to nuisance
transformations" is often demonstrated by applying augmentations and observing that the
representation moves little; here the nuisance factors are known, so invariance is the
:math:`R^2` of a probe that tries to recover them — and a low value is evidence rather than
an impression.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class LatentDataset:
    """Observations with their generating factors recorded.

    Attributes:
        patches: Shape ``(N, n_patches, d_patch)`` observations.
        content: Shape ``(N, d_content)`` factors shared across a sample's patches.
        nuisance: Shape ``(N, n_patches, d_nuisance)`` per-patch factors.
    """

    patches: Tensor
    content: Tensor
    nuisance: Tensor

    def __post_init__(self) -> None:
        """Validate the dataset.

        Raises:
            ValueError: If the tensors disagree on sample or patch counts.
        """

        if self.patches.shape[0] != self.content.shape[0]:
            raise ValueError("patches and content must agree on sample count")
        if self.patches.shape[:2] != self.nuisance.shape[:2]:
            raise ValueError("patches and nuisance must agree on sample and patch counts")

    @property
    def n_samples(self) -> int:
        """Number of samples."""

        return int(self.patches.shape[0])

    @property
    def n_patches(self) -> int:
        """Patches per sample."""

        return int(self.patches.shape[1])

    @property
    def d_patch(self) -> int:
        """Width of one patch."""

        return int(self.patches.shape[2])

    def split(self, train_fraction: float = 0.7) -> tuple[LatentDataset, LatentDataset]:
        """Split into two disjoint parts.

        Args:
            train_fraction: Fraction of samples in the first part.

        Returns:
            ``(first, second)``.
        """

        cut = int(self.n_samples * train_fraction)
        return (
            LatentDataset(self.patches[:cut], self.content[:cut], self.nuisance[:cut]),
            LatentDataset(self.patches[cut:], self.content[cut:], self.nuisance[cut:]),
        )


def generate(
    *,
    n_samples: int,
    n_patches: int = 8,
    d_patch: int = 12,
    d_content: int = 4,
    d_nuisance: int = 3,
    noise: float = 0.05,
    seed: int = 0,
) -> LatentDataset:
    """Generate patches from shared content and per-patch nuisance factors.

    Args:
        n_samples: Number of samples.
        n_patches: Patches per sample.
        d_patch: Width of one patch.
        d_content: Number of content factors, shared within a sample.
        d_nuisance: Number of nuisance factors, independent per patch.
        noise: Standard deviation of additive observation noise.
        seed: Seed for the mixing matrices and every draw.

    Returns:
        The dataset.

    Raises:
        ValueError: If any size is not positive.
    """

    if min(n_samples, n_patches, d_patch, d_content, d_nuisance) <= 0:
        raise ValueError("all sizes must be positive")
    if noise < 0:
        raise ValueError("noise must be non-negative")

    generator = torch.Generator().manual_seed(seed)

    # Fixed mixing matrices: the same generative process across train and test, so this
    # measures what the representation captured, not generalization to a new process.
    content_mix = torch.randn((d_content, d_patch), generator=generator)
    nuisance_mix = torch.randn((d_nuisance, d_patch), generator=generator)

    content = torch.randn((n_samples, d_content), generator=generator)
    nuisance = torch.randn((n_samples, n_patches, d_nuisance), generator=generator)

    content_part = (content @ content_mix).unsqueeze(1)
    nuisance_part = nuisance @ nuisance_mix
    patches = torch.tanh(content_part + nuisance_part)
    patches = patches + noise * torch.randn(patches.shape, generator=generator)

    return LatentDataset(patches=patches, content=content, nuisance=nuisance)


def sample_masks(
    n_samples: int,
    n_patches: int,
    *,
    n_targets: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Split patches into a visible context and a masked target set, per sample.

    Args:
        n_samples: Number of samples.
        n_patches: Patches per sample.
        n_targets: Patches hidden from the encoder and predicted instead.
        generator: Seeded generator.

    Returns:
        ``(context_mask, target_mask)``, each shape ``(n_samples, n_patches)`` and boolean.
        They are complementary: every patch is either context or target, never both, which
        is what stops the model from reading the answer it is predicting.

    Raises:
        ValueError: If ``n_targets`` leaves no context or exceeds the patch count.
    """

    if not 1 <= n_targets < n_patches:
        raise ValueError(f"n_targets must lie in [1, {n_patches - 1}], got {n_targets}")

    scores = torch.rand((n_samples, n_patches), generator=generator)
    target_index = scores.argsort(dim=-1)[:, :n_targets]

    target_mask = torch.zeros((n_samples, n_patches), dtype=torch.bool)
    target_mask.scatter_(1, target_index, True)
    return ~target_mask, target_mask
