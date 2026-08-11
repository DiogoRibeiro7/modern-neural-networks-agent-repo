"""JEPA: predicting representations of masked inputs rather than reconstructing them.

The claim under test is that a model can learn a useful representation by predicting *its
own* embeddings of hidden parts of an input. The immediate objection is that such an
objective has a trivial solution — a constant encoder makes every prediction exact — so the
whole track is organized around whether collapse happens and how it is prevented.

Two commitments follow from that, and both are structural rather than editorial:

- **Every collapse metric is independent of the training loss.** Representation variance and
  effective rank are computed on held-out data and never appear in any objective, so a model
  cannot do well on them by optimizing them.
- **The generative factors are known**, and split into content shared across a sample's
  patches and nuisance drawn independently per patch. "Contains the content" and "discarded
  the nuisance" are then two separate measurements rather than one impression, and the
  second is what a representation that memorizes everything fails.
"""

from modern_nn_lab.tracks.jepa.config import AntiCollapse, JEPAConfig, JEPAExperimentConfig
from modern_nn_lab.tracks.jepa.data import LatentDataset, generate, sample_masks
from modern_nn_lab.tracks.jepa.metrics import (
    collapse_report,
    effective_rank,
    linear_probe,
    normalized_effective_rank,
    representation_variance,
)
from modern_nn_lab.tracks.jepa.model import (
    JEPA,
    Autoencoder,
    ContrastiveLearner,
    RawFeatures,
    build_encoder,
    jepa_loss,
)

__all__ = [
    "JEPA",
    "AntiCollapse",
    "Autoencoder",
    "ContrastiveLearner",
    "JEPAConfig",
    "JEPAExperimentConfig",
    "LatentDataset",
    "RawFeatures",
    "build_encoder",
    "collapse_report",
    "effective_rank",
    "generate",
    "jepa_loss",
    "linear_probe",
    "normalized_effective_rank",
    "representation_variance",
    "sample_masks",
]
