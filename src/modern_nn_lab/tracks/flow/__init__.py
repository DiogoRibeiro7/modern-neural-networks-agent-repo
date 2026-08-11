"""Flow matching: learning a velocity field that transports one distribution onto another.

Built from first principles on two-dimensional distributions, as the track prompt requires,
and deliberately not extended to a high-dimensional model — at that size the interesting
question becomes engineering, and the property this track exists to examine would be harder
to see rather than easier.

The acceptance criterion is to separate **vector-field approximation error** from **ODE
discretization error**. Sample quality alone confounds them: a poor sample set could mean
the network learned the wrong field or that the solver took too few steps, and no scatter
plot distinguishes those. :mod:`~modern_nn_lab.tracks.flow.analytic` resolves it by giving
the Gaussian-to-Gaussian case a closed-form marginal velocity field, so each error can be
measured with the other held at zero.
"""

from modern_nn_lab.tracks.flow.analytic import (
    GaussianEndpoints,
    marginal_velocity,
    projection_residual,
)
from modern_nn_lab.tracks.flow.config import FlowConfig, FlowExperimentConfig
from modern_nn_lab.tracks.flow.data import (
    DATASETS,
    MIXTURE_CENTRES,
    Dataset,
    energy_distance,
    mode_coverage,
    sample_source,
    sample_target,
)
from modern_nn_lab.tracks.flow.field import TimeEmbedding, VectorField, flow_matching_loss
from modern_nn_lab.tracks.flow.paths import (
    PATHS,
    LinearPath,
    ProbabilityPath,
    TrigonometricPath,
    build_path,
)
from modern_nn_lab.tracks.flow.solver import (
    EVALUATIONS_PER_STEP,
    Method,
    Trajectory,
    integrate,
)

__all__ = [
    "DATASETS",
    "EVALUATIONS_PER_STEP",
    "MIXTURE_CENTRES",
    "PATHS",
    "Dataset",
    "FlowConfig",
    "FlowExperimentConfig",
    "GaussianEndpoints",
    "LinearPath",
    "Method",
    "ProbabilityPath",
    "TimeEmbedding",
    "Trajectory",
    "TrigonometricPath",
    "VectorField",
    "build_path",
    "energy_distance",
    "flow_matching_loss",
    "integrate",
    "marginal_velocity",
    "mode_coverage",
    "projection_residual",
    "sample_source",
    "sample_target",
]
