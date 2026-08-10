"""Kolmogorov-Arnold Networks.

Mechanism under study: replacing the scalar weight on each edge with a learnable
univariate function. See ``README.md`` in this package for the mathematical
specification, the equation-to-code mapping, and the deviations from the primary source.
"""

from modern_nn_lab.tracks.kan.config import KANConfig, KANExperimentConfig
from modern_nn_lab.tracks.kan.layers import KANLayer
from modern_nn_lab.tracks.kan.model import (
    KAN,
    MLP,
    count_parameters,
    match_parameter_budget,
)
from modern_nn_lab.tracks.kan.spline import (
    adaptive_grid,
    b_spline_basis,
    build_grid,
    curve_to_coefficients,
)

__all__ = [
    "KAN",
    "MLP",
    "KANConfig",
    "KANExperimentConfig",
    "KANLayer",
    "adaptive_grid",
    "b_spline_basis",
    "build_grid",
    "count_parameters",
    "curve_to_coefficients",
    "match_parameter_budget",
]
