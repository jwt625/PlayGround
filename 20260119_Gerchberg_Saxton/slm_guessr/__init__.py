"""
SLM-Guessr Pattern Generator Package

Generates phase-intensity pairs for SLM training.
"""

from .patterns import (
    create_uniform_phase,
    create_linear_ramp,
    create_quadratic_phase,
    create_cubic_phase,
    create_spot_target,
    create_gaussian_spot_target,
    create_rectangular_slab_target,
)

from .generator import generate_sample, generate_all_samples

__all__ = [
    "create_uniform_phase",
    "create_linear_ramp",
    "create_quadratic_phase",
    "create_cubic_phase",
    "create_spot_target",
    "create_gaussian_spot_target",
    "create_rectangular_slab_target",
    "generate_sample",
    "generate_all_samples",
]

