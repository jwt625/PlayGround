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
    create_binary_grating,
    create_sinusoidal_grating,
    create_blazed_grating,
    create_checkerboard,
    create_crossed_gratings,
    create_multi_frequency_grating,
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

