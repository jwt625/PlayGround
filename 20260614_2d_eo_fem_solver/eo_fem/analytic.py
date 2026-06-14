from __future__ import annotations

import math

from .constants import EPS0


def parallel_plate_capacitance(eps_r: float, width: float, gap: float) -> float:
    """Capacitance per unit length for two infinite-width parallel plates."""
    if width <= 0.0 or gap <= 0.0:
        raise ValueError("width and gap must be positive")
    return EPS0 * eps_r * width / gap


def two_cylinder_capacitance(eps_r: float, radius: float, center_distance: float) -> float:
    """Capacitance per unit length between two identical parallel cylinders."""
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if center_distance <= 2.0 * radius:
        raise ValueError("center_distance must be larger than 2 * radius")
    return math.pi * EPS0 * eps_r / math.acosh(center_distance / (2.0 * radius))
