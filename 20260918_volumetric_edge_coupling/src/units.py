"""Unit conventions for the volumetric edge-coupler reproduction.

Internal length unit: 1 micrometre (Meep ``a = 1 um``).
Display units: thickness/gap in nm, angles in degrees.
Meep frequency unit: c / (1 um), so ``freq = 1 / wavelength_um``.

All external configuration must carry explicit units in its field names or
schema; conversion happens only here.
"""

from __future__ import annotations

import math

# Meep base length unit.
UM: float = 1.0
NM_PER_UM: float = 1000.0


def nm_to_um(nm: float) -> float:
    """Convert nanometres to the internal micrometre unit."""
    return nm / NM_PER_UM


def um_to_nm(um: float) -> float:
    """Convert the internal micrometre unit to nanometres."""
    return um * NM_PER_UM


def freq_from_wavelength_um(wavelength_um: float) -> float:
    """Meep frequency for a vacuum wavelength expressed in micrometres."""
    return 1.0 / wavelength_um


def freq_from_wavelength_nm(wavelength_nm: float) -> float:
    """Meep frequency for a vacuum wavelength expressed in nanometres."""
    return freq_from_wavelength_um(nm_to_um(wavelength_nm))


def wavelength_um_from_freq(freq: float) -> float:
    return 1.0 / freq


def wavelength_nm_from_freq(freq: float) -> float:
    return um_to_nm(wavelength_um_from_freq(freq))


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi
