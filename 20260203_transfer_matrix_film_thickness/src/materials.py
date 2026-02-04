"""
Optical constants (refractive indices) for thin film materials.

All functions return complex refractive index n + ik as a function of wavelength in nm.

Data sources:
- Si: Aspnes & Studna (1983), widely used for crystalline silicon
- SiO2: Malitson (1965) Sellmeier equation
- BaTiO3 (BTO): Wemple (1970) and various literature sources
"""

import numpy as np
from scipy.interpolate import interp1d


def n_air(wavelength_nm: float) -> complex:
    """Refractive index of air (approximately 1)."""
    return complex(1.0, 0.0)


def n_sio2(wavelength_nm: float) -> complex:
    """Refractive index of SiO2 (fused silica) using Malitson Sellmeier equation.
    
    Valid range: 210-3710 nm
    Reference: Malitson, J. Opt. Soc. Am. 55, 1205-1209 (1965)
    """
    wl_um = wavelength_nm / 1000.0  # Convert to micrometers
    
    # Sellmeier coefficients
    n_sq = 1 + (0.6961663 * wl_um**2 / (wl_um**2 - 0.0684043**2) +
                0.4079426 * wl_um**2 / (wl_um**2 - 0.1162414**2) +
                0.8974794 * wl_um**2 / (wl_um**2 - 9.896161**2))
    
    return complex(np.sqrt(n_sq), 0.0)


def n_bto(wavelength_nm: float) -> complex:
    """Refractive index of BaTiO3 (barium titanate).
    
    Using simplified Sellmeier-type dispersion for ordinary ray.
    BTO is birefringent, but for normal incidence on c-axis oriented film,
    this is a reasonable approximation.
    
    Reference: Wemple, Phys. Rev. B 2, 2679 (1970)
    Approximate range: 400-700 nm gives n ~ 2.3-2.5
    """
    wl_um = wavelength_nm / 1000.0
    
    # Simplified Sellmeier for BTO (ordinary ray)
    # n^2 = A + B*lambda^2/(lambda^2 - C^2)
    A = 4.064  # Approximate high-frequency contribution
    B = 1.216
    C = 0.177  # Resonance wavelength in um
    
    n_sq = A + B * wl_um**2 / (wl_um**2 - C**2)
    n = np.sqrt(np.maximum(n_sq, 1.0))  # Ensure physical
    
    # BTO has very low absorption in visible range
    k = 0.0
    
    return complex(n, k)


# Silicon optical constants from Aspnes & Studna (1983)
# Wavelength (nm), n, k
_SI_DATA = np.array([
    [300, 4.37, 4.22],
    [310, 4.64, 4.15],
    [320, 5.01, 3.98],
    [330, 5.34, 3.71],
    [340, 5.55, 3.36],
    [350, 5.59, 2.99],
    [360, 5.49, 2.68],
    [370, 5.35, 2.46],
    [380, 5.22, 2.30],
    [390, 5.10, 2.18],
    [400, 5.00, 2.07],
    [410, 4.91, 1.97],
    [420, 4.83, 1.88],
    [430, 4.75, 1.79],
    [440, 4.68, 1.71],
    [450, 4.61, 1.63],
    [460, 4.54, 1.55],
    [470, 4.48, 1.48],
    [480, 4.42, 1.41],
    [490, 4.36, 1.35],
    [500, 4.30, 1.29],
    [510, 4.25, 1.23],
    [520, 4.20, 1.17],
    [530, 4.15, 1.12],
    [540, 4.10, 1.07],
    [550, 4.06, 1.02],
    [560, 4.02, 0.976],
    [570, 3.98, 0.933],
    [580, 3.94, 0.892],
    [590, 3.90, 0.854],
    [600, 3.87, 0.817],
    [620, 3.80, 0.749],
    [640, 3.74, 0.687],
    [660, 3.69, 0.632],
    [680, 3.64, 0.581],
    [700, 3.59, 0.535],
    [720, 3.55, 0.494],
    [740, 3.51, 0.456],
    [760, 3.48, 0.421],
    [780, 3.45, 0.390],
    [800, 3.42, 0.361],
    [850, 3.36, 0.298],
    [900, 3.32, 0.246],
    [950, 3.28, 0.203],
    [1000, 3.25, 0.167],
])

# Create interpolation functions for Si
_si_n_interp = interp1d(_SI_DATA[:, 0], _SI_DATA[:, 1], 
                        kind='cubic', fill_value='extrapolate')
_si_k_interp = interp1d(_SI_DATA[:, 0], _SI_DATA[:, 2], 
                        kind='cubic', fill_value='extrapolate')


def n_si(wavelength_nm: float) -> complex:
    """Refractive index of crystalline Silicon.
    
    Reference: Aspnes & Studna, Phys. Rev. B 27, 985 (1983)
    Valid range: 300-1000 nm (interpolated)
    """
    n = float(_si_n_interp(wavelength_nm))
    k = float(_si_k_interp(wavelength_nm))
    return complex(n, k)


# Material lookup dictionary
MATERIALS = {
    'air': n_air,
    'sio2': n_sio2,
    'SiO2': n_sio2,
    'bto': n_bto,
    'BTO': n_bto,
    'BaTiO3': n_bto,
    'si': n_si,
    'Si': n_si,
    'silicon': n_si,
}


def get_material(name: str):
    """Get refractive index function by material name."""
    if name in MATERIALS:
        return MATERIALS[name]
    raise ValueError(f"Unknown material: {name}. Available: {list(MATERIALS.keys())}")

