"""GeO2-SiO2 glass dispersion.

Sellmeier: n^2 - 1 = sum_i B_i lam^2 / (lam^2 - L_i^2), lam in um.
SiO2: Malitson, JOSA 55, 1205 (1965). GeO2: Fleming, Appl. Opt. 23, 4486 (1984).
Binary glass: B_i and L_i linearly interpolated in GeO2 mole fraction x (as in ofiber.refraction.doped_glass).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

SA = np.array([0.6961663, 0.4079426, 0.8974794])
SL = np.array([0.0684043, 0.1162414, 9.896161])
GA = np.array([0.80686642, 0.71815848, 0.85416831])
GL = np.array([0.068972606, 0.15396605, 11.841931])


def _coeffs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)[..., None]
    return SA + x * (GA - SA), SL + x * (GL - SL)


def index(lam_um: float, x: np.ndarray | float) -> np.ndarray:
    """Refractive index of x GeO2 : (1-x) SiO2 glass."""
    b, lr = _coeffs(x)
    l2 = lam_um**2
    return np.sqrt(1.0 + np.sum(b * l2 / (l2 - lr**2), axis=-1))


def index_and_group(lam_um: float, x: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return (n, N) with N = n - lam dn/dlam (material group index), analytic derivative."""
    b, lr = _coeffs(x)
    l2 = lam_um**2
    n2 = 1.0 + np.sum(b * l2 / (l2 - lr**2), axis=-1)
    n = np.sqrt(n2)
    # d(n^2)/dlam = sum b * (-2 lam L^2) / (lam^2 - L^2)^2
    dn2 = np.sum(-2.0 * b * lam_um * lr**2 / (l2 - lr**2) ** 2, axis=-1)
    dn = dn2 / (2.0 * n)
    return n, n - lam_um * dn


def x_for_na(lam_um: float, na: float) -> float:
    """GeO2 mole fraction giving NA = sqrt(n_core^2 - n_SiO2^2) at lam."""
    nc = float(index(lam_um, 0.0))
    return float(brentq(lambda x: float(index(lam_um, x)) ** 2 - nc**2 - na**2, 0.0, 1.0))
