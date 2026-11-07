"""Monotonic axially symmetric dopant profiles x(r) = x0 (1 - F(r/a)), F(0) = 0, F nondecreasing, x = 0 for r >= a."""

from __future__ import annotations

import numpy as np
from scipy.special import comb


def alpha_shape(rho: np.ndarray, alpha: float) -> np.ndarray:
    return np.clip(rho, 0.0, 1.0) ** alpha


def bernstein_coeffs(params: np.ndarray, edge_step: bool = False) -> np.ndarray:
    """Monotone Bernstein coefficients c_0 = 0 <= ... <= c_N (= 1 unless edge_step) from unconstrained params.

    Without edge_step: N = len(params) increments via softmax, cumulative sum ends at 1.
    With edge_step: last param is a logit for c_N in (0, 1), remaining N params set increments.
    """
    p = np.asarray(params, dtype=float)
    if edge_step:
        top = 1.0 / (1.0 + np.exp(-p[-1]))
        p = p[:-1]
    else:
        top = 1.0
    w = np.exp(p - p.max())
    w = w / w.sum() * top
    return np.concatenate([[0.0], np.cumsum(w)])


def bernstein_shape(rho: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    n = len(coeffs) - 1
    t = np.clip(rho, 0.0, 1.0)[..., None]
    k = np.arange(n + 1)
    basis = comb(n, k) * t**k * (1.0 - t) ** (n - k)
    return basis @ coeffs


def fit_bernstein_params(target: np.ndarray, rho: np.ndarray, degree: int) -> np.ndarray:
    """Unconstrained params whose monotone Bernstein shape least-squares fits target(rho)."""
    from scipy.optimize import least_squares

    def resid(p: np.ndarray) -> np.ndarray:
        return bernstein_shape(rho, bernstein_coeffs(p)) - target

    p0 = np.zeros(degree)
    return least_squares(resid, p0, xtol=1e-12, ftol=1e-12).x


def _softmax(p: np.ndarray) -> np.ndarray:
    w = np.exp(p - p.max())
    return w / w.sum()


def staircase_layers(level_params: np.ndarray, radius_params: np.ndarray | None, n_layers: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Monotone staircase: returns (outer radii as fraction of a, levels as fraction of x0).

    Levels: s = cumulative softmax of n_layers params; level_j = 1 - s_{j-1}, level_1 = 1, all > 0 (final drop to
    cladding is the last softmax weight). Radii: cumulative softmax of n_layers thickness params, last = 1;
    radius_params None means equal thickness.
    """
    wl = _softmax(np.asarray(level_params, dtype=float))
    levels = 1.0 - np.concatenate([[0.0], np.cumsum(wl)[:-1]])
    if radius_params is None:
        radii = np.arange(1, n_layers + 1) / n_layers
    else:
        radii = np.cumsum(_softmax(np.asarray(radius_params, dtype=float)))
        radii[-1] = 1.0
    return radii, levels


def staircase_shape(rho: np.ndarray, radii: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """F(rho) = 1 - level of the layer containing rho (0 inside first layer)."""
    idx = np.searchsorted(radii, np.clip(rho, 0.0, 1.0), side="right")
    idx = np.minimum(idx, len(levels) - 1)
    return 1.0 - levels[idx]


def staircase_init_from_shape(shape_fn, n_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-thickness quantization of a smooth shape F: level of layer j = 1 - F(midpoint)."""
    mid = (np.arange(n_layers) + 0.5) / n_layers
    lev = 1.0 - shape_fn(mid)
    lev[0] = 1.0
    drops = np.concatenate([-np.diff(lev), [lev[-1]]])
    drops = np.clip(drops, 1e-6, None)
    return np.log(drops / drops.sum()), np.zeros(n_layers)


def dopant_profile(r: np.ndarray, a: float, x0: float, shape: np.ndarray) -> np.ndarray:
    """Mole fraction on grid r given shape F evaluated at r/a (shape array same length as r)."""
    x = x0 * (1.0 - shape)
    x[r >= a] = 0.0
    return np.clip(x, 0.0, None)
