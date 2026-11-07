"""Evaluate a fiber design: OFL modal RMS delay spread over a Gaussian source spectrum and related metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import material
from .solver import Grid, Modes, solve_modes

ShapeFn = Callable[[np.ndarray], np.ndarray]  # F(rho), rho = r / a


@dataclass
class Source:
    lam0_um: float
    fwhm_nm: float
    n_nodes: int = 5

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Gauss-Hermite nodes/weights for a Gaussian spectrum (weights sum to 1)."""
        if self.fwhm_nm <= 0 or self.n_nodes <= 1:
            return np.array([self.lam0_um]), np.array([1.0])
        t, w = np.polynomial.hermite.hermgauss(self.n_nodes)
        sig = self.fwhm_nm * 1e-3 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        return self.lam0_um + np.sqrt(2.0) * sig * t, w / w.sum()


@dataclass
class Numerics:
    dr_um: float = 0.05
    clad_margin_um: float = 15.0
    oversample: int = 8


@dataclass
class Result:
    sigma_modal_ps_per_m: float  # spectrally averaged OFL intermodal RMS
    sigma_modal_center_ps_per_m: float
    sigma_total_ps_per_m: float  # modal + chromatic, pooled over spectrum
    max_dmd_center_ps_per_m: float
    mode_count_center: float
    etendue_um2_sr: float
    modes_center: Modes = field(repr=False)
    tau_by_lam: list[np.ndarray] = field(repr=False, default_factory=list)
    w_by_lam: list[np.ndarray] = field(repr=False, default_factory=list)
    s_by_lam: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def material_on_grid(grid: Grid, lam_um: float, a_um: float, x0: float, shape: ShapeFn, oversample: int
                     ) -> tuple[np.ndarray, np.ndarray, float]:
    """Cell-averaged n (via <n^2>) and <n N> on the grid; returns (n, N_eff, n_clad)."""
    dr = grid.dr
    sub = (np.arange(oversample) + 0.5) / oversample - 0.5
    rs = grid.r[:, None] + sub[None, :] * dr
    rs = np.clip(rs, 1e-9, None)
    x = x0 * (1.0 - shape(rs / a_um))
    x[rs >= a_um] = 0.0
    x = np.clip(x, 0.0, None)
    n, ng = material.index_and_group(lam_um, x)
    wts = rs / rs.sum(axis=1, keepdims=True)  # area weighting within a cell
    n2 = np.sum(wts * n**2, axis=1)
    nn = np.sum(wts * n * ng, axis=1)
    n_cell = np.sqrt(n2)
    n_clad = float(material.index(lam_um, 0.0))
    return n_cell, nn / n_cell, n_clad


def etendue(a_um: float, x0: float, shape: ShapeFn, lam_um: float, num: Numerics | None = None) -> float:
    """Fiber etendue G = int pi (n^2 - n_clad^2) 2 pi r dr (um^2 sr); proportional to guided mode count."""
    num = num or Numerics()
    grid = Grid.make(a_um + num.clad_margin_um, num.dr_um)
    n, _, ncl = material_on_grid(grid, lam_um, a_um, x0, shape, num.oversample)
    return float(np.sum(np.pi * np.clip(n**2 - ncl**2, 0, None) * 2 * np.pi * grid.r) * grid.dr)


def ofl_stats(modes: Modes) -> tuple[float, float, np.ndarray, np.ndarray]:
    w = modes.degeneracy
    tau = modes.tau_ps_per_m
    mu = np.sum(w * tau) / w.sum()
    return float(np.sqrt(np.sum(w * (tau - mu) ** 2) / w.sum())), float(mu), tau, w


def evaluate(a_um: float, x0: float, shape: ShapeFn, source: Source, num: Numerics | None = None) -> Result:
    num = num or Numerics()
    grid = Grid.make(a_um + num.clad_margin_um, num.dr_um)
    lams, s = source.nodes()
    var_modal = 0.0
    taus, ws = [], []
    center_modes = None
    for lam in lams:
        n, ngm, ncl = material_on_grid(grid, lam, a_um, x0, shape, num.oversample)
        modes = solve_modes(grid, lam, n, ngm, ncl)
        if abs(lam - source.lam0_um) < 1e-12:
            center_modes = modes
        sig, _, tau, w = ofl_stats(modes)
        taus.append(tau)
        ws.append(w)
    for sj, tau, w in zip(s, taus, ws, strict=True):
        mu = np.sum(w * tau) / w.sum()
        var_modal += sj * np.sum(w * (tau - mu) ** 2) / w.sum()
    # pooled: each wavelength carries spectral weight s_j, split equally over its guided modes (OFL)
    all_tau = np.concatenate(taus)
    all_w = np.concatenate([sj * w / w.sum() for sj, w in zip(s, ws, strict=True)])
    mu_all = np.sum(all_w * all_tau)
    sig_total = float(np.sqrt(np.sum(all_w * (all_tau - mu_all) ** 2)))

    n, ngm, ncl = material_on_grid(grid, source.lam0_um, a_um, x0, shape, num.oversample)
    if center_modes is None:
        mc = solve_modes(grid, source.lam0_um, n, ngm, ncl)
    else:
        mc = center_modes
    sig_c, _, tau_c, _ = ofl_stats(mc)
    r = grid.r
    etendue = float(np.sum(np.pi * np.clip(n**2 - ncl**2, 0, None) * 2 * np.pi * r) * grid.dr)
    return Result(
        sigma_modal_ps_per_m=float(np.sqrt(var_modal)),
        sigma_modal_center_ps_per_m=sig_c,
        sigma_total_ps_per_m=sig_total,
        max_dmd_center_ps_per_m=float(tau_c.max() - tau_c.min()) if tau_c.size else 0.0,
        mode_count_center=mc.count,
        etendue_um2_sr=etendue,
        modes_center=mc,
        tau_by_lam=taus,
        w_by_lam=ws,
        s_by_lam=s,
    )


def bandwidth_length_mhz_km(taus: list[np.ndarray], ws: list[np.ndarray], s: np.ndarray) -> float:
    """-3 dB optical bandwidth-length product (first |H| = 0.5) of the pooled impulse response, MHz km."""
    tau = np.concatenate(taus) * 1e3  # ps over 1 km
    w = np.concatenate([sj * wi / wi.sum() for sj, wi in zip(s, ws, strict=True)])
    tau = tau - np.sum(w * tau)

    def mag(f_thz: np.ndarray) -> np.ndarray:
        return np.abs(np.exp(-2j * np.pi * np.outer(f_thz, tau)) @ w)

    f = np.logspace(-7, 0, 701)  # THz, i.e. 0.1 MHz km .. 1e6 MHz km
    h = mag(f)
    below = np.nonzero(h < 0.5)[0]
    if below.size == 0:
        return float("inf")
    i = below[0]
    lo = f[i - 1] if i > 0 else 0.0
    f = np.linspace(lo, f[i], 401)
    h = mag(f)
    j = np.nonzero(h < 0.5)[0][0]
    f0, f1, h0, h1 = f[j - 1], f[j], h[j - 1], h[j]
    return float((f0 + (f1 - f0) * (h0 - 0.5) / (h0 - h1)) * 1e6)
