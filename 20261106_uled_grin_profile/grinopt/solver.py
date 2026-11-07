"""Scalar LP mode solver for axially symmetric weakly guiding fibers.

Per azimuthal order l: psi'' + psi'/r - l^2/r^2 psi + k^2 n^2(r) psi = beta^2 psi.
Cell-centered radial finite differences r_i = (i + 1/2) dr, zero flux at r = 0, psi = 0 at r = R.
With phi = sqrt(r) psi the problem is a standard symmetric tridiagonal eigenproblem.
Group index (Hellmann-Feynman, exact for the discrete problem):
    d(beta^2)/dk = <phi| 2 k n N |phi>,  n_g = d beta / d k = <phi| n N |phi> k / beta,
N = n - lam dn/dlam the local material group index.
Lengths in um.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

C_PS_PER_M = 1e12 / 299_792_458.0  # 1/c in ps/m


@dataclass
class Grid:
    dr: float
    n_pts: int

    @property
    def r(self) -> np.ndarray:
        return (np.arange(self.n_pts) + 0.5) * self.dr

    @classmethod
    def make(cls, r_max: float, dr: float) -> Grid:
        return cls(dr=dr, n_pts=int(np.ceil(r_max / dr)))


@dataclass
class Modes:
    l: np.ndarray  # azimuthal order
    m: np.ndarray  # radial order (1-based)
    beta: np.ndarray  # propagation constant, 1/um
    ng: np.ndarray  # group index d beta / d k
    k: float
    n_clad: float

    @property
    def degeneracy(self) -> np.ndarray:
        return np.where(self.l == 0, 1.0, 2.0)

    @property
    def tau_ps_per_m(self) -> np.ndarray:
        return self.ng * C_PS_PER_M

    @property
    def principal(self) -> np.ndarray:
        """Principal mode number 2m + l - 1 (mode group index)."""
        return 2 * self.m + self.l - 1

    @property
    def count(self) -> float:
        return float(self.degeneracy.sum())


def solve_modes(grid: Grid, lam_um: float, n: np.ndarray, ng_mat: np.ndarray, n_clad: float,
                l_max: int | None = None, trim_tol: float = 1e-10) -> Modes:
    """All guided scalar LP modes (beta > k n_clad) for index n(r) and local material group index ng_mat(r)."""
    r = grid.r
    dr = grid.dr
    k = 2.0 * np.pi / lam_um
    rp = r + 0.5 * dr  # r_{i+1/2}
    rm = r - 0.5 * dr  # r_{i-1/2}, rm[0] = 0
    off = rp[:-1] / (dr**2 * np.sqrt(r[:-1] * r[1:]))
    base = -(rp + rm) / (dr**2 * r) + (k * n) ** 2
    lo = (k * n_clad) ** 2
    hi = (k * n.max()) ** 2 + 1.0
    weight = n * ng_mat
    ls, ms, bs, gs = [], [], [], []
    kt_max = np.sqrt(max(hi - 1.0 - lo, 1e-30))  # max transverse wavenumber of any guided mode
    l = 0
    while l_max is None or l <= l_max:
        # trim the evanescent inner region: psi ~ (r / r_turn)^l < trim_tol there, r_turn >= l / kt_max
        i0 = 0
        if trim_tol > 0 and l >= 4:
            r_cut = (l / kt_max) * trim_tol ** (1.0 / l)
            i0 = int(np.searchsorted(r, r_cut))
        d = (base - l**2 / r**2)[i0:]
        try:
            w, v = eigh_tridiagonal(d, off[i0:], select="v", select_range=(lo, hi), lapack_driver="stemr")
        except ValueError:
            w = np.array([])
        if w.size == 0:
            break
        order = np.argsort(w)[::-1]
        w, v = w[order], v[:, order]
        beta = np.sqrt(w)
        ng = k / beta * np.einsum("ij,i,ij->j", v, weight[i0:], v) / np.einsum("ij,ij->j", v, v)
        ls.append(np.full(w.size, l))
        ms.append(np.arange(1, w.size + 1))
        bs.append(beta)
        gs.append(ng)
        l += 1
    if not ls:
        e = np.array([])
        return Modes(e.astype(int), e.astype(int), e, e, k, n_clad)
    return Modes(np.concatenate(ls), np.concatenate(ms), np.concatenate(bs), np.concatenate(gs), k, n_clad)
