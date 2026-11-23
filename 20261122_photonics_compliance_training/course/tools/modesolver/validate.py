"""Validation of fdmode.py. Run from this folder:

    uv run --no-project --with numpy --with scipy python validate.py

V1  isotropic three-layer slab (laterally periodic 2D solve) vs the analytic slab dispersion relation, TE0 and TM0,
    with grid refinement.
V2  anisotropic x-cut slab at in-plane angles 0, 30, 60, 90 deg (laterally periodic 2D solve, full tensor with the
    lateral-longitudinal coupling) vs an independent 1D 4 x 4 transfer-matrix (Berreman) solution of the same layer
    stack. This is the check of the off-diagonal tensor treatment.
V3  rib at 0 and 90 deg (diagonal tensor): quadratic full-vector solver vs the independent linear transverse-field
    formulation on the same grid.
V4  rib at a few angles: grid convergence and computational window checks.
Prints a table; writes nothing.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq, minimize_scalar

import fdmode as fm

LAM = 1.55
NO, NE = fm.n_ln_zelmon(LAM)
NS = fm.n_sio2_malitson(LAM)
K0 = 2 * np.pi / LAM


def slab_analytic(t: float, n1: float, ns: float, nc: float, pol: str) -> float:
    """Fundamental mode of a three-layer slab (core n1, substrate ns, cover nc)."""

    def f(n):
        k = K0 * np.sqrt(n1**2 - n**2)
        ps = K0 * np.sqrt(n**2 - ns**2)
        pc = K0 * np.sqrt(n**2 - nc**2)
        if pol == "TE":
            return k * t - np.arctan(ps / k) - np.arctan(pc / k)
        return k * t - np.arctan(n1**2 / ns**2 * ps / k) - np.arctan(n1**2 / nc**2 * pc / k)

    return brentq(f, max(ns, nc) + 1e-9, n1 - 1e-9, xtol=1e-14)


def berreman_det(n: float, layers, sub: dict, cov: dict) -> float:
    """|det| of the matching matrix for a 1D stack (y normal) with in-plane wavevector along z, kx = 0.
    Psi = (Ex, Hz, Ez, Hx) with H scaled by the vacuum impedance; dPsi/dy = i k0 D Psi,
    D = [[0, -1, 0, 0], [n^2 - exx, 0, -exz, 0], [0, 0, 0, 1 - n^2 / eyy], [exz, 0, ezz, 0]]."""

    def D(e):
        return np.array([[0, -1, 0, 0], [n * n - e["xx"], 0, -e["xz"], 0], [0, 0, 0, 1 - n * n / e["yy"]],
                         [e["xz"], 0, e["zz"], 0]], dtype=complex)

    def decaying(e, down: bool):
        w, v = np.linalg.eig(D(e))
        # exp(i k0 w y): decays toward -inf if Im w < 0, toward +inf if Im w > 0
        idx = np.argsort(w.imag)
        return v[:, idx[:2]] if down else v[:, idx[2:]]

    T = np.eye(4, dtype=complex)
    for th, e in layers:
        T = expm(1j * K0 * D(e) * th) @ T
    vs = decaying(sub, True)
    vc = decaying(cov, False)
    F = np.column_stack([T @ vs[:, 0], T @ vs[:, 1], vc[:, 0], vc[:, 1]])
    return float(abs(np.linalg.det(F)))


def berreman_modes(layers, sub, cov, nmin, nmax, nscan=4000) -> list[float]:
    ns = np.linspace(nmin + 1e-6, nmax - 1e-6, nscan)
    d = np.array([berreman_det(n, layers, sub, cov) for n in ns])
    out = []
    for i in range(1, nscan - 1):
        if d[i] < d[i - 1] and d[i] < d[i + 1]:
            r = minimize_scalar(lambda n: berreman_det(n, layers, sub, cov), bracket=(ns[i - 1], ns[i], ns[i + 1]),
                                tol=1e-14)
            # accept true zeros only (relative to neighbourhood scale)
            if r.fun < 1e-6 * max(d[i - 1], d[i + 1]):
                out.append(float(r.x))
    return sorted(out, reverse=True)


def slab2d(t: float, ln: dict, h: float, nmodes: int = 2) -> list[fm.Mode]:
    g = fm.make_grid(0.0, 4 * h, -1.6, t + 1.2, h, periodic_x=True)
    st = fm.Structure(fm.iso(1.0), [(lambda x, y: y < 0, fm.iso(NS)), (lambda x, y: (y >= 0) & (y <= t), ln)])
    op = fm.build(g, st, LAM)
    return fm.solve(op, NO, k=6, n_min=NS)[:nmodes]


def rib(t, slab, w, sw, theta, h, xw=3.0, box=1.4, top=1.0, cover=1.0):
    g = fm.make_grid(-xw, xw, -box, t + top, h)
    st = fm.rib_structure(t, slab, w, sw, fm.ln_tensor_xcut(NO, NE, theta), NS, cover)
    return g, st


def main() -> None:
    T, SL, W, SW = 0.8, 0.4, 1.2, 65.0
    print(f"indices at {LAM} um: n_o {NO:.5f}, n_e {NE:.5f} (Zelmon 1997), SiO2 {NS:.5f} (Malitson 1965)")
    t0 = time.time()

    print("\nV1 isotropic slab, film n_o, t 0.8 um, SiO2 below, air above")
    ref = {p: slab_analytic(T, NO, NS, 1.0, p) for p in ("TE", "TM")}
    print(f"  analytic TE0 {ref['TE']:.6f}  TM0 {ref['TM']:.6f}")
    for h in (0.04, 0.02, 0.01, 0.005):
        ms = slab2d(T, fm.iso(NO), h)
        te = [m.neff for m in ms if m.te_frac > 0.5][0]
        tm = [m.neff for m in ms if m.te_frac < 0.5][0]
        print(f"  h {h*1000:5.1f} nm  TE0 {te:.6f} ({te-ref['TE']:+.1e})  TM0 {tm:.6f} ({tm-ref['TM']:+.1e})")

    print("\nV2 anisotropic x-cut slab, t 0.8 um, full rotated tensor, vs 4x4 transfer matrix")
    for th in (0.0, 30.0, 60.0, 90.0):
        ln = fm.ln_tensor_xcut(NO, NE, th)
        bm = berreman_modes([(T, ln)], fm.iso(NS), fm.iso(1.0), NS, NO)[:2]
        ms = slab2d(T, ln, 0.005)
        line = "  ".join(f"{m.neff:.6f} (TE {m.te_frac:.3f}, d {m.neff-b:+.1e})" for m, b in zip(ms, bm))
        print(f"  theta {th:4.0f}  transfer matrix {', '.join(f'{b:.6f}' for b in bm)}  |  FD {line}")

    print(f"\nV3 rib t {T}, slab {SL}, top width {W}, sidewall {SW} deg, air top: quadratic solver vs transverse formulation")
    for th in (0.0, 90.0):
        g, st = rib(T, SL, W, SW, th, 0.02)
        q = [m.neff for m in fm.solve(fm.build(g, st, LAM), 2.15, k=6, n_min=NS)][:3]
        d = fm.transverse_diag(g, st, LAM, 2.15, k=6)[:3]
        print(f"  theta {th:3.0f}  quadratic {', '.join(f'{v:.7f}' for v in q)}  transverse {', '.join(f'{v:.7f}' for v in d)}"
              f"  max diff {max(abs(a-b) for a, b in zip(q, d)):.1e}")

    print("\nV4 rib grid convergence (first two modes)")
    for th in (0.0, 45.0):
        for h in (0.04, 0.03, 0.02, 0.015, 0.01):
            g, st = rib(T, SL, W, SW, th, h, xw=2.5, box=1.2, top=0.9)
            ms = fm.solve(fm.build(g, st, LAM), 2.15, k=4, n_min=NS)[:2]
            print(f"  theta {th:3.0f}  h {h*1000:4.0f} nm  " + "  ".join(f"{m.neff:.6f} (TE {m.te_frac:.3f})" for m in ms),
                  flush=True)
    print("\nV4b computational window (theta 45, h 20 nm)")
    for xw, box, top in ((2.5, 1.2, 0.9), (3.0, 1.4, 1.0), (4.0, 2.0, 1.4)):
        g, st = rib(T, SL, W, SW, 45.0, 0.02, xw=xw, box=box, top=top)
        ms = fm.solve(fm.build(g, st, LAM), 2.15, k=4, n_min=NS)[:2]
        print(f"  half width {xw} um, oxide {box} um, air {top} um: " + "  ".join(f"{m.neff:.6f}" for m in ms))
    print("\nV5 mirror symmetry: theta and 180 - theta give the same indices (h 20 nm)")
    for th in (30.0, 150.0):
        g, st = rib(T, SL, W, SW, th, 0.02)
        ms = fm.solve(fm.build(g, st, LAM), 2.15, k=4, n_min=NS)[:2]
        print(f"  theta {th:5.1f}  " + "  ".join(f"{m.neff:.7f}" for m in ms))
    print(f"\ndone in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
