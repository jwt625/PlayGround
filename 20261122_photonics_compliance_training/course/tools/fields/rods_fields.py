"""Fields for the "rods at twelve o'clock" cross-section keyframe: x-cut thin-film LiNbO3 rib in one gap of a
G-S-G coplanar line, RF quasi-static field and the TE0 / TM0 optical modes, overlap factors, compact JS export.

Stages (each cached as .npz in --cache, skipped if present unless --force):
  es     2D electrostatics (finite volume, anisotropic, non-uniform grid, mirror half domain), three permittivity sets
  modes  TE0 and TM0 of the rib at 1550 nm with the course's full-vector solver (../modesolver/fdmode.py)
  eval   overlap factors (scalar Gamma) and full-vector first-order Pockels index change per volt, all variants
  export compact data file for the canvas keyframe (anims/fe_rods_fields_data.js)

  uv run --no-project --with numpy --with scipy python rods_fields.py all --cache <scratch>/cache --h 0.025
See README.md for geometry, materials, method and checks.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")

import argparse  # noqa: E402
import base64  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from scipy.sparse.linalg import spsolve  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "modesolver"))
import fdmode  # noqa: E402

EPS0_PF_PER_M = 8.8541878128  # vacuum permittivity in pF/m

# ---------------------------------------------------------------------------------------------------------------
# Geometry (um). Frame: x lateral (crystal z), y film normal (crystal x), z propagation (crystal y). Signal centred
# at x = 0; mirror plane x = 0; rib A centred in the right gap.
T_FILM, T_SLAB, W_TOP, SIDEWALL_DEG = 0.6, 0.3, 1.0, 60.0
T_BOX, T_SI, T_EL = 4.7, 500.0, 0.8
W_SIG, GAP, W_GND = 20.0, 5.0, 100.0
X_RIB = W_SIG / 2 + GAP / 2  # 12.5
X_MAX, Y_TOP = 500.0, 500.0
LAM = 1.55

_cot = 1.0 / np.tan(np.deg2rad(SIDEWALL_DEG))
RIB_BOT_HALF = 0.5 * W_TOP + (T_FILM - T_SLAB) * _cot

# permittivity / Pockels sets. eps: (lateral = crystal z = par, vertical = crystal x = perp)
VARIANTS = {
    # cross-check set: same constants as an independent FEM reference computation of this geometry
    "check": {"eps_par": 28.0, "eps_perp": 44.0, "r": {"r13": 9.6, "r22": 6.8, "r33": 30.9, "r51": 32.6}},
    # RF (clamped): Weis and Gaylord, Appl. Phys. A 37, 191 (1985), Table 1 eps^S [23], Table 4 r^S 633 nm [45]
    "rf": {"eps_par": 27.9, "eps_perp": 44.3, "r": {"r13": 8.6, "r22": 3.4, "r33": 30.8, "r51": 28.0}},
    # low frequency (unclamped): same tables, eps^T [23], r^T 633 nm [37] (r13, r22, r33) and [41] (r51)
    "lf": {"eps_par": 28.7, "eps_perp": 85.2, "r": {"r13": 10.0, "r22": 6.81, "r33": 32.2, "r51": 32.0}},
}
EPS_SIO2, EPS_SI = 3.9, 11.7


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Electrostatics


def graded_axis(lo: float, hi: float, must: list[float], fine: tuple[float, float], hmin: float, hmax: float,
                growth: float = 0.15) -> np.ndarray:
    """Node coordinates with spacing hmin inside `fine`, growing linearly with distance outside (slope `growth`,
    about a geometric ratio of 1 + growth), capped at hmax; every value in `must` is a node."""
    pts = sorted(set([lo, hi] + [m for m in must if lo <= m <= hi]))

    def h(s):
        d = np.maximum(0.0, np.maximum(fine[0] - s, s - fine[1]))
        return np.minimum(hmax, hmin + growth * d)

    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        s = np.linspace(a, b, 4001)
        w = 1.0 / h(s)
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(s))])
        n = max(1, int(np.ceil(cum[-1])))
        out.extend(np.interp(np.linspace(0, cum[-1], n + 1)[1:], cum, s))
    return np.array(out)


def materials_cells(xc: np.ndarray, yc: np.ndarray, dx: np.ndarray, dy: np.ndarray, var: dict, S: int = 4):
    """Cell permittivity (eps_xx, eps_yy), sub-sampled S x S (arithmetic average; exact for grid-aligned
    interfaces, area-weighted on the slanted rib sidewalls)."""
    off = (np.arange(S) + 0.5) / S - 0.5
    X = xc[:, None, None, None] + dx[:, None, None, None] * off[None, None, :, None]
    Y = yc[None, :, None, None] + dy[None, :, None, None] * off[None, None, None, :]
    X, Y = np.broadcast_arrays(X, Y)
    exx = np.ones(X.shape)
    eyy = np.ones(X.shape)
    si = Y < -T_BOX
    box = (Y >= -T_BOX) & (Y < 0)
    slab = (Y >= 0) & (Y < T_SLAB)
    half = 0.5 * W_TOP + (T_FILM - Y) * _cot
    rib = (Y >= T_SLAB) & (Y < T_FILM) & (np.abs(np.abs(X) - X_RIB) <= half)
    ln = slab | rib
    exx[si] = eyy[si] = var.get("eps_si", EPS_SI)
    exx[box] = eyy[box] = var.get("eps_sio2", EPS_SIO2)
    exx[ln] = var["eps_par"]
    eyy[ln] = var["eps_perp"]
    return exx.mean(axis=(2, 3)), eyy.mean(axis=(2, 3))


def es_grid(hf: float):
    xr = [0.0, W_SIG / 2, X_RIB - RIB_BOT_HALF, X_RIB - W_TOP / 2, X_RIB + W_TOP / 2, X_RIB + RIB_BOT_HALF,
          W_SIG / 2 + GAP, W_SIG / 2 + GAP + W_GND, X_MAX]
    yr = [-T_BOX - T_SI, -T_BOX, 0.0, T_SLAB, T_FILM, T_SLAB + T_EL, Y_TOP]
    xs = graded_axis(0.0, X_MAX, xr, (6.0, 19.0), hf, 20.0)
    ys = graded_axis(-T_BOX - T_SI, Y_TOP, yr, (-2.5, 3.0), hf, 20.0)
    return xs, ys


def solve_es(var: dict, hf: float):
    xs, ys = es_grid(hf)
    nx, ny = len(xs), len(ys)
    dx, dy = np.diff(xs), np.diff(ys)
    xc, yc = 0.5 * (xs[1:] + xs[:-1]), 0.5 * (ys[1:] + ys[:-1])
    exx, eyy = materials_cells(xc, yc, dx, dy, var)
    # x-edge (i,j)-(i+1,j): eps_xx of the cells above and below, weighted by half heights
    cx = np.zeros((nx - 1, ny))
    cx[:, :-1] += exx * dy[None, :] / 2
    cx[:, 1:] += exx * dy[None, :] / 2
    cx /= dx[:, None]
    cy = np.zeros((nx, ny - 1))
    cy[:-1, :] += eyy * dx[:, None] / 2
    cy[1:, :] += eyy * dx[:, None] / 2
    cy /= dy[None, :]
    idx = np.arange(nx * ny).reshape(nx, ny)
    a, b = idx[:-1, :].ravel(), idx[1:, :].ravel()
    c, d = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    rows = np.concatenate([a, b, a, b, c, d, c, d])
    cols = np.concatenate([a, b, b, a, c, d, d, c])
    vals = np.concatenate([cx.ravel(), cx.ravel(), -cx.ravel(), -cx.ravel(), cy.ravel(), cy.ravel(), -cy.ravel(), -cy.ravel()])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(nx * ny, nx * ny))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    tol = 1e-9
    ely = (Y >= T_SLAB - tol) & (Y <= T_SLAB + T_EL + tol)
    sig = ely & (X <= W_SIG / 2 + tol)
    gnd = ely & (X >= W_SIG / 2 + GAP - tol) & (X <= W_SIG / 2 + GAP + W_GND + tol)
    fixed = (sig | gnd).ravel()
    V = np.zeros(nx * ny)
    V[sig.ravel()] = 1.0
    free = ~fixed
    Aff = A[free][:, free].tocsc()
    rhs = -A[free][:, fixed] @ V[fixed]
    V[free] = spsolve(Aff, rhs)
    C = 2 * EPS0_PF_PER_M * float(V @ (A @ V))  # full line (two halves), pF/m, signal at 1 V
    Vg = V.reshape(nx, ny)
    # cell-centred field (exact gradient of the bilinear interpolant at the cell centre)
    Ex = -0.5 * ((Vg[1:, :-1] - Vg[:-1, :-1]) + (Vg[1:, 1:] - Vg[:-1, 1:])) / dx[:, None]
    Ey = -0.5 * ((Vg[:-1, 1:] - Vg[:-1, :-1]) + (Vg[1:, 1:] - Vg[1:, :-1])) / dy[None, :]
    return {"xs": xs, "ys": ys, "V": Vg, "xc": xc, "yc": yc, "Ex": Ex, "Ey": Ey, "C_pF_per_m": C,
            "exx": exx, "eyy": eyy}


def stage_es(cache: Path, hf: float, force: bool) -> None:
    for name, var in list(VARIANTS.items()) + [("vac", {"eps_par": 1.0, "eps_perp": 1.0, "eps_sio2": 1.0, "eps_si": 1.0})]:
        f = cache / f"es_{name}_h{hf:g}.npz"
        if f.exists() and not force:
            log(f"es {name}: cached")
            continue
        t0 = time.time()
        r = solve_es(var, hf)
        np.savez_compressed(f, **{k: v for k, v in r.items()})
        log(f"es {name}: grid {len(r['xs'])} x {len(r['ys'])}, C = {r['C_pF_per_m']:.3f} pF/m, {time.time() - t0:.1f} s")


# ---------------------------------------------------------------------------------------------------------------
# Optical modes (course solver)


def stage_modes(cache: Path, h: float, force: bool, win: tuple[float, float, float, float]) -> None:
    f = cache / f"modes_h{h:g}_w{win[1] - win[0]:g}x{win[3] - win[2]:g}.npz"
    if f.exists() and not force:
        log(f"modes h={h}: cached")
        return
    no, ne = fdmode.n_ln_zelmon(LAM)
    nbox = fdmode.n_sio2_malitson(LAM)
    st = fdmode.rib_structure(T_FILM, T_SLAB, W_TOP, SIDEWALL_DEG, fdmode.ln_tensor_xcut(no, ne, 0.0), nbox, 1.0)
    g = fdmode.make_grid(*win, h)
    t0 = time.time()
    op = fdmode.build(g, st, LAM)
    modes = fdmode.solve(op, n_sigma=2.0, k=10, n_min=nbox + 1e-3)
    log(f"modes h={h}: grid {g.nx} x {g.ny}, {len(modes)} modes, {time.time() - t0:.1f} s")
    X, Y = np.meshgrid(g.xn, g.yn, indexing="ij")
    core = (np.abs(X) < 1.2) & (Y > -0.8) & (Y < 1.4)
    rows = []
    for m in modes:
        p = np.abs(m.Ex) ** 2 + np.abs(m.Ey) ** 2 + np.abs(m.Ez) ** 2
        rows.append((m, float(p[core].sum() / p.sum())))
        log(f"  neff {m.neff:.5f}  TE frac {m.te_frac:.4f}  core frac {rows[-1][1]:.3f}")
    te = [m for m, cf in rows if m.te_frac > 0.8 and cf > 0.6]
    tm = [m for m, cf in rows if m.te_frac < 0.2 and cf > 0.6]
    if not te or not tm:
        raise SystemExit("TE0 or TM0 not found")
    te0, tm0 = te[0], tm[0]
    np.savez_compressed(f, xn=g.xn, yn=g.yn, h=h, no=no, ne=ne, nbox=nbox,
                        te_neff=te0.neff, te_frac_te=te0.te_frac, te_Ex=te0.Ex, te_Ey=te0.Ey, te_Ez=te0.Ez,
                        tm_neff=tm0.neff, tm_frac_te=tm0.te_frac, tm_Ex=tm0.Ex, tm_Ey=tm0.Ey, tm_Ez=tm0.Ez)
    log(f"modes h={h}: TE0 {te0.neff:.5f} (TE frac {te0.te_frac:.4f}), TM0 {tm0.neff:.5f} (TE frac {tm0.te_frac:.4f})")


# ---------------------------------------------------------------------------------------------------------------
# Overlap and first-order Pockels perturbation


def ln_mask(X, Y):
    half = 0.5 * W_TOP + (T_FILM - Y) * _cot
    return ((Y >= 0) & (Y <= T_SLAB)) | ((Y > T_SLAB) & (Y <= T_FILM) & (np.abs(X) <= half))


def static_on(es: dict, X, Y):
    """Static field (per volt of signal) interpolated from cell centres onto points (X, Y) in rib coordinates."""
    pts = np.stack([(X + X_RIB).ravel(), Y.ravel()], axis=1)
    out = []
    for k in ("Ex", "Ey"):
        it = RegularGridInterpolator((es["xc"], es["yc"]), es[k], bounds_error=False, fill_value=0.0)
        out.append(it(pts).reshape(X.shape))
    return out  # V/um per volt


def perturb(md: dict, pre: str, es: dict, r: dict, no: float, ne: float):
    xn, yn = md["xn"], md["yn"]
    X, Y = np.meshgrid(xn, yn, indexing="ij")
    h = float(md["h"])
    Ex, Ey, Ez = md[pre + "Ex"], md[pre + "Ey"], md[pre + "Ez"]
    neff = float(md[pre + "neff"])
    k0 = 2 * np.pi / LAM
    beta = neff * k0
    m = ln_mask(X, Y)
    sx, sy = static_on(es, X, Y)  # V/um per V
    sx, sy = sx * m, sy * m
    # scalar overlap, conventional definition: Gamma = (g / V) int_LN E_RF,x |E_opt,i|^2 / int |E_opt,i|^2
    gam = {}
    for lab, comp in (("x", Ex), ("y", Ey)):
        I = np.abs(comp) ** 2
        gam[lab] = float(GAP * np.sum(sx * I) / np.sum(I))
    # full-vector first-order Pockels perturbation, full 3m tensor (pm/V -> um/V: 1e-6)
    E1, E3 = sy, sx  # crystal x = lab y, crystal z = lab x
    rr = {k: v * 1e-6 for k, v in r.items()}
    deta = {"xx": rr["r33"] * E3, "yy": rr["r13"] * E3, "zz": rr["r13"] * E3,
            "xy": rr["r51"] * E1, "yz": -rr["r22"] * E1, "xz": 0.0 * E1}
    epsd = {"x": ne**2, "y": no**2, "z": no**2}
    F = {"x": Ex, "y": Ey, "z": Ez}
    num = 0.0 + 0.0j
    parts = {}
    for key, de in deta.items():
        i, j = key[0], key[1]
        de_ij = -epsd[i] * epsd[j] * de
        term = np.sum(de_ij * np.conj(F[i]) * F[j])
        if i != j:
            term = term + np.sum(de_ij * np.conj(F[j]) * F[i])
        parts[key] = term * h * h
        num += parts[key]
    dEz_dx = np.gradient(Ez, xn, axis=0)
    dEz_dy = np.gradient(Ez, yn, axis=1)
    Hx = (dEz_dy - 1j * beta * Ey) / (1j * k0)
    Hy = (1j * beta * Ex - dEz_dx) / (1j * k0)
    P = 0.5 * np.real(np.sum(Ex * np.conj(Hy) - Ey * np.conj(Hx))) * h * h
    dn = float(np.real(num) / (4 * P))  # per volt of signal
    share = {k: float(np.real(v) / np.real(num)) for k, v in parts.items()}
    return {"neff": neff, "gamma_x": gam["x"], "gamma_y": gam["y"], "dn_per_V": dn, "parts": share}


def stage_eval(cache: Path, hf: float, hm: float, win, write: bool = True) -> dict:
    md = dict(np.load(cache / f"modes_h{hm:g}_w{win[1] - win[0]:g}x{win[3] - win[2]:g}.npz"))
    no, ne = float(md["no"]), float(md["ne"])
    res = {"h_es_um": hf, "h_mode_um": hm, "n_o": no, "n_e": ne, "te_neff": float(md["te_neff"]), "tm_neff": float(md["tm_neff"]),
           "te_frac_of_TE0": float(md["te_frac_te"]), "te_frac_of_TM0": float(md["tm_frac_te"]), "variants": {}}
    vac = dict(np.load(cache / f"es_vac_h{hf:g}.npz"))
    C0 = float(vac["C_pF_per_m"])
    for name, var in VARIANTS.items():
        es = dict(np.load(cache / f"es_{name}_h{hf:g}.npz"))
        te = perturb(md, "te_", es, var["r"], no, ne)
        tm = perturb(md, "tm_", es, var["r"], no, ne)
        C = float(es["C_pF_per_m"])
        n_rf = float(np.sqrt(C / C0))
        # push-pull: arm B is the mirror image with the opposite index change, VpiL = lambda / (2 |dn_A - dn_B|), V cm
        vpil_te = LAM / (4 * abs(te["dn_per_V"])) * 1e-4
        vpil_tm = LAM / (4 * abs(tm["dn_per_V"])) * 1e-4
        r = var["r"]
        res["variants"][name] = {
            "C_pF_per_m": C, "C0_pF_per_m": C0, "n_rf": n_rf, "Z0_ohm": 1 / (299792458 * np.sqrt(C * C0) * 1e-12),
            "TE": te, "TM": tm, "vpil_te_vcm": vpil_te, "vpil_tm_vcm": vpil_tm, "vpi_ratio_tm_te": vpil_tm / vpil_te,
            "ratio_equal_overlap": (ne**3 * r["r33"]) / (no**3 * r["r13"]),
            "ratio_scalar": (ne**3 * r["r33"] * te["gamma_x"]) / (no**3 * r["r13"] * tm["gamma_y"]),
        }
        v = res["variants"][name]
        log(f"{name}: C {C:.3f} pF/m, n_rf {n_rf:.4f}, Z0 {v['Z0_ohm']:.2f} ohm; TE Gamma {te['gamma_x']:.4f} dn/V {te['dn_per_V']:.4e} "
            f"VpiL {vpil_te:.4f} V cm; TM Gamma {tm['gamma_y']:.4f} dn/V {tm['dn_per_V']:.4e} VpiL {vpil_tm:.4f} V cm; "
            f"ratio full {v['vpi_ratio_tm_te']:.3f}, scalar {v['ratio_scalar']:.3f}, equal-overlap {v['ratio_equal_overlap']:.3f}")
        log(f"   TE parts {json.dumps({k: round(x, 4) for k, x in te['parts'].items()})}")
        log(f"   TM parts {json.dumps({k: round(x, 4) for k, x in tm['parts'].items()})}")
    if write:
        (cache / f"eval_h{hf:g}_m{hm:g}.json").write_text(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------------------------------------------------------
# Field lines


def field_lines(es: dict, view, n_lines: int, step: float = 0.01):
    """Lines of E, seeded on the signal electrode faces (top, side, bottom) at equal increments of
    sqrt(eps_n) E_n integrated along the faces: the geometric mean of E flux and D flux. With D flux alone the eps 44
    slab under the electrodes takes nearly all lines and the air gap stays empty; with E flux alone the air above
    the electrodes takes most of them. Traced with RK4 along E until they reach the ground electrode or leave the
    tracing box."""
    xc, yc = es["xc"], es["yc"]
    itx = RegularGridInterpolator((xc, yc), es["Ex"], bounds_error=False, fill_value=0.0)
    ity = RegularGridInterpolator((xc, yc), es["Ey"], bounds_error=False, fill_value=0.0)
    epx = RegularGridInterpolator((xc, yc), es["exx"], bounds_error=False, fill_value=1.0, method="nearest")
    epy = RegularGridInterpolator((xc, yc), es["eyy"], bounds_error=False, fill_value=1.0, method="nearest")
    xs0 = view[0] - 3.0
    e0, e1 = T_SLAB, T_SLAB + T_EL
    d = 0.004
    # boundary path around the signal corner region inside the seeding range: top face (right to left reversed),
    # side face, bottom face; parameter s, outward normal n, charge density sigma = eps E . n
    path = []
    for x in np.arange(xs0, W_SIG / 2, 0.002):
        path.append((x, e1 + d, 0.0, 1.0))
    for y in np.arange(e1, e0, -0.002):
        path.append((W_SIG / 2 + d, y, 1.0, 0.0))
    for x in np.arange(W_SIG / 2, xs0, -0.002):
        path.append((x, e0 - d, 0.0, -1.0))
    path = np.array(path)
    pts = path[:, :2]
    ex, ey = itx(pts), ity(pts)
    sig = np.sqrt(epx(pts)) * ex * path[:, 2] + np.sqrt(epy(pts)) * ey * path[:, 3]
    sig = np.maximum(sig, 0.0)
    seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (sig[1:] + sig[:-1]) * seg)])
    targets = (np.arange(n_lines) + 0.5) / n_lines * cum[-1]
    seeds = np.stack([np.interp(targets, cum, pts[:, 0]), np.interp(targets, cum, pts[:, 1])], axis=1)
    box = (view[0] - 6, view[1] + 6, view[2] - 8, view[3] + 8)

    def f(p):
        e = np.array([itx(p)[0], ity(p)[0]])
        n = np.hypot(*e)
        return e / n if n > 0 else e

    def in_gnd(p):
        return p[0] >= W_SIG / 2 + GAP and e0 <= p[1] <= e1

    def in_sig(p):
        return p[0] <= W_SIG / 2 and e0 <= p[1] <= e1

    lines = []
    for s in seeds:
        p = s.copy()
        out = [p.copy()]
        for _ in range(8000):
            k1 = f(p)
            k2 = f(p + 0.5 * step * k1)
            k3 = f(p + 0.5 * step * k2)
            k4 = f(p + step * k3)
            p = p + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(p.copy())
            if in_gnd(p) or in_sig(p) or not (box[0] < p[0] < box[1] and box[2] < p[1] < box[3]):
                break
        lines.append(np.array(out))
    return lines, seeds


def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline decimation."""
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    ab = b - a
    L = np.hypot(*ab)
    if L == 0:
        dists = np.hypot(*(pts - a).T)
    else:
        dists = np.abs(ab[0] * (pts[:, 1] - a[1]) - ab[1] * (pts[:, 0] - a[0])) / L
    i = int(np.argmax(dists))
    if dists[i] > eps:
        return np.vstack([rdp(pts[: i + 1], eps)[:-1], rdp(pts[i:], eps)])
    return np.vstack([a, b])


def b64u8(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def stage_export(cache: Path, hf: float, hm: float, win, out: Path, n_lines: int) -> None:
    ev = json.loads((cache / f"eval_h{hf:g}_m{hm:g}.json").read_text())
    es = dict(np.load(cache / f"es_rf_h{hf:g}.npz"))
    md = dict(np.load(cache / f"modes_h{hm:g}_w{win[1] - win[0]:g}x{win[3] - win[2]:g}.npz"))
    # view in rib-centred coordinates (x = 0 at the rib centre, y = 0 at the bottom of the film)
    view = (-5.6, 5.6, -1.6, 3.4)
    vabs = (view[0] + X_RIB, view[1] + X_RIB, view[2], view[3])
    lines, _ = field_lines(es, vabs, n_lines)
    pl = []
    for ln in lines:
        q = rdp(ln, 0.006)
        q = q - np.array([X_RIB, 0.0])
        pl.append([round(float(v), 3) for v in q.ravel()])
    # |E_RF| map over the view (per volt), log-scaled to uint8
    NX, NY = 224, 100
    gx = np.linspace(vabs[0], vabs[1], NX)
    gy = np.linspace(vabs[2], vabs[3], NY)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    pts = np.stack([GX.ravel(), GY.ravel()], 1)
    itx = RegularGridInterpolator((es["xc"], es["yc"]), es["Ex"], bounds_error=False, fill_value=0.0)
    ity = RegularGridInterpolator((es["xc"], es["yc"]), es["Ey"], bounds_error=False, fill_value=0.0)
    Emag = np.hypot(itx(pts), ity(pts)).reshape(NX, NY)  # V/um per V
    metal = ((GY >= T_SLAB) & (GY <= T_SLAB + T_EL)) & ((GX <= W_SIG / 2) | (GX >= W_SIG / 2 + GAP))
    Emag[metal] = 0.0
    lo, hi = 0.01, 1.0  # V/um per volt, log range
    q = np.clip((np.log10(np.maximum(Emag, 1e-9)) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)), 0, 1)
    emap = np.round(q * 255).astype(np.uint8).T[::-1]  # rows top to bottom
    # field in the rib centre (per volt) for the label
    ec = RegularGridInterpolator((es["xc"], es["yc"]), es["Ex"])([[X_RIB, 0.45]])[0]
    # optical intensity maps (|E|^2 normalised to peak), cropped
    xn, yn = md["xn"], md["yn"]
    crop = (-2.6, 2.6, -1.4, 1.8)
    ix = (xn >= crop[0] - 1e-9) & (xn <= crop[1] + 1e-9)
    iy = (yn >= crop[2] - 1e-9) & (yn <= crop[3] + 1e-9)
    maps = {}
    for pre in ("te_", "tm_"):
        I = np.abs(md[pre + "Ex"]) ** 2 + np.abs(md[pre + "Ey"]) ** 2 + np.abs(md[pre + "Ez"]) ** 2
        I = I[ix][:, iy]
        I = I / I.max()
        # resample to a 0.025 um grid if finer
        step = max(1, round(0.025 / float(md["h"])))
        I = I[::step, ::step]
        maps[pre] = {"nx": int(I.shape[0]), "ny": int(I.shape[1]), "b64": b64u8(np.round(np.sqrt(I) * 255).T[::-1])}
        # dominant transverse polarization direction and its peak location
        ax = float(np.sum(np.abs(md[pre + "Ex"]) ** 2))
        ay = float(np.sum(np.abs(md[pre + "Ey"]) ** 2))
        maps[pre]["pol"] = "x" if ax > ay else "y"
        Ipk = np.abs(md[pre + "Ex"]) ** 2 + np.abs(md[pre + "Ey"]) ** 2
        k = np.unravel_index(np.argmax(Ipk), Ipk.shape)
        maps[pre]["peak"] = [round(float(xn[k[0]]), 3), round(float(yn[k[1]]), 3)]
        maps[pre]["te_frac"] = round(float(md[pre + "frac_te"]), 4)
        maps[pre]["neff"] = round(float(md[pre + "neff"]), 4)
    xs_crop = [float(xn[ix][0]), float(xn[ix][-1]), float(yn[iy][0]), float(yn[iy][-1])]
    rf = ev["variants"]["rf"]
    lf = ev["variants"]["lf"]
    hb = RIB_BOT_HALF
    data = {
        "units": "um; x lateral (crystal z), y up (crystal x), origin at the rib centre on the oxide surface",
        "view": list(view),
        "geom": {
            "si_top": -T_BOX, "box_top": 0.0, "slab_top": T_SLAB, "film_top": T_FILM,
            "rib": [[-hb, T_SLAB], [hb, T_SLAB], [W_TOP / 2, T_FILM], [-W_TOP / 2, T_FILM]],
            "signal": [view[0] - 10, W_SIG / 2 - X_RIB, T_SLAB, T_SLAB + T_EL],
            "ground": [W_SIG / 2 + GAP - X_RIB, view[1] + 10, T_SLAB, T_SLAB + T_EL],
        },
        "lines": pl,
        "emap": {"nx": NX, "ny": NY, "extent": list(view), "scale": "log10 |E| from 0.01 to 1 V/um per volt", "b64": b64u8(emap)},
        "modes": {"extent": [round(v, 4) for v in xs_crop], "TE": maps["te_"], "TM": maps["tm_"]},
        "numbers": {
            "E_centre_V_per_um_per_V": round(float(ec), 4),
            "gamma_TE": round(rf["TE"]["gamma_x"], 3), "gamma_TM": round(rf["TM"]["gamma_y"], 3),
            "neff_TE": round(ev["te_neff"], 4), "neff_TM": round(ev["tm_neff"], 4),
            "vpil_TE_rf": round(rf["vpil_te_vcm"], 2), "vpil_TM_rf": round(rf["vpil_tm_vcm"], 2),
            "ratio_rf": round(rf["vpi_ratio_tm_te"], 2), "ratio_lf": round(lf["vpi_ratio_tm_te"], 2),
            "gamma_TE_lf": round(lf["TE"]["gamma_x"], 3), "gamma_TM_lf": round(lf["TM"]["gamma_y"], 3),
            "r33_rf": VARIANTS["rf"]["r"]["r33"], "r13_rf": VARIANTS["rf"]["r"]["r13"],
            "Z0_ohm": round(rf["Z0_ohm"], 1), "n_rf": round(rf["n_rf"], 3),
        },
        "meta": {
            "stack": "600 nm x-cut LiNbO3, 300 nm etch, 1.0 um top width, 60 deg sidewalls, on 4.7 um SiO2 on Si, air above; "
                     "Au 0.8 um on the slab, signal 20 um, gaps 5 um, grounds 100 um; waveguide along crystal y, one rib centred in each gap",
            "rf": "2D quasi-static finite-volume solve, clamped permittivity eps11 44.3, eps33 27.9 (Weis and Gaylord, Appl. Phys. A 37, 191 (1985), Table 1)",
            "optics": "full-vector finite-difference modes at 1550 nm; n_o, n_e from Zelmon et al., JOSA B 14, 3319 (1997); SiO2 Malitson, JOSA 55, 1205 (1965)",
            "pockels": "r33 30.8, r13 8.6, r22 3.4, r51 28 pm/V clamped, 633 nm (Weis and Gaylord 1985, Table 4); low-frequency set r33 32.2, r13 10.0, r22 6.81, r51 32 pm/V",
            "gamma": "Gamma = (g / V) * integral over LN of E_RF,z |E_opt|^2 / integral of |E_opt|^2 (dominant optical component; g = 5 um)",
            "generated": dt.date.today().isoformat(),
        },
    }
    js = ("/* Computed cross-section fields for the ferroelectric course, rods keyframe. Generated by tools/fields/rods_fields.py;\n"
          "   do not edit by hand. See tools/fields/README.md for geometry, materials and method. */\n"
          "window.LMS_DRAW_DATA = window.LMS_DRAW_DATA || {};\n"
          "window.LMS_DRAW_DATA.rodsFields = " + json.dumps(data, separators=(",", ":")) + ";\n")
    out.write_text(js)
    log(f"export: {out.name} {len(js) / 1024:.1f} KB, {len(pl)} lines, {sum(len(p) for p in pl) // 2} points")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["es", "modes", "eval", "export", "all"])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--hes", type=float, default=0.02, help="electrostatic grid spacing near the gap (um)")
    ap.add_argument("--h", type=float, default=0.025, help="mode solver grid (um)")
    ap.add_argument("--win", type=float, nargs=4, default=[-4.0, 4.0, -2.5, 2.5])
    ap.add_argument("--lines", type=int, default=40)
    ap.add_argument("--out", default=str(HERE.parent.parent / "anims" / "fe_rods_fields_data.js"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    win = tuple(a.win)
    if a.stage in ("es", "all"):
        stage_es(cache, a.hes, a.force)
    if a.stage in ("modes", "all"):
        stage_modes(cache, a.h, a.force, win)
    if a.stage in ("eval", "all"):
        stage_eval(cache, a.hes, a.h, win)
    if a.stage in ("export", "all"):
        stage_export(cache, a.hes, a.h, win, Path(a.out), a.lines)


if __name__ == "__main__":
    main()
