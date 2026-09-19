"""Slab mode solvers for the receiving guide (T03 / R1).

Two independent routes:

* :func:`analytic_slab_modes` uses a transfer-matrix formulation with decaying
  cladding conditions (no ``tan`` asymptotes, so no spurious roots).
* :func:`mpb_slab_modes` uses Meep/MPB through an axis-aligned mode monitor on
  an isolated straight guide.

Polarizations use the Meep branch labels:
``"ez"`` = E out of plane (TE-like, Meep ``Ez``),
``"hz"`` = H out of plane (TM-like, Meep ``Hz``).
"""

from __future__ import annotations

import numpy as np


def _k0(wavelength_um: float) -> float:
    return 2.0 * np.pi / wavelength_um


def _layer_matrix_te(k, n, t):
    c = np.cos(k * t)
    s = np.sin(k * t)
    return np.array([[c, s / k], [-k * s, c]], dtype=complex)


def _layer_matrix_tm(k, n, t):
    c = np.cos(k * t)
    s = np.sin(k * t)
    return np.array([[c, (n * n / k) * s], [-(k / (n * n)) * s, c]], dtype=complex)


def _boundary_state_te(beta, stack, n_top, n_bot, wavelength_um):
    """Return ``(state, q_b)`` after propagating from the top cladding."""
    k0 = _k0(wavelength_um)
    q_t = np.sqrt(beta**2 - (k0 * np.maximum(n_top, 1e-9)) ** 2 + 0j)
    q_b = np.sqrt(beta**2 - (k0 * np.maximum(n_bot, 1e-9)) ** 2 + 0j)
    state = np.array([1.0, q_t], dtype=complex)
    for n, t in stack:
        k = np.sqrt((k0 * n) ** 2 - beta**2 + 0j)
        state = _layer_matrix_te(k, n, t) @ state
    return state, q_b


def _boundary_state_tm(beta, stack, n_top, n_bot, wavelength_um):
    k0 = _k0(wavelength_um)
    q_t = np.sqrt(beta**2 - (k0 * n_top) ** 2 + 0j)
    q_b = np.sqrt(beta**2 - (k0 * n_bot) ** 2 + 0j)
    state = np.array([1.0, q_t / n_top**2], dtype=complex)
    for n, t in stack:
        k = np.sqrt((k0 * n) ** 2 - beta**2 + 0j)
        state = _layer_matrix_tm(k, n, t) @ state
    return state, q_b / n_bot**2


def _residual_te(beta, stack, n_top, n_bot, wavelength_um):
    """Undivided lower-cladding residual (no division: no pole roots)."""
    state, q_b = _boundary_state_te(beta, stack, n_top, n_bot, wavelength_um)
    return state[1] + q_b * state[0]


def _residual_tm(beta, stack, n_top, n_bot, wavelength_um):
    state, q_b = _boundary_state_tm(beta, stack, n_top, n_bot, wavelength_um)
    return state[1] + q_b * state[0]


def normalized_boundary_residual(beta, stack, n_top, n_bot, wavelength_um, polarization):
    """Dimensionless residual used to accept a root (threshold ~1e-8)."""
    if polarization == "ez":
        state, q_b = _boundary_state_te(beta, stack, n_top, n_bot, wavelength_um)
    else:
        state, q_b = _boundary_state_tm(beta, stack, n_top, n_bot, wavelength_um)
    resid = state[1] + q_b * state[0]
    scale = abs(state[1]) + abs(q_b * state[0]) + 1e-300
    return abs(resid) / scale


def _scan_roots(func, lo, hi, n=20000):
    """Bracket sign changes of ``Re(func)`` and refine on the real part.

    The caller must still reject roots with a large normalised boundary
    residual; this routine only finds candidate brackets.
    """
    from scipy.optimize import brentq

    xs = np.linspace(lo + 1e-12, hi - 1e-12, n)
    vals = np.array([func(x).real for x in xs])
    roots = []
    for i in range(len(xs) - 1):
        a, b = vals[i], vals[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0:
            roots.append(xs[i])
        elif a * b < 0:
            roots.append(brentq(lambda z: func(z).real, xs[i], xs[i + 1], xtol=1e-14, rtol=1e-15))
    return roots


def slab_profile(
    core_index: float,
    thickness_um: float,
    cladding_top: float,
    cladding_bot: float,
    wavelength_um: float,
    polarization: str,
    n_eff: float,
    n_core_points: int = 200,
    n_clad_points: int = 200,
    clad_span_um: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the transverse field of a bound mode, normalised to max |E|.

    Returns ``(y, field, material)`` where ``material`` is 0/1/2 for
    top cladding / core / bottom cladding.
    """
    k0 = _k0(wavelength_um)
    beta = k0 * n_eff
    d = thickness_um
    q_t = np.sqrt(beta**2 - (k0 * cladding_top) ** 2 + 0j).real
    q_b = np.sqrt(beta**2 - (k0 * cladding_bot) ** 2 + 0j).real
    kappa = np.sqrt((k0 * core_index) ** 2 - beta**2 + 0j).real
    if polarization == "ez":
        state = np.array([1.0, q_t], dtype=complex)
        mat = lambda k, n, t: _layer_matrix_te(k, n, t)
        comp = lambda s, k, yp: s[0] * np.cos(k * yp) + s[1] / k * np.sin(k * yp)
    else:
        state = np.array([1.0, q_t / cladding_top**2], dtype=complex)
        mat = lambda k, n, t: _layer_matrix_tm(k, n, t)
        comp = lambda s, k, yp: s[0] * np.cos(k * yp) + s[1] / (k / core_index**2) * np.sin(k * yp)
    yt = np.linspace(d / 2, d / 2 + clad_span_um, n_clad_points)
    top = state[0] * np.exp(-q_t * (yt - d / 2))
    yc = np.linspace(d / 2, -d / 2, n_core_points)
    yp = d / 2 - yc
    core = comp(state, kappa, yp)
    sb = mat(kappa, core_index, d) @ state
    yb = np.linspace(-d / 2, -d / 2 - clad_span_um, n_clad_points)
    bot = sb[0] * np.exp(q_b * (yb + d / 2))
    y = np.concatenate([yt, yc, yb])
    field = np.concatenate([top, core, bot])
    material = np.concatenate([np.zeros(n_clad_points), np.ones(n_core_points), 2 * np.ones(n_clad_points)])
    field = field / np.max(np.abs(field))
    return y, field, material


def analytic_slab_modes(
    core_index: float,
    thickness_um: float,
    cladding_top: float,
    cladding_bot: float,
    wavelength_um: float,
    polarization: str,
    n_roots: int = 6,
    resolution: int = 20000,
) -> list[float]:
    """Bound slab modes of a single core sandwiched by two semi-infinite claddings."""
    k0 = _k0(wavelength_um)
    n_hi = core_index
    n_lo = max(cladding_top, cladding_bot)
    stack = [(core_index, thickness_um)]
    if polarization == "ez":
        func = lambda b: _residual_te(b, stack, cladding_top, cladding_bot, wavelength_um)
    elif polarization == "hz":
        func = lambda b: _residual_tm(b, stack, cladding_top, cladding_bot, wavelength_um)
    else:
        raise ValueError(polarization)
    roots_beta = _scan_roots(func, k0 * n_lo, k0 * n_hi, resolution)
    accepted = []
    for b in roots_beta:
        nres = normalized_boundary_residual(
            b, stack, cladding_top, cladding_bot, wavelength_um, polarization
        )
        if nres < 1e-8:
            accepted.append(b / k0)
    return sorted(accepted, reverse=True)[:n_roots]


def solve_straight_guide(
    core_index: float,
    thickness_um: float,
    cladding_top: float,
    cladding_bot: float,
    wavelength_um: float,
    polarization: str,
    n_bands: int = 3,
    resolution: float = 100.0,
) -> list[float]:
    """Run a short straight-guide sim and return MPB n_eff per band."""
    import meep as mp

    fcen = 1.0 / wavelength_um
    cell_y = 8.0 * thickness_um + 2.0
    cell_x = 3.0
    pml = 1.0
    parity = mp.ODD_Z if polarization == "ez" else mp.EVEN_Z
    geo = [
        mp.Block(size=mp.Vector3(mp.inf, 2 * cell_y, 0),
                 center=mp.Vector3(0, cell_y, 0),
                 material=mp.Medium(epsilon=cladding_top**2)),
        mp.Block(size=mp.Vector3(mp.inf, thickness_um, 0),
                 center=mp.Vector3(0, 0, 0),
                 material=mp.Medium(epsilon=core_index**2)),
    ]
    sim = mp.Simulation(
        cell_size=mp.Vector3(cell_x, cell_y, 0),
        resolution=resolution,
        geometry=geo,
        default_material=mp.Medium(epsilon=cladding_bot**2),
        boundary_layers=[mp.PML(pml)],
        dimensions=2,
    )
    src = mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=0.05),
        center=mp.Vector3(-1, 0),
        size=mp.Vector3(0, cell_y - 2 * pml),
        direction=mp.NO_DIRECTION,
        eig_kpoint=mp.Vector3(1, 0, 0),
        eig_band=1,
        eig_parity=parity,
        eig_match_freq=True,
    )
    sim.sources = [src]
    mon = sim.add_mode_monitor(
        fcen, 0, 1,
        mp.ModeRegion(center=mp.Vector3(1.0, 0), size=mp.Vector3(0, cell_y - 2 * pml)),
    )
    sim.run(until_after_sources=30)
    bands = list(range(1, n_bands + 1))
    res = sim.get_eigenmode_coefficients(
        mon, bands, eig_parity=parity, direction=mp.NO_DIRECTION,
        kpoint_func=lambda f, n: mp.Vector3(1, 0, 0),
    )
    kpoints = np.atleast_1d(res.kpoints)
    return [float(k) / fcen for k in kpoints]
