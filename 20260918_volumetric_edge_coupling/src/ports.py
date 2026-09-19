"""Monitors and power extraction.

The receiving guide is inclined at ``theta_wg`` in the paper frame, so Meep's
axis-aligned mode monitors cannot be used directly.  We therefore accumulate
the complex DFT fields on an axis-aligned rectangle and sample them along a
line perpendicular to the guide, then integrate the Poynting flux through that
line.  ``get_dft_array`` returns Yee-grid fields already bilinearly collocated
to voxel centres; the returned array is indexed ``[ix, iy]`` (matching
``get_array_metadata``), and only components belonging to the active
polarization branch are populated, so we request exactly those.

``Tflux`` (signed net flux through the output aperture) is the primary
provisional observable until a power-normalised modal projection is validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import meep as mp
import numpy as np

from .config import CouplerConfig
from .geometry import Plan
from .units import freq_from_wavelength_nm, wavelength_nm_from_freq


_COMP = {
    "Ex": mp.Ex,
    "Ey": mp.Ey,
    "Ez": mp.Ez,
    "Hx": mp.Hx,
    "Hy": mp.Hy,
    "Hz": mp.Hz,
}
# 2D structures invariant in z: Ez decouples from (Ex, Ey).
BRANCH_COMPONENTS = {
    "ez": ("Ez", "Hx", "Hy"),
    "hz": ("Hz", "Ex", "Ey"),
}
DF_FREQ_WIDTH = 0.05
DF_NFREQ = 51


@dataclass
class ExtractionLine:
    center_xy: np.ndarray
    direction: np.ndarray  # unit guide direction (normal to the line)
    q_lo: float
    q_hi: float
    s_ext_um: float
    branch: str = "ez"

    def normal(self) -> np.ndarray:
        return np.array([-self.direction[1], self.direction[0]])

    def points(self, n: int = 240) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(self.q_lo, self.q_hi, n)
        m = self.normal()
        pts = self.center_xy[None, :] + t[:, None] * m[None, :]
        return pts, t


def extraction_line(cfg: CouplerConfig, plan: Plan) -> ExtractionLine:
    g = cfg.geometry
    s_ext = g.stack_start_um + g.stack_length_um + min(
        cfg.monitor.output_offset_um, g.guide_ext_down_um
    )
    r = plan.r_top + s_ext * plan.u
    return ExtractionLine(
        center_xy=r + plan.shift,
        direction=plan.d,
        q_lo=0.05,
        q_hi=cfg.monitor.aperture_um,
        s_ext_um=s_ext,
        branch=cfg.source.branch,
    )


def add_output_dft(sim: mp.Simulation, cfg: CouplerConfig, line: ExtractionLine):
    """Add a collocated-rectangle DFT monitor that encloses the output line."""
    fcen = freq_from_wavelength_nm(cfg.source.wavelength_nm)
    comps = BRANCH_COMPONENTS[cfg.source.branch]
    pts = np.vstack(
        [
            line.center_xy + line.q_lo * line.normal(),
            line.center_xy + line.q_hi * line.normal(),
        ]
    )
    lo = pts.min(axis=0) - 0.5
    hi = pts.max(axis=0) + 0.5
    center = mp.Vector3(0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0)
    size = mp.Vector3(hi[0] - lo[0], hi[1] - lo[1], 0)
    mon = sim.add_dft_fields(
        [_COMP[c] for c in comps], fcen, DF_FREQ_WIDTH, DF_NFREQ, center=center, size=size
    )
    return mon, center, size


def _bilinear(xs, ys, arr, px, py):
    """Interpolate ``arr[ix, iy]`` at paper-frame point (px, py)."""
    nx = len(xs)
    ny = len(ys)
    ix = int(np.clip(np.searchsorted(xs, px) - 1, 0, nx - 2))
    iy = int(np.clip(np.searchsorted(ys, py) - 1, 0, ny - 2))
    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]
    tx = 0.0 if x1 == x0 else (px - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (py - y0) / (y1 - y0)
    a = arr[ix, iy]
    b = arr[ix + 1, iy]
    c = arr[ix, iy + 1]
    d = arr[ix + 1, iy + 1]
    return (
        a * (1 - tx) * (1 - ty)
        + b * tx * (1 - ty)
        + c * (1 - tx) * ty
        + d * tx * ty
    )


def sample_field(sim, mon, center, size, comp: str, ifreq: int, pts: np.ndarray, xs, ys):
    arr = np.asarray(sim.get_dft_array(mon, _COMP[comp], ifreq))
    return np.array([_bilinear(xs, ys, arr, px, py) for px, py in pts])


def line_flux(sim: mp.Simulation, mon, center, size, line: ExtractionLine, ifreq: int) -> complex:
    """Signed net Poynting flux (per invariant length) through the output line."""
    x, y, z, w = sim.get_array_metadata(center=center, size=size)
    xs = np.asarray(x)
    ys = np.asarray(y)
    pts, t = line.points()
    dl = t[1] - t[0]
    n = line.direction
    comps = BRANCH_COMPONENTS[line.branch]
    vals = {c: sample_field(sim, mon, center, size, c, ifreq, pts, xs, ys) for c in comps}
    # Meep's dft_flux integrates Re(E x H*) with no 1/2 (see Meep v1.34.0
    # src/dft.cpp); this matches mp.get_fluxes.  The physical time average
    # carries the 1/2, but we must use the solver's own convention.
    if "Ez" in vals:
        Ez = vals["Ez"]
        Hx = vals["Hx"]
        Hy = vals["Hy"]
        Sx = -np.real(Ez * np.conj(Hy))
        Sy = np.real(Ez * np.conj(Hx))
    else:
        Hz = vals["Hz"]
        Ex = vals["Ex"]
        Ey = vals["Ey"]
        Sx = np.real(Ey * np.conj(Hz))
        Sy = -np.real(Ex * np.conj(Hz))
    return complex(np.sum((Sx * n[0] + Sy * n[1]) * dl))


# --------------------------------------------------------------------------
# Oblique-guide modal decomposition (Meep v1.34.0 route)
# --------------------------------------------------------------------------
def guide_centerline(cfg: CouplerConfig, plan: Plan, s_um: float) -> np.ndarray:
    g = cfg.geometry
    r = plan.r_top + s_um * plan.u
    return r + (g.gap_at(s_um) + g.twg_nm / 2000.0) * plan.n


def add_oblique_mode_monitor(
    sim: mp.Simulation,
    cfg: CouplerConfig,
    plan: Plan,
    s_mon: float,
    half_span_um: float = 6.0,
    single_freq: bool = False,
):
    """Axis-aligned line crossing the inclined guide, for oblique mode extraction."""
    center_paper = guide_centerline(cfg, plan, s_mon)
    center = plan.meep_xy(center_paper[0], center_paper[1])
    fcen = freq_from_wavelength_nm(cfg.source.wavelength_nm)
    if single_freq:
        mon = sim.add_mode_monitor(
            fcen, 0, 1,
            mp.ModeRegion(center=center, size=mp.Vector3(0, 2 * half_span_um, 0)),
        )
    else:
        mon = sim.add_mode_monitor(
            fcen, DF_FREQ_WIDTH, DF_NFREQ,
            mp.ModeRegion(center=center, size=mp.Vector3(0, 2 * half_span_um, 0)),
        )
    return mon, center


def oblique_modal_powers(
    sim: mp.Simulation,
    mon,
    cfg: CouplerConfig,
    plan: Plan,
    bands: list[int] | None = None,
) -> dict:
    """Forward/backward guided power per band using the oblique kpoint route."""
    bands = bands or [1, 2, 3, 4, 5, 6]
    kdir = mp.Vector3(plan.d[0], plan.d[1], 0)
    parity = mp.ODD_Z if cfg.source.branch == "ez" else mp.EVEN_Z
    res = sim.get_eigenmode_coefficients(
        mon, bands, eig_parity=parity, direction=mp.NO_DIRECTION,
        kpoint_func=lambda f, n: kdir,
    )
    fcen = freq_from_wavelength_nm(cfg.source.wavelength_nm)
    try:
        freqs = np.array(mp.get_flux_freqs(mon), dtype=float)
    except Exception:  # noqa: BLE001
        freqs = np.linspace(fcen - DF_FREQ_WIDTH / 2, fcen + DF_FREQ_WIDTH / 2, DF_NFREQ)
    nfreq = len(freqs)
    alpha = np.asarray(res.alpha)  # (nbands, nfreq, 2): forward, backward
    kp_flat = list(res.kpoints)  # flattened index band*nfreq + freq
    if len(kp_flat) != len(bands) * nfreq:
        raise RuntimeError(
            f"kpoints length {len(kp_flat)} != nbands*nfreq {len(bands)*nfreq}"
        )
    out = {}
    for bi, band in enumerate(bands):
        neff = np.zeros(nfreq)
        kvec = np.zeros((nfreq, 2))
        for fi in range(nfreq):
            kp = kp_flat[bi * nfreq + fi]
            kvec[fi] = [kp.x, kp.y]
            # kpoint is in units of 2*pi/a; n_eff = |k| / f
            neff[fi] = float(np.linalg.norm([kp.x, kp.y, kp.z])) / freqs[fi]
        out[band] = {
            "neff": neff,                # per frequency
            "kvec": kvec,                # per frequency, (kx, ky)
            "forward": np.abs(alpha[bi, :, 0]) ** 2,
            "backward": np.abs(alpha[bi, :, 1]) ** 2,
            "vgrp": None,
        }
    return out


# --------------------------------------------------------------------------
# Incident reference
# --------------------------------------------------------------------------
def run_incident_reference(cfg: CouplerConfig, plan: Plan) -> dict:
    """Homogeneous-silicon reference with the same source, mesh and PML."""
    from .sources import build_source
    from .materials import IndexSet

    idx = IndexSet(n_si=cfg.materials.n_si, n_oxide=cfg.materials.n_oxide,
                   cladding=cfg.materials.cladding)
    fcen = freq_from_wavelength_nm(cfg.source.wavelength_nm)
    src = build_source(cfg, plan)
    freg = mp.FluxRegion(
        center=plan.meep_xy(plan.points["x_impact"] - 2.0, cfg.source.y_source_um),
        size=mp.Vector3(0, 8 * cfg.source.w0_um, 0),
    )
    sim = mp.Simulation(
        cell_size=mp.Vector3(plan.domain["sx"], plan.domain["sy"], 0),
        resolution=cfg.numerics.resolution,
        sources=src,
        boundary_layers=[mp.PML(cfg.numerics.pml_um)],
        default_material=idx.medium("si"),
        dimensions=2,
    )
    flux = sim.add_flux(fcen, DF_FREQ_WIDTH, DF_NFREQ, freg)
    sim.run(until_after_sources=cfg.numerics.until_after_sources)
    freqs = np.array(mp.get_flux_freqs(flux))
    vals = np.array(mp.get_fluxes(flux), dtype=float)
    return {"freqs": freqs, "flux": vals}


def freqs_to_nm(freqs: np.ndarray) -> np.ndarray:
    return np.array([wavelength_nm_from_freq(f) for f in freqs])
