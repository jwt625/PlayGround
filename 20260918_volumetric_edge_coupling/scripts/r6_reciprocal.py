"""R6a: reciprocal Hz guide-mode launch (declared reconstruction).

Launches the receiving-guide mode from the output toward the interaction and
measures the guide reflection, the emitted field in silicon, and the overlap
of the emitted field with the time-reversed incident Gaussian.  Thickness,
angles, wedge and source-height convention are unchanged.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from src.config import CouplerConfig
from src.geometry import build_geometry, make_plan
from src.ports import add_oblique_mode_monitor, add_same_aperture_flux, oblique_modal_powers
from src.units import freq_from_wavelength_nm


def run(cfg: CouplerConfig, res: float, until: float, out: str) -> None:
    cfg = CouplerConfig.from_dict(cfg.to_dict())
    cfg.numerics.resolution = res
    cfg.numerics.until_after_sources = until
    plan = make_plan(cfg)
    geom, plan, idx = build_geometry(cfg)
    g = cfg.geometry
    s_mon = g.stack_start_um + g.stack_length_um + min(cfg.monitor.output_offset_um, g.guide_ext_down_um)
    fcen = freq_from_wavelength_nm(cfg.source.wavelength_nm)
    sz = mp.Vector3(plan.domain["sx"], plan.domain["sy"], 0)

    r = plan.r_top + s_mon * plan.u
    c_paper = r + (g.gap_at(s_mon) + g.twg_nm / 2000.0) * plan.n
    center = plan.meep_xy(c_paper[0], c_paper[1])
    kdir = mp.Vector3(-plan.d[0], -plan.d[1], 0)
    parity = mp.ODD_Z if cfg.source.branch == "ez" else mp.EVEN_Z
    src = mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=0.1), center=center,
        size=mp.Vector3(0, 12.0, 0), direction=mp.NO_DIRECTION,
        eig_kpoint=kdir, eig_band=1, eig_parity=parity, eig_match_freq=True,
    )
    sim = mp.Simulation(cell_size=sz, resolution=cfg.numerics.resolution, geometry=geom,
                        sources=[src], boundary_layers=[mp.PML(cfg.numerics.pml_um)],
                        default_material=idx.medium("clad"), dimensions=2)
    dmap = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz], fcen, 0, 1,
                              center=mp.Vector3(), size=sz)
    mmon, mcenter, msize = add_oblique_mode_monitor(sim, cfg, plan, s_mon - 4.0)
    fmon = add_same_aperture_flux(sim, cfg, mcenter, msize)
    sim.run(until_after_sources=cfg.numerics.until_after_sources)

    from src.ports import DF_NFREQ
    ic = DF_NFREQ // 2
    modal = oblique_modal_powers(sim, mmon, cfg, plan)
    # launched mode travels toward -d, so its "backward" coefficient is the incident
    b1 = modal[1]
    p_inc = b1["backward"][ic]  # toward -d
    p_ref = b1["forward"][ic]   # reflected back toward +d
    p_inc_native = abs(mp.get_fluxes(fmon)[ic])
    print(f"R6 launched guide modal={p_inc:.2f} native={p_inc_native:.2f} "
          f"reflected={p_ref:.2f} reflection={p_ref/max(p_inc,1e-12):.4f}")

    comp = {"Ex": mp.Ex, "Ey": mp.Ey, "Ez": mp.Ez, "Hx": mp.Hx, "Hy": mp.Hy, "Hz": mp.Hz}
    arrs = {c: np.asarray(sim.get_dft_array(dmap, comp[c], 0)) for c in comp}
    if cfg.source.branch == "ez":
        Sx = -np.real(arrs["Ez"] * np.conj(arrs["Hy"]))
        field = arrs["Ez"]
    else:
        Sx = np.real(arrs["Ey"] * np.conj(arrs["Hz"]))
        field = arrs["Hz"]
    x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=sz)
    xpaper = np.asarray(x) + plan.domain["cell_center"][0]
    ypaper = np.asarray(y) + plan.domain["cell_center"][1]
    dx = abs(xpaper[1] - xpaper[0]) if len(xpaper) > 1 else 1.0
    dy = abs(ypaper[1] - ypaper[0]) if len(ypaper) > 1 else 1.0

    # reference column in homogeneous silicon
    ix_ref = int(np.argmin(np.abs(xpaper - cfg.source.x_source_um)))
    sx_col = Sx[ix_ref, :]
    prof = field[ix_ref, :]
    mask = sx_col < 0  # outgoing toward -x
    p_emit = -np.sum(np.minimum(sx_col, 0)) * dy
    gauss = np.exp(-((ypaper - cfg.source.y_source_um) / cfg.source.w0_um) ** 2)
    num = np.sum(np.conj(gauss[mask]) * prof[mask]) * dy
    den = np.sqrt(np.sum(np.abs(gauss) ** 2) * dy * np.sum(np.abs(prof[mask]) ** 2) * dy) if mask.any() else 0
    overlap = abs(num / den) if den > 0 else 0.0
    m_pow = overlap**2
    p_target = p_emit * m_pow
    print(f"R6 emitted -x power at x={xpaper[ix_ref]:.2f} um = {p_emit:.3f} "
          f"(f_collection={p_emit/max(p_inc_native,1e-12):.4f}); "
          f"mode_overlap_amp={overlap:.4f} M={m_pow:.4f} P_target={p_target:.4f}")
    print(f"R6 eta_reverse(native)={p_target/max(p_inc_native,1e-12):.5f} "
          f"eta_reverse(guide band)={p_target/max(p_inc,1e-12):.5f}")

    fig, ax = plt.subplots(figsize=(12, 8))
    v = np.percentile(np.abs(Sx), 99.5)
    pcm = ax.pcolormesh(xpaper, ypaper, Sx.T, shading="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.plot(plan.guide_poly[:, 0], plan.guide_poly[:, 1], "k-", lw=1.2)
    ax.plot([xpaper[ix_ref], xpaper[ix_ref]], [ypaper.min(), ypaper.max()], "g--", lw=1, label="Si ref plane")
    ax.set_aspect("equal")
    ax.set_xlim(plan.domain["x_lo"], plan.domain["x_hi"])
    ax.set_ylim(plan.domain["y_lo"], plan.domain["y_hi"])
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(f"R6 reverse {cfg.source.branch} launch: Sx (blue = -x outgoing), cfg {cfg.config_hash()}")
    ax.legend(fontsize=8)
    fig.colorbar(pcm, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--branch", default="hz")
    ap.add_argument("--res", type=float, default=25.0)
    ap.add_argument("--until", type=float, default=80.0)
    ap.add_argument("--out", default="reports/figs/fig_r6_reverse_hz.png")
    args = ap.parse_args()
    cfg = CouplerConfig.load(args.config)
    cfg.source.branch = args.branch
    run(cfg, args.res, args.until, args.out)


if __name__ == "__main__":
    main()
