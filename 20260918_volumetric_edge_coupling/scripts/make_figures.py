"""Generate review artifacts: geometry, gap profile, mode profiles, Poynting map.

Static figures need no solver.  ``--poynting`` runs a pulsed 2D nominal case and
plots the time-averaged Poynting flow at 1550 nm.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from src.config import CouplerConfig
from src.geometry import make_plan
from src.modes import analytic_slab_modes, slab_profile


def fig_geometry(cfg: CouplerConfig, out: Path) -> None:
    plan = make_plan(cfg)
    g = cfg.geometry
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, zoom in [(axes[0], False), (axes[1], True)]:
        ax.add_patch(Polygon(plan.oxide_poly, closed=True, facecolor="#8ecae6",
                             edgecolor="k", lw=1.0, label="oxide gap"))
        ax.add_patch(Polygon(plan.guide_poly, closed=True, facecolor="#ffb703",
                             edgecolor="k", lw=1.0, label="Si guide (262 nm)"))
        # sidewall interface line
        rr = plan.r_top
        ss = np.linspace(-40, 60, 2)
        ax.plot(rr[0] + ss * math.cos(g.alpha_rad), rr[1] - ss * math.sin(g.alpha_rad),
                "r--", lw=1.0, label="KOH sidewall 54.74 deg")
        # source beam axis + Gaussian envelope
        y_src = cfg.source.y_source_um
        w0 = cfg.source.w0_um
        xs = np.linspace(cfg.source.x_source_um, plan.points["x_impact"] + 2, 60)
        ax.plot(xs, np.full_like(xs, y_src), "k-", lw=0.8)
        ax.plot(xs, np.full_like(xs, y_src + 2 * w0), "k:", lw=0.6)
        ax.plot(xs, np.full_like(xs, y_src - 2 * w0), "k:", lw=0.6)
        ax.plot(cfg.source.x_source_um, y_src, "kv", ms=8, label="Gaussian source")
        ax.plot(*plan.points["impact_xy"], "ko", ms=7, mfc="w", label="TIR impact")
        # monitors
        line = plan  # extraction line drawn via plan points
        sx, sy = plan.domain["sx"], plan.domain["sy"]
        cx, cy = plan.domain["cell_center"]
        ax.add_patch(plt.Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fill=False,
                                   ec="0.5", lw=0.8, ls=":", label="cell + PML"))
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal")
        if zoom:
            ip = plan.points["impact_xy"]
            ax.set_xlim(ip[0] - 6, ip[0] + 18)
            ax.set_ylim(ip[1] - 16, ip[1] + 6)
            ax.set_title("geometry zoom (wedge + guide)")
        else:
            ax.set_xlim(cx - sx / 2, cx + sx / 2)
            ax.set_ylim(cy - sy / 2, cy + sy / 2)
            ax.set_title("geometry full (paper frame)")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um above cavity base)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def fig_gap_profile(cfg: CouplerConfig, out: Path) -> None:
    plan = make_plan(cfg)
    g = cfg.geometry
    s = np.linspace(g.stack_start_um - g.guide_ext_up_um,
                    g.stack_start_um + g.stack_length_um + g.guide_ext_down_um, 400)
    gap = np.array([g.gap_at(v) * 1000 for v in s])
    # beam amplitude along the sidewall (transverse coordinate is y)
    ys = plan.r_top[1] - s * math.sin(g.alpha_rad)
    amp = np.exp(-((ys - cfg.source.y_source_um) / cfg.source.w0_um) ** 2)
    s_imp = g.s_impact(cfg.source.y_source_um)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, gap, "b-", label="oxide gap g(s)")
    ax.axvline(s_imp, color="k", ls="--", lw=1, label=f"beam impact s={s_imp:.2f} um")
    ax.plot(s, amp * gap.max(), color="0.5", ls=":", label="beam amplitude along sidewall (a.u.)")
    ax.axvspan(g.stack_start_um, g.stack_start_um + g.stack_length_um, color="orange", alpha=0.12,
               label="interaction stack")
    ax.set_xlabel("sidewall coordinate s (um)")
    ax.set_ylabel("gap (nm)")
    ax.set_title(f"gap profile; gap@impact = {g.gap_at(s_imp)*1000:.0f} nm")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def fig_mode_profiles(cfg: CouplerConfig, out: Path) -> None:
    from src.materials import IndexSet
    idx = IndexSet(n_si=cfg.materials.n_si, n_oxide=cfg.materials.n_oxide,
                   cladding=cfg.materials.cladding)
    n_clad = idx.n("clad")
    t = cfg.geometry.twg_nm / 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, pol, label in [(axes[0], "ez", "|Ez|"), (axes[1], "hz", "|Hz|")]:
        roots = analytic_slab_modes(cfg.materials.n_si, t, n_clad,
                                    cfg.materials.n_oxide, cfg.source.wavelength_nm / 1000.0, pol)
        y, f, m = slab_profile(cfg.materials.n_si, t, n_clad,
                               cfg.materials.n_oxide, cfg.source.wavelength_nm / 1000.0, pol, roots[0])
        # horizontal band for the core: position is on the vertical axis
        ax.axhspan(-t / 2, t / 2, color="orange", alpha=0.2)
        ax.axhline(t / 2, color="0.4", lw=0.6)
        ax.axhline(-t / 2, color="0.4", lw=0.6)
        ax.plot(np.abs(f), y, "b-", label=f"{label}, n_eff={roots[0]:.4f}")
        ax.set_xlabel(f"{label} (normalised)")
        ax.set_title(f"{label} branch, t={cfg.geometry.twg_nm:.0f} nm")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("y (um)  [Si core shaded]")
    fig.suptitle(f"isolated receiving-guide modes at 1550 nm (cfg {cfg.config_hash()})")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out)


def fig_poynting(cfg: CouplerConfig, out: Path, res: float = 25.0, until: float = 60.0) -> None:
    import meep as mp
    from src.geometry import build_geometry
    from src.sources import build_source
    from src.ports import (extraction_line, line_flux, DF_FREQ_WIDTH, DF_NFREQ,
                           add_output_dft, run_incident_reference)

    cfg = CouplerConfig.from_dict(cfg.to_dict())
    cfg.numerics.resolution = res
    cfg.numerics.until_after_sources = until
    plan = make_plan(cfg)
    geom, plan, idx = build_geometry(cfg)
    fcen = 1.0 / (cfg.source.wavelength_nm / 1000.0)
    sim = mp.Simulation(
        cell_size=mp.Vector3(plan.domain["sx"], plan.domain["sy"], 0),
        resolution=cfg.numerics.resolution, geometry=geom, sources=build_source(cfg, plan),
        boundary_layers=[mp.PML(cfg.numerics.pml_um)], default_material=idx.medium("clad"), dimensions=2,
    )
    comps = [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]
    dmon = sim.add_dft_fields(comps, fcen, 0, 1, center=mp.Vector3(), size=mp.Vector3(plan.domain["sx"], plan.domain["sy"], 0))
    line = extraction_line(cfg, plan)
    fmon, fc, fs = add_output_dft(sim, cfg, line)
    sim.run(until_after_sources=cfg.numerics.until_after_sources)
    arrs = {c: np.asarray(sim.get_dft_array(dmon, c, 0)) for c in comps}
    if cfg.source.branch == "ez":
        Sx = -np.real(arrs[mp.Ez] * np.conj(arrs[mp.Hy]))
        Sy = np.real(arrs[mp.Ez] * np.conj(arrs[mp.Hx]))
    else:
        Sx = np.real(arrs[mp.Ey] * np.conj(arrs[mp.Hz]))
        Sy = -np.real(arrs[mp.Ex] * np.conj(arrs[mp.Hz]))
    Smag = np.hypot(Sx, Sy)
    x, y, z, w = sim.get_array_metadata(center=mp.Vector3(), size=mp.Vector3(plan.domain["sx"], plan.domain["sy"], 0))
    xs = np.asarray(x) + plan.domain["cell_center"][0]
    ys = np.asarray(y) + plan.domain["cell_center"][1]
    fig, ax = plt.subplots(figsize=(12, 9))
    pcm = ax.pcolormesh(xs, ys, Smag.T, shading="auto", cmap="inferno",
                        vmin=0, vmax=np.percentile(Smag, 99.5))
    # sparse quiver of flow direction
    step = max(1, len(xs) // 40)
    ax.quiver(xs[::step], ys[::step], Sx[::step, ::step].T, Sy[::step, ::step].T,
              color="cyan", alpha=0.5, scale=None)
    ax.plot(plan.oxide_poly[:, 0], plan.oxide_poly[:, 1], "c-", lw=1)
    ax.plot(plan.guide_poly[:, 0], plan.guide_poly[:, 1], "w-", lw=1.5)
    rr = plan.r_top
    ss = np.linspace(-40, 60, 2)
    ax.plot(rr[0] + ss * math.cos(cfg.geometry.alpha_rad), rr[1] - ss * math.sin(cfg.geometry.alpha_rad), "r--", lw=1)
    ax.set_aspect("equal")
    ax.set_xlim(plan.domain["x_lo"], plan.domain["x_hi"])
    ax.set_ylim(plan.domain["y_lo"], plan.domain["y_hi"])
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    pinc = run_incident_reference(cfg, plan)["flux"][DF_NFREQ // 2]
    pout = line_flux(sim, fmon, fc, fs, line, DF_NFREQ // 2).real
    ax.set_title(
        f"time-averaged Poynting |S|, 1550 nm, branch={cfg.source.branch} "
        f"(cfg {cfg.config_hash()}); net output line flux/Pinc={pout/pinc*100:.2f}%"
    )
    fig.colorbar(pcm, ax=ax, fraction=0.04, label="|S|")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("saved", out, f"eta_net={pout/pinc*100:.3f}%")


def write_provenance(cfg: CouplerConfig, outdir: Path, branch: str, args) -> None:
    import time
    try:
        import meep as mp
        solver = f"meep={mp.__version__}"
    except Exception:  # noqa: BLE001
        solver = "meep=unknown"
    path = outdir / "provenance.md"
    if not path.exists():
        path.write_text(
            "# Figure provenance\n\n"
            "Each row records the configuration hash, effective branch, mesh, "
            "run time and solver build for the figure set.\n\n"
            "| timestamp | config | branch | resolution | until | solver |\n"
            "|---|---|---|---|---|---|\n"
        )
    path.write_text(
        path.read_text()
        + f"| {time.strftime('%Y-%m-%dT%H:%M:%S%z')} | {cfg.config_hash()} | {branch} | "
        f"{args.res}/{args.until} | {solver} |\n"
    )
    print("saved", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--outdir", default="reports/figs")
    ap.add_argument("--poynting", action="store_true")
    ap.add_argument("--branch", default=None, help="override source branch for the field figure")
    ap.add_argument("--res", type=float, default=25.0)
    ap.add_argument("--until", type=float, default=60.0)
    args = ap.parse_args()
    cfg = CouplerConfig.load(args.config)
    branch = args.branch or cfg.source.branch
    cfg.source.branch = branch
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig_geometry(cfg, outdir / "fig_geometry.png")
    fig_gap_profile(cfg, outdir / "fig_gap_profile.png")
    fig_mode_profiles(cfg, outdir / "fig_mode_profiles.png")
    if args.poynting:
        fig_poynting(cfg, outdir / f"fig_nominal_poynting_{branch}.png", args.res, args.until)
    write_provenance(cfg, outdir, branch, args)


if __name__ == "__main__":
    main()
