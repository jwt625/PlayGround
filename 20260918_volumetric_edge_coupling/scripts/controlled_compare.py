"""C0/C1: controlled nominal-vs-port comparison in one common cell/grid.

Both variants use the identical cell, resolution, source, frequency and time
window.  Saves initialized dielectric and complex fields over the interaction,
signed flux and modal coefficients at broad/narrow probes.  Hz primary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import meep as mp
import numpy as np

from src.config import CouplerConfig
from src.geometry import build_geometry, make_plan
from src.sources import build_source
from src.ports import add_oblique_mode_monitor, oblique_modal_powers, run_incident_reference
from src.units import freq_from_wavelength_nm

PROBES = [(28.0, 6.0), (31.0, 2.4), (50.0, 2.52)]


def run_variant(cfg: CouplerConfig, tag: str, outdir: Path, res: float, until: float):
    cfg = CouplerConfig.from_dict(cfg.to_dict())
    cfg.numerics.resolution = res
    cfg.numerics.until_after_sources = until
    plan = make_plan(cfg)
    geom, plan, idx = build_geometry(cfg)
    fcen = freq_from_wavelength_nm(cfg.source.wavelength_nm)
    sim = mp.Simulation(cell_size=mp.Vector3(plan.domain["sx"], plan.domain["sy"], 0),
                        resolution=cfg.numerics.resolution, geometry=geom,
                        sources=build_source(cfg, plan),
                        boundary_layers=[mp.PML(cfg.numerics.pml_um)],
                        default_material=idx.medium("clad"), dimensions=2)
    # fields over the interaction + upstream region
    region = mp.Vector3(45.0, 40.0, 0)
    rc = plan.meep_xy(7.0, 8.0)
    dmon = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz], fcen, 0, 1,
                              center=rc, size=region)
    probes = []
    for s, hs in PROBES:
        if s > cfg.geometry.port_start_s_um + cfg.geometry.port_transition_um and not cfg.geometry.isolated_port:
            continue
        m, c, sz = add_oblique_mode_monitor(sim, cfg, plan, s, half_span_um=hs, single_freq=True)
        f = sim.add_flux(fcen, 0, 1, mp.FluxRegion(center=c, size=sz, direction=mp.X))
        probes.append((s, hs, m, f))
    trace = {"t": [], "v": []}
    trace_pt = plan.meep_xy(plan.points["impact_xy"][0], plan.points["impact_xy"][1])

    def record(s):
        trace["t"].append(s.meep_time())
        trace["v"].append(abs(s.get_field_point(mp.Hz, trace_pt)))

    sim.run(mp.at_every(2.0, record), until_after_sources=cfg.numerics.until_after_sources)

    eps = np.asarray(sim.get_array(center=mp.Vector3(), size=mp.Vector3(plan.domain["sx"], plan.domain["sy"]),
                                   component=mp.Dielectric))
    fields = {c: np.asarray(sim.get_dft_array(dmon, getattr(mp, c), 0)) for c in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]}
    x, y, z, w = sim.get_array_metadata(center=rc, size=region)
    probe_out = {}
    for s, hs, m, f in probes:
        modal = oblique_modal_powers(sim, m, cfg, plan, bands=list(range(1, 7)))
        probe_out[s] = {
            "native": mp.get_fluxes(f)[0],
            "neff": [modal[b]["neff"][0] for b in range(1, 7)],
            "fwd": [modal[b]["forward"][0] for b in range(1, 7)],
            "bwd": [modal[b]["backward"][0] for b in range(1, 7)],
        }
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(outdir / f"{tag}.npz", eps=eps, x=np.asarray(x), y=np.asarray(y),
             trace_t=np.array(trace["t"]), trace_v=np.array(trace["v"]), **{f"dft_{k}": v for k, v in fields.items()})
    return plan, probe_out, eps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=25.0)
    ap.add_argument("--until", type=float, default=60.0)
    ap.add_argument("--branch", default="hz")
    ap.add_argument("--outdir", default="results/controlled_compare")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    # common cell = the port config's auto domain
    port = CouplerConfig.load("configs/isolated_port.yaml")
    nom = CouplerConfig.load("configs/nominal.yaml")
    pplan = make_plan(port)
    for c in (port, nom):
        c.source.branch = args.branch
        c.numerics.cell_sx_um = float(pplan.domain["sx"])
        c.numerics.cell_sy_um = float(pplan.domain["sy"])
        c.numerics.cell_center_x_um = float(pplan.domain["cell_center"][0])
        c.numerics.cell_center_y_um = float(pplan.domain["cell_center"][1])

    _, pout, peps = run_variant(port, "port", outdir, args.res, args.until)
    _, nout, neps = run_variant(nom, "nominal", outdir, args.res, args.until)

    print(f"cell={pplan.domain['sx']:.2f} x {pplan.domain['sy']:.2f} um, "
          f"center={np.round(pplan.domain['cell_center'],3)}, res={args.res}, until={args.until}")
    if neps.shape == peps.shape:
        d = np.abs(neps - peps)
        print(f"dielectric max|delta|={d.max():.4g} count(|d|>1e-6)={int((d>1e-6).sum())}")
    print("probe comparison (native signed flux, and modal bands with n_eff>1.45):")
    for s in sorted(set(nout) | set(pout)):
        n = nout.get(s); p = pout.get(s)
        print(f"  s={s}: native nominal={n['native'] if n else None} port={p['native'] if p else None}")
        if n and p:
            for i, b in enumerate(range(1, 7)):
                if n["neff"][i] > 1.45 or p["neff"][i] > 1.45:
                    print(f"      b{b}: nom(neff={n['neff'][i]:.3f} fwd={n['fwd'][i]:.3f}) "
                          f"port(neff={p['neff'][i]:.3f} fwd={p['fwd'][i]:.3f})")
    print("saved raw arrays in", outdir)


if __name__ == "__main__":
    main()
