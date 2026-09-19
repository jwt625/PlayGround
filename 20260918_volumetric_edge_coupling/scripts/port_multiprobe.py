"""D2/D3: corrected coarse Hz multiprobe along the isolated-output-port-v1 path.

Single frequency (1550 nm), simultaneous mode+flux monitors at several
sidewall coordinates, plus a radiated-power proxy.  Diagnostic, not production.
"""

from __future__ import annotations

import argparse

import meep as mp
import numpy as np

from src.config import CouplerConfig
from src.geometry import build_geometry, make_plan
from src.sources import build_source
from src.ports import (
    add_oblique_mode_monitor,
    add_same_aperture_flux,
    oblique_modal_powers,
)
from src.units import freq_from_wavelength_nm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/isolated_port.yaml")
    ap.add_argument("--branch", default="hz")
    ap.add_argument("--res", type=float, default=25.0)
    ap.add_argument("--until", type=float, default=60.0)
    ap.add_argument("--probes", type=float, nargs="+", default=[31, 34, 38, 41, 44, 50])
    ap.add_argument("--half-span", type=float, default=4.0)
    args = ap.parse_args()

    cfg = CouplerConfig.load(args.config)
    cfg.source.branch = args.branch
    cfg.numerics.resolution = args.res
    cfg.numerics.until_after_sources = args.until
    plan = make_plan(cfg)
    geom, plan, idx = build_geometry(cfg)
    sim = mp.Simulation(
        cell_size=mp.Vector3(plan.domain["sx"], plan.domain["sy"], 0),
        resolution=cfg.numerics.resolution, geometry=geom, sources=build_source(cfg, plan),
        boundary_layers=[mp.PML(cfg.numerics.pml_um)], default_material=idx.medium("clad"), dimensions=2,
    )
    mons = []
    for s in args.probes:
        m, c, sz = add_oblique_mode_monitor(sim, cfg, plan, s, half_span_um=args.half_span, single_freq=True)
        f = add_same_aperture_flux(sim, cfg, c, sz)
        mons.append((s, m, f))
    sim.run(until_after_sources=cfg.numerics.until_after_sources)

    print(f"[{args.branch}] probes half-span={args.half_span} um, single freq 1550 nm")
    for s, m, f in mons:
        modal = oblique_modal_powers(sim, m, cfg, plan, bands=[1, 2])
        nat = mp.get_fluxes(f)[0]
        b1 = modal[1]
        b2 = modal[2]
        print(f"s={s:5.1f} gap={cfg.geometry.gap_at(s)*1000:6.0f}nm native={nat:8.3f} "
              f"b1(neff={b1['neff'][0]:.3f} fwd={b1['forward'][0]:7.3f} bwd={b1['backward'][0]:6.3f}) "
              f"b2(neff={b2['neff'][0]:.3f} fwd={b2['forward'][0]:7.3f} bwd={b2['backward'][0]:6.3f})")


if __name__ == "__main__":
    main()
