"""V1: validate a receiving-guide port and same-aperture power accounting.

Part A checks the method on an effectively isolated straight guide (native
signed flux vs modal forward power on the identical aperture).  Part B applies
it to the nominal coupler output for the Hz branch.
"""

from __future__ import annotations

import argparse

import meep as mp
import numpy as np

from src.config import CouplerConfig
from src.geometry import build_geometry, make_plan
from src.sources import build_source
from src.ports import (
    DF_FREQ_WIDTH,
    DF_NFREQ,
    add_oblique_mode_monitor,
    add_same_aperture_flux,
    oblique_modal_powers,
)
from src.units import freq_from_wavelength_nm


def part_a(res: float = 100.0) -> None:
    t, f = 0.262, 1.0 / 1.55
    cell_y, pml = 6.0, 1.0
    geo = [
        mp.Block(size=mp.Vector3(mp.inf, 2 * cell_y, 0), center=mp.Vector3(0, cell_y, 0),
                 material=mp.Medium(epsilon=1.0)),
        mp.Block(size=mp.Vector3(mp.inf, t, 0), center=mp.Vector3(0, 0, 0),
                 material=mp.Medium(epsilon=3.48**2)),
    ]
    src = mp.EigenModeSource(src=mp.GaussianSource(f, fwidth=0.05), center=mp.Vector3(-1, 0),
                             size=mp.Vector3(0, cell_y - 2 * pml), direction=mp.X,
                             eig_band=1, eig_parity=mp.ODD_Z, eig_match_freq=True)
    sim = mp.Simulation(cell_size=mp.Vector3(4.0, cell_y, 0), resolution=res, geometry=geo,
                        default_material=mp.Medium(epsilon=1.444**2),
                        boundary_layers=[mp.PML(pml)], dimensions=2, sources=[src])
    mon = sim.add_mode_monitor(f, DF_FREQ_WIDTH, DF_NFREQ,
                               mp.ModeRegion(center=mp.Vector3(1.0, 0), size=mp.Vector3(0, cell_y - 2 * pml)))
    fmon = sim.add_flux(f, DF_FREQ_WIDTH, DF_NFREQ,
                        mp.FluxRegion(center=mp.Vector3(1.0, 0), size=mp.Vector3(0, cell_y - 2 * pml),
                                      direction=mp.X))
    sim.run(until_after_sources=40)
    i = DF_NFREQ // 2
    flux = mp.get_fluxes(fmon)[i]
    res_modal = sim.get_eigenmode_coefficients(mon, [1], eig_parity=mp.ODD_Z)
    fwd = abs(res_modal.alpha[0, i, 0]) ** 2
    bwd = abs(res_modal.alpha[0, i, 1]) ** 2
    print(f"[A isolated guide] native_flux={flux:.4f} modal_fwd={fwd:.4f} modal_bwd={bwd:.4f} "
          f"ratio_flux/fwd={flux/fwd:.4f} backward_frac={bwd/fwd:.2e}")


def part_b(cfg: CouplerConfig, res: float = 25.0, until: float = 60.0, half_span: float = 6.0) -> None:
    cfg = CouplerConfig.from_dict(cfg.to_dict())
    cfg.numerics.resolution = res
    cfg.numerics.until_after_sources = until
    plan = make_plan(cfg)
    geom, plan, idx = build_geometry(cfg)
    sim = mp.Simulation(cell_size=mp.Vector3(plan.domain["sx"], plan.domain["sy"], 0),
                        resolution=cfg.numerics.resolution, geometry=geom,
                        sources=build_source(cfg, plan),
                        boundary_layers=[mp.PML(cfg.numerics.pml_um)],
                        default_material=idx.medium("clad"), dimensions=2)
    g = cfg.geometry
    s_mon = g.stack_start_um + g.stack_length_um + min(cfg.monitor.output_offset_um, g.guide_ext_down_um)
    mmon, center, size = add_oblique_mode_monitor(sim, cfg, plan, s_mon, half_span_um=half_span)
    fmon = add_same_aperture_flux(sim, cfg, center, size)
    sim.run(until_after_sources=cfg.numerics.until_after_sources)
    i = DF_NFREQ // 2
    flux = mp.get_fluxes(fmon)[i]
    modal = oblique_modal_powers(sim, mmon, cfg, plan)
    fwd = sum(d["forward"][i] for d in modal.values())
    bwd = sum(d["backward"][i] for d in modal.values())
    print(f"[B coupler {cfg.source.branch}] same-aperture native_flux={flux:.2f} "
          f"sum_fwd={fwd:.2f} sum_bwd={bwd:.2f} modal_net={fwd-bwd:.2f} "
          f"residual_frac={(flux-(fwd-bwd))/max(abs(flux),1e-12):.3f}")
    for b, d in modal.items():
        print(f"   band {b} neff={d['neff'][i]:.4f} fwd={d['forward'][i]:.2f} bwd={d['backward'][i]:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--branch", default="hz")
    ap.add_argument("--res", type=float, default=25.0)
    ap.add_argument("--until", type=float, default=60.0)
    args = ap.parse_args()
    part_a()
    cfg = CouplerConfig.load(args.config)
    cfg.source.branch = args.branch
    part_b(cfg, args.res, args.until)


if __name__ == "__main__":
    main()
