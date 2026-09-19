"""Inspect the nominal coupler: run and save field maps with overlays.

Usage: PYTHONPATH=. python scripts/inspect.py [--res 25] [--until 60]
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
from src.ports import extraction_line
from src.sources import build_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/nominal.yaml")
    ap.add_argument("--res", type=float, default=25.0)
    ap.add_argument("--until", type=float, default=80.0)
    ap.add_argument("--out", default="reports/nominal_fields.png")
    args = ap.parse_args()

    cfg = CouplerConfig.load(args.config)
    cfg.numerics.resolution = args.res
    cfg.numerics.until_after_sources = args.until
    plan = make_plan(cfg)
    geom, plan, idx = build_geometry(cfg)

    sim = mp.Simulation(
        cell_size=mp.Vector3(plan.domain["sx"], plan.domain["sy"], 0),
        resolution=cfg.numerics.resolution,
        geometry=geom,
        sources=build_source(cfg, plan, cw=True),
        boundary_layers=[mp.PML(cfg.numerics.pml_um)],
        default_material=idx.medium("clad"),
        dimensions=2,
    )
    comp = mp.Ez if cfg.source.branch == "ez" else mp.Hz
    sim.run(until=120.0)
    arr = sim.get_array(center=mp.Vector3(), size=mp.Vector3(plan.domain["sx"], plan.domain["sy"]), component=comp)

    sx, sy = plan.domain["sx"], plan.domain["sy"]
    cx, cy = plan.domain["cell_center"]
    # array is [ix, iy] per this Meep build
    # Meep grid is centred on the origin; paper = meep + cell_center
    xs = np.linspace(-sx / 2, sx / 2, arr.shape[0])
    ys = np.linspace(-sy / 2, sy / 2, arr.shape[1])
    pxs = xs + cx
    pys = ys + cy

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, sl, title in [
        (axes[0], (slice(None), slice(None)), "full"),
    ]:
        im = ax.pcolormesh(pxs, pys, np.abs(arr).T, shading="auto", cmap="inferno")
        ax.plot(plan.oxide_poly[:, 0], plan.oxide_poly[:, 1], "c-", lw=1)
        ax.plot(plan.guide_poly[:, 0], plan.guide_poly[:, 1], "w-", lw=1)
        ss = np.linspace(-40, 40, 2)
        ax.plot(plan.r_top[0] + ss * plan.u[0], plan.r_top[1] + ss * plan.u[1], "r--", lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"|{ 'Ez' if comp==mp.Ez else 'Hz' }| {title}")
        fig.colorbar(im, ax=ax, fraction=0.04)
    # zoom near coupling
    ax = axes[1]
    im = ax.pcolormesh(pxs, pys, np.abs(arr).T, shading="auto", cmap="inferno")
    ax.plot(plan.oxide_poly[:, 0], plan.oxide_poly[:, 1], "c-", lw=1.5)
    ax.plot(plan.guide_poly[:, 0], plan.guide_poly[:, 1], "w-", lw=1.5)
    ax.plot(plan.r_top[0] + ss * plan.u[0], plan.r_top[1] + ss * plan.u[1], "r--", lw=1)
    ip = plan.points["impact_xy"]
    ax.set_xlim(ip[0] - 10, ip[0] + 16)
    ax.set_ylim(ip[1] - 14, ip[1] + 8)
    ax.set_aspect("equal")
    ax.set_title("coupling zoom")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("saved", args.out)


if __name__ == "__main__":
    main()
