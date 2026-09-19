"""R5 screen: loaded receiving-guide index versus local oxide gap.

Parallel-layer approximation (declared): air / Si(262 nm) / oxide(g) /
semi-infinite Si substrate, solved by MPB through an axis-aligned mode
monitor.  The guide-associated band is identified by continuity from the
isolated limit (large gap), not by whichever real index lies nearest 2.07.
"""

from __future__ import annotations

import argparse

import meep as mp
import numpy as np

ISOLATED = {"ez": 2.9598, "hz": 2.2858}


def solve(core_index, t, g, n_sub, wavelength_um, branch, n_bands=6, resolution=60.0):
    fcen = 1.0 / wavelength_um
    cell_y = 10.0
    pml = 1.5
    parity = mp.ODD_Z if branch == "ez" else mp.EVEN_Z
    # air background; explicit substrate below the oxide
    geo = [
        mp.Block(size=mp.Vector3(mp.inf, 2 * cell_y, 0),
                 center=mp.Vector3(0, -t / 2 - g - cell_y, 0),
                 material=mp.Medium(epsilon=n_sub**2)),
        mp.Block(size=mp.Vector3(mp.inf, g, 0),
                 center=mp.Vector3(0, -t / 2 - g / 2, 0),
                 material=mp.Medium(epsilon=1.444**2)),
        mp.Block(size=mp.Vector3(mp.inf, t, 0),
                 center=mp.Vector3(0, 0, 0),
                 material=mp.Medium(epsilon=core_index**2)),
    ]
    sim = mp.Simulation(
        cell_size=mp.Vector3(4.0, cell_y, 0), resolution=resolution, geometry=geo,
        default_material=mp.Medium(epsilon=1.0),
        boundary_layers=[mp.PML(pml)], dimensions=2,
    )
    src = mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=0.05), center=mp.Vector3(-0.5, 0),
        size=mp.Vector3(0, cell_y - 2 * pml), direction=mp.X, eig_band=1,
        eig_parity=parity, eig_match_freq=True,
    )
    sim.sources = [src]
    mon = sim.add_mode_monitor(
        fcen, 0, 1,
        mp.ModeRegion(center=mp.Vector3(0.5, 0), size=mp.Vector3(0, cell_y - 2 * pml)),
    )
    sim.run(until_after_sources=10)
    bands = list(range(1, n_bands + 1))
    res = sim.get_eigenmode_coefficients(mon, bands, eig_parity=parity)
    vals = []
    for bi, band in enumerate(bands):
        kp = res.kpoints[bi]
        vals.append((float(np.linalg.norm([kp.x, kp.y, kp.z])) / fcen, abs(res.alpha[bi, 0, 0]) ** 2))
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=60.0)
    ap.add_argument("--gaps-nm", type=float, nargs="+", default=[64, 150, 244, 500, 5000])
    ap.add_argument("--branch", nargs="+", default=["hz", "ez"])
    args = ap.parse_args()
    for branch in args.branch:
        print(f"--- branch {branch} (isolated {ISOLATED[branch]}) ---")
        for g_nm in args.gaps_nm:
            vals = solve(3.48, 0.262, g_nm / 1000.0, 3.48, 1.55, branch, resolution=args.res)
            # guide branch = index closest to isolated among bands with index > 1.45
            cand = [(n, p) for n, p in vals if n > 1.45]
            guide = min(cand, key=lambda v: abs(v[0] - ISOLATED[branch])) if cand else None
            top = max(cand, key=lambda v: v[1]) if cand else None
            print(f"g={g_nm:6.0f}nm  nearest-to-isolated={guide}  strongest={top}")


if __name__ == "__main__":
    main()
