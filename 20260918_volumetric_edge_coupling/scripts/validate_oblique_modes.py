"""R3: validate oblique-guide modal decomposition against a net flux monitor.

First reproduces the official Meep v1.34.0 ``oblique-source.py`` example as a
known reference, then applies the same route to the receiving guide geometry.
"""

from __future__ import annotations

import argparse

import meep as mp
import numpy as np


def official_example(resolution: float = 50.0) -> None:
    cell_size = mp.Vector3(14, 14)
    pml_layers = [mp.PML(thickness=2)]
    rot_angle = np.radians(20)
    w = 1.0
    geometry = [
        mp.Block(
            center=mp.Vector3(),
            size=mp.Vector3(mp.inf, w, mp.inf),
            e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), rot_angle),
            e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), rot_angle),
            material=mp.Medium(epsilon=12),
        )
    ]
    fsrc = 0.15
    bnum = 1
    kpoint = mp.Vector3(x=1).rotate(mp.Vector3(z=1), rot_angle)
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(fsrc, fwidth=0.2 * fsrc),
            center=mp.Vector3(),
            size=mp.Vector3(y=3 * w),
            direction=mp.NO_DIRECTION,
            eig_kpoint=kpoint,
            eig_band=bnum,
            eig_parity=mp.ODD_Z,
            eig_match_freq=True,
        )
    ]
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        boundary_layers=pml_layers,
        sources=sources,
        geometry=geometry,
    )
    tran = sim.add_flux(
        fsrc, 0, 1, mp.FluxRegion(center=mp.Vector3(x=5), size=mp.Vector3(y=14))
    )
    sim.run(until_after_sources=50)
    res = sim.get_eigenmode_coefficients(
        tran, [1], eig_parity=mp.ODD_Z, direction=mp.NO_DIRECTION,
        kpoint_func=lambda f, n: kpoint,
    )
    flux = mp.get_fluxes(tran)[0]
    modal = abs(res.alpha[0, 0, 0]) ** 2
    print(f"[official example] flux={flux:.6f} modal|alpha|^2={modal:.6f} "
          f"ratio={modal/flux:.4f} kpoints={res.kpoints}")
    return flux, modal


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=50.0)
    args = ap.parse_args()
    official_example(args.res)


if __name__ == "__main__":
    main()
