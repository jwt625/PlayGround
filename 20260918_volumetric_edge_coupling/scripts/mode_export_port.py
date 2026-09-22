"""R6-P1 D1: compact mode-field export at the isolated port (no full coupler run).

Builds a short straight guide at the declared orientation and mesh for two
stacks -- the isolated air/Si/oxide reference and the port air/Si/oxide(3um)/Si
-- identifies the guide band, launches it, exports the complex field slice and
compares indices and profiles.  Hz primary; Ez optional.

Caveat: the guide band is identified by index proximity here as a first pass;
field-localization identity is the next step.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from src.units import freq_from_wavelength_nm


def build(port: bool, res: float, branch: str):
    t = 0.262
    theta = np.radians(53.50)
    fcen = freq_from_wavelength_nm(1550.0)
    d = mp.Vector3(np.cos(theta), -np.sin(theta), 0)
    g = mp.Vector3(np.sin(theta), np.cos(theta), 0)
    # guide as a rotated block so orientation/subcell match the coupler
    guide = mp.Block(
        center=mp.Vector3(), size=mp.Vector3(mp.inf, t, 0),
        e1=mp.Vector3(x=1).rotate(mp.Vector3(z=1), -theta),
        e2=mp.Vector3(y=1).rotate(mp.Vector3(z=1), -theta),
        material=mp.Medium(epsilon=3.48**2),
    )
    ex = mp.Vector3(x=1).rotate(mp.Vector3(z=1), -theta)  # along guide
    ey = mp.Vector3(y=1).rotate(mp.Vector3(z=1), -theta)  # guide normal
    if port:
        D = 3.0
        geo = [
            mp.Block(center=(-(t / 2 + D + 4.0)) * ey, size=mp.Vector3(mp.inf, 8.0, 0),
                     e1=ex, e2=ey, material=mp.Medium(epsilon=3.48**2)),
            mp.Block(center=(-(t / 2 + D / 2)) * ey, size=mp.Vector3(mp.inf, D, 0),
                     e1=ex, e2=ey, material=mp.Medium(epsilon=1.444**2)),
            guide,
        ]
        default = mp.Medium(epsilon=1.0)
    else:
        geo = [
            mp.Block(center=((t / 2 + 2.0)) * ey, size=mp.Vector3(mp.inf, 4.0, 0),
                     e1=ex, e2=ey, material=mp.Medium(epsilon=1.0)),
            guide,
        ]
        default = mp.Medium(epsilon=1.444**2)
    return geo, default, fcen, d, g, theta, t


def solve(port: bool, res: float, branch: str):
    geo, default, fcen, d, g, theta, t = build(port, res, branch)
    parity = mp.ODD_Z if branch == "ez" else mp.EVEN_Z
    src = mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=0.05), center=mp.Vector3(-0.5, 0),
        size=mp.Vector3(0, 5.0, 0), direction=mp.NO_DIRECTION,
        eig_kpoint=mp.Vector3(d.x, d.y, 0), eig_band=1, eig_parity=parity, eig_match_freq=True,
    )
    sim = mp.Simulation(cell_size=mp.Vector3(4.0, 10.0, 0), resolution=res, geometry=geo,
                        default_material=default, boundary_layers=[mp.PML(1.0)], dimensions=2, sources=[src])
    mon = sim.add_mode_monitor(fcen, 0, 1,
                               mp.ModeRegion(center=mp.Vector3(0.7, 0), size=mp.Vector3(0, 5.0)))
    sim.run(until_after_sources=20)
    kdir = mp.Vector3(d.x, d.y, 0)
    res_bands = sim.get_eigenmode_coefficients(
        mon, list(range(1, 7)), eig_parity=parity,
        direction=mp.NO_DIRECTION, kpoint_func=lambda f, n: kdir,
    )
    neff = []
    for bi in range(6):
        kp = res_bands.kpoints[bi]
        neff.append(float(np.linalg.norm([kp.x, kp.y, kp.z])) / fcen)
    return sim, neff, fcen, g, theta, t


def profile_from_launch(port, band, res, branch):
    geo, default, fcen, d, g, theta, t = build(port, res, branch)
    parity = mp.ODD_Z if branch == "ez" else mp.EVEN_Z
    src = mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=0.05), center=mp.Vector3(-0.5, 0),
        size=mp.Vector3(0, 5.0, 0), direction=mp.NO_DIRECTION,
        eig_kpoint=mp.Vector3(d.x, d.y, 0), eig_band=band, eig_parity=parity, eig_match_freq=True,
    )
    sim = mp.Simulation(cell_size=mp.Vector3(4.0, 10.0, 0), resolution=res, geometry=geo,
                        default_material=default, boundary_layers=[mp.PML(1.0)], dimensions=2, sources=[src])
    comp = mp.Hz if branch == "hz" else mp.Ez
    line = sim.add_dft_fields([comp], fcen, 0, 1, center=mp.Vector3(0.7, 0), size=mp.Vector3(0, 5.0))
    sim.run(until_after_sources=20)
    arr = np.asarray(sim.get_dft_array(line, comp, 0))
    x, y, z, w = sim.get_array_metadata(center=mp.Vector3(0.7, 0), size=mp.Vector3(0, 5.0))
    return np.asarray(y), arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="hz")
    ap.add_argument("--res", type=float, default=60.0)
    ap.add_argument("--out", default="reports/figs/fig_port_mode_export.png")
    args = ap.parse_args()

    _, neff_iso, *_ = solve(False, args.res, args.branch)
    _, neff_port, *_ = solve(True, args.res, args.branch)
    print(f"[{args.branch}] isolated bands n_eff:", [f"{v:.4f}" for v in neff_iso])
    print(f"[{args.branch}] port     bands n_eff:", [f"{v:.4f}" for v in neff_port])
    ref = neff_iso[0]
    guide_band = min(range(len(neff_port)), key=lambda i: abs(neff_port[i] - ref)) + 1
    print(f"isolated guide band=1 n_eff={ref:.4f}; port guide band={guide_band} "
          f"n_eff={neff_port[guide_band-1]:.4f} delta={neff_port[guide_band-1]-ref:+.4f}")

    yi, pi = profile_from_launch(False, 1, args.res, args.branch)
    yp, pp = profile_from_launch(True, guide_band, args.res, args.branch)
    pi = pi / np.max(np.abs(pi))
    pp = pp / np.max(np.abs(pp))
    num = np.sum(np.conj(pi) * pp)
    den = np.sqrt(np.sum(np.abs(pi) ** 2) * np.sum(np.abs(pp) ** 2))
    print(f"profile normalized overlap = {abs(num/den):.4f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(np.abs(pi), yi, "b-", label=f"isolated (n_eff={ref:.4f})")
    ax.plot(np.abs(pp), yp, "r--", label=f"port (n_eff={neff_port[guide_band-1]:.4f})")
    ax.set_xlabel(f"|{ 'Hz' if args.branch=='hz' else 'Ez' }| (norm.)")
    ax.set_ylabel("vertical coordinate (um)")
    ax.set_title(f"isolated-port mode export, {args.branch}, res {args.res:.0f}/um, "
                 f"overlap {abs(num/den):.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print("saved", args.out)


if __name__ == "__main__":
    main()
