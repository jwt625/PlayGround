"""Angle sweep of the fundamental mode pair of an x-cut thin-film LiNbO3 rib, and export for the course widget.

    uv run --no-project --with numpy --with scipy python sweep.py --profile thick --out sweep_thick.npz
    uv run --no-project --with numpy --with scipy python export.py ...      (see export.py)

theta is the in-plane angle between the propagation direction and the crystal y axis (0 = along y, TE polarized
along the c axis; 90 = along z, TE polarized along y). Only 0..90 deg is solved: theta -> 180 - theta (equivalently
-theta) flips the sign of eps_xz, which is the mirror image x -> -x of a left-right symmetric rib, so the indices are
identical and the fields are mirror images (checked in validate.py, V5).

Stage 1 solves a uniform angle list; stage 2 refines around the smallest splitting of the first two modes.
Resumable: results are cached per angle in the --cache folder (one .npz per profile and angle) and skipped if present.
Runs one solve at a time by default (--workers 1, two threads); more workers only on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import fdmode as fm

LAM = 1.55

# Representative profiles. All numbers are modelling choices (see README.md), not a specific published device.
PROFILES = {
    # 800 nm x-cut film, 400 nm etch (half the film), 1.2 um top width, 65 deg sidewall, air above, oxide below
    "thick": dict(t_film=0.8, t_slab=0.4, w_top=1.2, sidewall=65.0, cover="air"),
    # 600 nm x-cut film, 300 nm etch, 1.2 um top width, 65 deg sidewall, air above, oxide below
    "thin": dict(t_film=0.6, t_slab=0.3, w_top=1.2, sidewall=65.0, cover="air"),
}
H = 0.02          # grid step, um
XW = 3.0          # half width of the window, um
BOX = 1.4         # oxide below the film inside the window, um
TOP = 1.0         # cover above the film top inside the window, um
NMODES = 2
# field map window and resolution exported for the widget
MAP_X = (-1.6, 1.6)
MAP_NX = 56
MAP_DY = (3.2 / MAP_NX)


def map_window(p: dict) -> tuple[tuple[float, float], int]:
    y0 = -0.55
    y1 = p["t_film"] + 0.45
    ny = int(round((y1 - y0) / MAP_DY))
    return (y0, y0 + ny * MAP_DY), ny


def solve_angle(args):
    name, theta, cache = args
    f = Path(cache) / f"{name}_{theta:08.4f}.npz"
    if f.exists():
        return str(f)
    p = PROFILES[name]
    no, ne = fm.n_ln_zelmon(LAM)
    ns = fm.n_sio2_malitson(LAM)
    cov = 1.0 if p["cover"] == "air" else ns
    g = fm.make_grid(-XW, XW, -BOX, p["t_film"] + TOP, H)
    st = fm.rib_structure(p["t_film"], p["t_slab"], p["w_top"], p["sidewall"], fm.ln_tensor_xcut(no, ne, theta), ns, cov)
    ms = fm.solve(fm.build(g, st, LAM), 2.15, k=6, n_min=ns)[:4]
    # sample fields on the map grid (nearest node; map pixel 57 nm vs grid 20 nm)
    (yw, ny) = map_window(p)
    xs = MAP_X[0] + (np.arange(MAP_NX) + 0.5) * (MAP_X[1] - MAP_X[0]) / MAP_NX
    ys = yw[0] + (np.arange(ny) + 0.5) * MAP_DY
    ix = np.clip(np.round((xs - g.x0) / g.h).astype(int), 0, g.nx - 1)
    iy = np.clip(np.round((ys - g.y0) / g.h).astype(int), 0, g.ny - 1)
    xn, yn = g.xn, g.yn
    core = (np.abs(xn)[:, None] < 1.5) & (yn[None, :] > -0.3) & (yn[None, :] < p["t_film"] + 0.3)
    out = dict(theta=theta, neff=np.array([m.neff for m in ms]), te=np.array([m.te_frac for m in ms]))
    loc = []
    for k, m in enumerate(ms[:NMODES]):
        e2 = np.abs(m.Ex) ** 2 + np.abs(m.Ey) ** 2 + np.abs(m.Ez) ** 2
        loc.append(float(e2[core].sum() / e2.sum()))
        out[f"ex{k}"] = np.abs(m.Ex[np.ix_(ix, iy)])
        out[f"ey{k}"] = np.abs(m.Ey[np.ix_(ix, iy)])
        out[f"ez{k}"] = np.abs(m.Ez[np.ix_(ix, iy)])
    out["loc"] = np.array(loc)
    np.savez(f, **out)
    return str(f)


def run(name: str, thetas, cache: str, workers: int):
    jobs = [(name, float(t), cache) for t in thetas]
    if workers <= 1:
        for j in jobs:
            solve_angle(j)
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(solve_angle, jobs):
            pass


def load(name: str, cache: str):
    rows = []
    for f in sorted(Path(cache).glob(f"{name}_*.npz")):
        d = dict(np.load(f))
        rows.append(d)
    rows.sort(key=lambda d: float(d["theta"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="thick", choices=list(PROFILES))
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--fine", type=float, default=0.1, help="refined step around the minimum splitting")
    ap.add_argument("--fine-span", type=float, default=2.0, help="half span of the refined region, deg")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    Path(a.cache).mkdir(exist_ok=True)
    coarse = np.round(np.arange(0.0, 90.0 + 1e-9, a.step), 4)
    if a.dry_run:
        print(f"{a.profile}: {len(coarse)} coarse angles, cache {a.cache}")
        return
    t0 = time.time()
    run(a.profile, coarse, a.cache, a.workers)
    rows = load(a.profile, a.cache)
    th = np.array([float(r["theta"]) for r in rows])
    gap = np.array([r["neff"][0] - r["neff"][1] for r in rows])
    i = int(np.argmin(gap))
    tc = th[i]
    print(f"{a.profile}: coarse minimum splitting {gap[i]:.5f} at {tc} deg ({time.time()-t0:.0f} s)", flush=True)
    if a.fine > 0 and 0 < tc < 90:
        fine = np.round(np.arange(tc - a.fine_span, tc + a.fine_span + 1e-9, a.fine), 4)
        fine = fine[(fine > 0) & (fine < 90)]
        run(a.profile, fine, a.cache, a.workers)
        rows = load(a.profile, a.cache)
        th = np.array([float(r["theta"]) for r in rows])
        gap = np.array([r["neff"][0] - r["neff"][1] for r in rows])
        i = int(np.argmin(gap))
        print(f"{a.profile}: refined minimum splitting {gap[i]:.6f} at {th[i]} deg", flush=True)
    summ = [dict(theta=float(r["theta"]), neff=[float(v) for v in r["neff"]], te=[float(v) for v in r["te"]],
                 loc=[float(v) for v in r["loc"]]) for r in rows]
    Path(a.cache, f"{a.profile}_summary.json").write_text(json.dumps(summ, indent=0))
    print(f"done {len(rows)} angles in {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
