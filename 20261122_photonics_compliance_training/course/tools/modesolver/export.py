"""Write course/widgets/xcut_modes_data.js from the sweep cache (see sweep.py).

    uv run --no-project --with numpy --with scipy python export.py --cache <cache> --out ../../widgets/xcut_modes_data.js

Main profile: neff, TE fraction and quantized field maps (|Ex|, |Ey|, uint8, base64) of the first two modes at every
solved angle. Comparison profile (optional): neff and TE fraction only.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

import fdmode as fm
from sweep import BOX, H, LAM, MAP_DY, MAP_NX, MAP_X, NMODES, PROFILES, TOP, XW, load, map_window


def profile_block(name: str, rows, with_fields: bool) -> dict:
    p = PROFILES[name]
    th = [round(float(r["theta"]), 4) for r in rows]
    modes = []
    for k in range(NMODES):
        modes.append({"neff": [round(float(r["neff"][k]), 6) for r in rows],
                      "teFrac": [round(float(r["te"][k]), 4) for r in rows]})
    gap = np.array([r["neff"][0] - r["neff"][1] for r in rows])
    i = int(np.argmin(gap))
    blk = {
        "geometry": {
            "film_um": p["t_film"], "etch_um": round(p["t_film"] - p["t_slab"], 4), "slab_um": p["t_slab"],
            "top_width_um": p["w_top"], "sidewall_deg_from_substrate": p["sidewall"],
            "bottom_width_um": round(p["w_top"] + 2 * (p["t_film"] - p["t_slab"]) / np.tan(np.deg2rad(p["sidewall"])), 4),
            "below": "silicon dioxide (buried oxide)", "above": "air" if p["cover"] == "air" else "silicon dioxide",
        },
        "outline": [[round(a, 4), round(b, 4)] for a, b in fm.rib_outline(p["t_film"], p["t_slab"], p["w_top"],
                                                                            p["sidewall"], MAP_X[0], MAP_X[1])],
        "theta": th,
        "modes": modes,
        "minSplitting": {"theta": th[i], "dneff": round(float(gap[i]), 6),
                         "teFrac": [round(float(rows[i]["te"][k]), 4) for k in range(NMODES)]},
        "localization": [round(float(min(r["loc"][k] for r in rows)), 4) for k in range(NMODES)],
    }
    if with_fields:
        (yw, ny) = map_window(p)
        maps = {"ex": [[] for _ in range(NMODES)], "ey": [[] for _ in range(NMODES)]}
        for r in rows:
            for k in range(NMODES):
                ex, ey = r[f"ex{k}"], r[f"ey{k}"]
                s = max(float(ex.max()), float(ey.max()), 1e-30)
                for nm, a in (("ex", ex), ("ey", ey)):
                    q = np.clip(np.round(a / s * 255), 0, 255).astype(np.uint8)
                    # row-major with y rows from top (largest y) to bottom, x columns left to right
                    img = q.T[::-1, :]
                    maps[nm][k].append(base64.b64encode(img.tobytes()).decode("ascii"))
        blk["fields"] = {"nx": MAP_NX, "ny": ny, "x": list(MAP_X), "y": [round(yw[0], 4), round(yw[1], 4)],
                         "order": "rows from top to bottom, columns from left to right; one byte per pixel",
                         "scale": "per mode and angle, |Ex| and |Ey| share one scale (255 = larger of the two maxima)",
                         "ex": maps["ex"], "ey": maps["ey"]}
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--main", default="thick")
    ap.add_argument("--compare", default="thin")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    no, ne = fm.n_ln_zelmon(LAM)
    ns = fm.n_sio2_malitson(LAM)
    main_rows = load(a.main, a.cache)
    data = {
        "meta": {
            "wavelength_um": LAM,
            "indices": {
                "LiNbO3_ordinary": round(no, 5), "LiNbO3_extraordinary": round(ne, 5), "SiO2": round(ns, 5), "air": 1.0,
                "sources": ["Zelmon, Small and Jundt, J. Opt. Soc. Am. B 14, 3319 (1997), congruent LiNbO3 Sellmeier",
                            "Malitson, J. Opt. Soc. Am. 55, 1205 (1965), fused silica Sellmeier"],
            },
            "crystal": "x-cut: film normal along crystal x; c axis (crystal z) in the film plane",
            "angle": "theta = in-plane angle between the propagation direction and the crystal y axis. Solved 0 to 90 "
                     "degrees; theta and 180 - theta give identical indices with mirror-image fields (left-right "
                     "symmetric rib, the lateral-longitudinal tensor component changes sign).",
            "solver": "Full-vector finite-difference eigenmode solver on a Yee grid with all three electric field "
                      "components and the full rotated permittivity tensor, including the lateral-longitudinal "
                      "component; quadratic eigenvalue problem in the propagation constant solved exactly by "
                      "companion linearization and shift-invert Arnoldi. Sub-cell permittivity averaging at the "
                      "slanted sidewalls; horizontal interfaces on grid lines; metal walls far from the mode.",
            "grid": {"step_um": H, "window_x_um": [-XW, XW], "oxide_below_film_um": BOX, "air_above_film_top_um": TOP,
                     "walls": "perfect electric conductor"},
            "validation": "Isotropic slab against the analytic slab equation; anisotropic slab at 0, 30, 60 and 90 "
                          "degrees against a 4 x 4 transfer-matrix solution (agreement about 1e-5 in effective index at "
                          "5 nm grid); diagonal-tensor rib against an independent transverse-field formulation (first "
                          "two modes identical to 1e-7); halving the grid from 20 to 10 nm changes the indices by at most "
                          "1.4e-4; closest approach 30.25 deg at 20 nm and 30.21 deg at 15 nm grid, splitting 0.0027.",
            "modeNames": ["Mode 1 (higher effective index)", "Mode 2"],
            "teFraction": "fraction of the transverse electric field intensity in the lateral component (not weighted by permittivity), "
                          "integral |Ex|^2 / integral (|Ex|^2 + |Ey|^2)",
        },
        "main": profile_block(a.main, main_rows, True),
    }
    if a.compare:
        cmp_rows = load(a.compare, a.cache)
        if cmp_rows:
            data["compare"] = profile_block(a.compare, cmp_rows, False)
    js = ("/* Generated by tools/modesolver/export.py from a full-vector anisotropic mode sweep (see tools/modesolver/README.md).\n"
          "   Do not edit by hand. */\n"
          "window.LMS_XCUT_MODES = " + json.dumps(data, separators=(",", ":")) + ";\n")
    Path(a.out).write_text(js)
    print(f"wrote {a.out}: {len(js)/1024:.0f} KB, {len(data['main']['theta'])} angles")


if __name__ == "__main__":
    main()
