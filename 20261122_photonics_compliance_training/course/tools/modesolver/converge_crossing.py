"""Grid convergence of the closest approach (angle and minimum splitting) of the first two modes, main profile.

    uv run --no-project --with numpy --with scipy python converge_crossing.py

For each grid step the splitting n1 - n2 is solved on 29.5 .. 31.5 deg (0.5 deg step) in a smaller window (checked
in validate.py V4b to change the indices by less than 1e-6) and the minimum is located by a parabola through the
three lowest samples. One process, two threads, solves run one after another (memory stays modest); each result is
printed as it arrives.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"

import numpy as np  # noqa: E402

import fdmode as fm  # noqa: E402
from sweep import LAM, PROFILES  # noqa: E402


def job(a):
    h, th = a
    p = PROFILES["thick"]
    no, ne = fm.n_ln_zelmon(LAM)
    ns = fm.n_sio2_malitson(LAM)
    g = fm.make_grid(-2.5, 2.5, -1.2, p["t_film"] + 0.9, h)
    st = fm.rib_structure(p["t_film"], p["t_slab"], p["w_top"], p["sidewall"], fm.ln_tensor_xcut(no, ne, th), ns, 1.0)
    ms = fm.solve(fm.build(g, st, LAM), 2.15, k=4, n_min=ns)
    return h, th, ms[0].neff, ms[1].neff


def main():
    ths = np.round(np.arange(29.5, 31.5 + 1e-9, 0.5), 3)
    hs = (0.04, 0.03, 0.02, 0.015)
    res = []
    for h in hs:
        for t in ths:
            r = job((h, float(t)))
            res.append(r)
            print(f"  h {h*1000:4.0f} nm  theta {t:5.2f}  n1 {r[2]:.6f}  n2 {r[3]:.6f}", flush=True)
    for h in hs:
        r = sorted([x for x in res if x[0] == h], key=lambda x: x[1])
        t = np.array([x[1] for x in r])
        d = np.array([x[2] - x[3] for x in r])
        i = int(np.clip(np.argmin(d), 1, len(d) - 2))
        c = np.polyfit(t[i - 1:i + 2], d[i - 1:i + 2], 2)
        tc = -c[1] / (2 * c[0])
        dmin = np.polyval(c, tc)
        print(f"h {h*1000:4.0f} nm: closest approach at {tc:.2f} deg, minimum splitting {dmin:.5f}, "
              f"n1 at 30.5 deg {r[2][2]:.5f}", flush=True)


if __name__ == "__main__":
    main()
