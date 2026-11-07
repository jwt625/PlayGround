"""Validation: nondispersive alpha-profile fiber, numerical optimum alpha vs Olshansky-Keck 2 - 12 Delta / 5.

WKB theory assumes many modes and ignores cutoff wave effects, so the comparison is made both for all guided modes
and for the inner mode groups only (principal number p <= frac * p_max). Writes outputs/validation/alpha_opt_*.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grinopt.solver import Grid, solve_modes  # noqa: E402


def sigma(alpha: float, lam: float, a: float, n1: float, delta: float, frac: float) -> float:
    grid = Grid.make(a + 15.0, 0.05)
    rho = np.clip(grid.r / a, 0, 1)
    n = n1 * np.sqrt(1 - 2 * delta * rho**alpha)
    n2 = float(n[-1])
    m = solve_modes(grid, lam, n, n, n2)  # nondispersive: material group index = n
    keep = m.principal <= frac * m.principal.max()
    w, t = m.degeneracy[keep], m.tau_ps_per_m[keep]
    mu = np.sum(w * t) / w.sum()
    return float(np.sqrt(np.sum(w * (t - mu) ** 2) / w.sum()))


def main() -> None:
    out = []
    for lam, a, delta in ((0.85, 25.0, 0.01), (0.43, 25.0, 0.01), (0.43, 40.0, 0.01), (0.43, 25.0, 0.005)):
        n1 = 1.46
        a_ok = 2 - 12 * delta / 5
        row = {"lam_um": lam, "a_um": a, "delta": delta, "V": 2 * np.pi / lam * a * n1 * np.sqrt(2 * delta),
               "alpha_ok": a_ok}
        for frac in (1.0, 0.8, 0.6):
            grid_al = np.arange(1.90, 2.20, 0.005)
            s = [sigma(al, lam, a, n1, delta, frac) for al in grid_al]
            i = int(np.argmin(s))
            def obj(al, lam=lam, a=a, n1=n1, delta=delta, frac=frac):
                return sigma(al, lam, a, n1, delta, frac)

            res = minimize_scalar(obj,
                                  bounds=(grid_al[max(i - 1, 0)], grid_al[min(i + 1, len(grid_al) - 1)]),
                                  method="bounded", options={"xatol": 1e-4})
            row[f"alpha_num_frac{frac}"] = float(res.x)
            row[f"sigma_min_frac{frac}_ps_per_m"] = float(res.fun)
        # recalled (UNVERIFIED, not found in a primary source 2026-11-06) WKB minimum rms: N1 Delta^2 / (20 sqrt(3) c)
        row["sigma_wkb_unverified_ps_per_m"] = n1 * delta**2 / (20 * np.sqrt(3)) * 1e12 / 299_792_458.0
        out.append(row)
        print(json.dumps(row), flush=True)
    d = ROOT / "outputs" / "validation"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"alpha_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
