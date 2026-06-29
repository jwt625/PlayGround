#!/usr/bin/env python3
"""Benchmark an Open-Meteo (ERA5) reliability cache against the NREL (NSRDB/WTK)
cache on identical axes. Matches sites by site_id and reports per-combo and
overall deltas, plus a Nevada focus.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def combo_index(si, wi, bi, nwind, nbess):
    return ((si * nwind) + wi) * nbess + bi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openmeteo", type=Path, required=True)
    ap.add_argument("--nrel", type=Path, default=Path("outputs/us_reliability_map_real_partial.json"))
    args = ap.parse_args()

    om = load(args.openmeteo)
    nr = load(args.nrel)

    for ax in ("solar_mw", "wind_mw", "bess_mwh"):
        if [round(x, 3) for x in om["axes"][ax]] != [round(x, 3) for x in nr["axes"][ax]]:
            raise SystemExit(f"Axis mismatch on {ax}: cannot benchmark.\n  om={om['axes'][ax]}\n  nr={nr['axes'][ax]}")

    solar_axis = om["axes"]["solar_mw"]
    wind_axis = om["axes"]["wind_mw"]
    bess_axis = om["axes"]["bess_mwh"]
    nwind, nbess = len(wind_axis), len(bess_axis)

    # site_id -> column index within each cache's values_by_combo slice
    om_col = {int(s["site_id"]): i for i, s in enumerate(om["sites"])}
    nr_col = {int(s["site_id"]): i for i, s in enumerate(nr["sites"])}
    common = sorted(set(om_col) & set(nr_col))
    state_by_id = {int(s["site_id"]): s.get("state", "") for s in nr["sites"]}
    name_by_id = {int(s["site_id"]): s.get("name", "") for s in nr["sites"]}
    print(f"Sites: open-meteo={len(om_col)}  nrel={len(nr_col)}  common={len(common)}")
    print(f"Combos: {len(om['values_by_combo'])}  (axes match: solar≤{int(max(solar_axis))} wind≤{int(max(wind_axis))} bess≤{int(max(bess_axis))})\n")

    # Overall stats across all combos x common sites
    n = 0
    sum_om = sum_nr = sum_d = sum_ad = sum_d2 = 0.0
    sxx = syy = sxy = sx = sy = 0.0
    for ci in range(len(om["values_by_combo"])):
        om_slice = om["values_by_combo"][ci]
        nr_slice = nr["values_by_combo"][ci]
        for sid in common:
            a = om_slice[om_col[sid]]
            b = nr_slice[nr_col[sid]]
            d = a - b
            n += 1
            sum_om += a; sum_nr += b; sum_d += d; sum_ad += abs(d); sum_d2 += d * d
            sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b
    mean_om, mean_nr = sum_om / n, sum_nr / n
    bias = sum_d / n
    mae = sum_ad / n
    rmse = math.sqrt(sum_d2 / n)
    denom = math.sqrt((n * sxx - sx * sx) * (n * syy - sy * sy))
    corr = (n * sxy - sx * sy) / denom if denom else float("nan")
    print("=== OVERALL (all combos x common sites) ===")
    print(f"  pairs:        {n:,}")
    print(f"  mean reliability:  open-meteo={mean_om:.2f}%   nrel={mean_nr:.2f}%")
    print(f"  bias (om - nrel):  {bias:+.2f} pts")
    print(f"  MAE:               {mae:.2f} pts")
    print(f"  RMSE:              {rmse:.2f} pts")
    print(f"  correlation r:     {corr:.4f}\n")

    # Reference configs
    def ci_for(solar, wind, bess):
        si = solar_axis.index(solar); wi = wind_axis.index(wind); bi = bess_axis.index(bess)
        return combo_index(si, wi, bi, nwind, nbess)

    refs = [
        (6000.0, 0.0, 200000.0),
        (6000.0, 0.0, 40000.0),
        (3000.0, 3000.0, 40000.0),
        (6000.0, 6000.0, 200000.0),
    ]
    print("=== REFERENCE CONFIGS (mean over common sites) ===")
    print(f"  {'solar/wind/bess':>26} | {'open-meteo':>10} | {'nrel':>7} | {'Δ':>7}")
    for solar, wind, bess in refs:
        ci = ci_for(solar, wind, bess)
        om_slice = om["values_by_combo"][ci]; nr_slice = nr["values_by_combo"][ci]
        a = sum(om_slice[om_col[s]] for s in common) / len(common)
        b = sum(nr_slice[nr_col[s]] for s in common) / len(common)
        label = f"{int(solar)}/{int(wind)}/{int(bess)}"
        print(f"  {label:>26} | {a:9.2f}% | {b:6.2f}% | {a-b:+6.2f}")
    print()

    # Nevada focus at solar=6000, wind=0, bess=200000
    ci = ci_for(6000.0, 0.0, 200000.0)
    om_slice = om["values_by_combo"][ci]; nr_slice = nr["values_by_combo"][ci]
    nv = [s for s in common if state_by_id.get(s) == "NV"]
    print(f"=== NEVADA @ solar=6000, wind=0, bess=200000 ({len(nv)} sites) ===")
    print(f"  {'site':<22} {'open-meteo':>10} {'nrel':>8} {'Δ':>7}")
    om_vals, nr_vals = [], []
    for sid in sorted(nv, key=lambda s: name_by_id.get(s, "")):
        a = om_slice[om_col[sid]]; b = nr_slice[nr_col[sid]]
        om_vals.append(a); nr_vals.append(b)
        print(f"  {name_by_id.get(sid,''):<22} {a:9.2f}% {b:7.2f}% {a-b:+6.2f}")
    if nv:
        print(f"  {'NV mean':<22} {sum(om_vals)/len(nv):9.2f}% {sum(nr_vals)/len(nv):7.2f}% {(sum(om_vals)-sum(nr_vals))/len(nv):+6.2f}")


if __name__ == "__main__":
    main()
