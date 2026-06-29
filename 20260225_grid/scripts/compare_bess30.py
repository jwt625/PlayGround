#!/usr/bin/env python3
"""NREL vs Open-Meteo at solar 6 GW & 10 GW, wind=0, BESS=30 GWh (30,000 MWh).

BESS 30,000 MWh is NOT an NREL grid point (grid is 20k,40k,...), so NREL is
linearly interpolated between 20k and 40k (t=0.5), exactly as the browser does.
Solar 10 GW exceeds the NREL solar axis max (6 GW) -> NREL clamps to its 6 GW
value (same number as the 6 GW row). Open-Meteo computes every point from profiles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from precompute_us_reliability_map import (
    BatchAssumptions,
    vectorized_reliability_for_site,
    load_sites,
    load_yaml,
    mean_value,
)
from resource_profiles import load_profile_cache

ROOT = Path(".")
PROFILES = ROOT / "references/data/solar_wind_profiles"
NREL = json.loads((ROOT / "outputs/us_reliability_map_real_partial.json").read_text())

config = load_yaml(ROOT / "config/assumptions_2026_us_ai_datacenter.yaml")
assumptions = BatchAssumptions(
    workload_mw=1000.0,
    battery_round_trip_efficiency=mean_value(config["technology_parameters"]["bess_li_ion"]["round_trip_efficiency"]),
    battery_usable_depth_of_discharge=mean_value(config["technology_parameters"]["bess_li_ion"]["usable_depth_of_discharge"]),
    battery_reserve_fraction=0.0,
)
sites = load_sites(ROOT / "outputs/us_towns_dedup.json")

BESS = 30000.0
ax = NREL["axes"]
nw, nb = len(ax["wind_mw"]), len(ax["bess_mwh"])
def nrel_combo(si, wi, bi):
    return ((si * nw) + wi) * nb + bi
si6 = ax["solar_mw"].index(6000.0); wi0 = ax["wind_mw"].index(0.0)
bi_lo = ax["bess_mwh"].index(20000.0); bi_hi = ax["bess_mwh"].index(40000.0)
t = (BESS - 20000.0) / (40000.0 - 20000.0)  # 0.5

# NREL @ (solar=6000, wind=0, bess=30000) interpolated; 10 GW clamps to same value.
slo = NREL["values_by_combo"][nrel_combo(si6, wi0, bi_lo)]
shi = NREL["values_by_combo"][nrel_combo(si6, wi0, bi_hi)]
nrel_by_id = {int(s["site_id"]): slo[i] * (1 - t) + shi[i] * t for i, s in enumerate(NREL["sites"])}

solar_vals = np.array([6000.0, 10000.0]); wind_vals = np.array([0.0]); bess_vals = np.array([BESS])
NV_IDS = {24,55,239,333,371,374,390,413,434,455,464,503,505,512,524,526,535,553,559}

rows = []  # sid, name, state, nrel6, nrel10, om6, om10
for site in sites:
    sid = int(site["site_id"])
    p = PROFILES / f"site_{sid:04d}.npz"
    if not p.exists() or sid not in nrel_by_id:
        continue
    sp, wp, _ = load_profile_cache(p)
    cube = vectorized_reliability_for_site(sp, wp, solar_vals, wind_vals, bess_vals, assumptions) * 100.0
    nrel6 = nrel_by_id[sid]
    rows.append((sid, site["name"], site["state"], nrel6, nrel6, float(cube[0,0,0]), float(cube[1,0,0])))

def agg(sub):
    n = len(sub)
    return (sum(r[3] for r in sub)/n, sum(r[5] for r in sub)/n,
            sum(r[4] for r in sub)/n, sum(r[6] for r in sub)/n, n)

print(f"Sites compared: {len(rows)}   |   wind=0, BESS=30,000 MWh (30 GWh)")
print("NREL: 6 GW = grid point (bess interpolated); 10 GW = clamped to 6 GW.  Open-Meteo: all computed.\n")

print("=== NEVADA ===")
print(f"  {'site':<22} {'6 GW':>17} | {'10 GW':>17}")
print(f"  {'':<22} {'NREL':>8} {'OM':>8} | {'NREL*':>8} {'OM':>8}")
nv = [r for r in rows if r[0] in NV_IDS]
for r in sorted(nv, key=lambda x: x[1]):
    _, name, _, nrel6, nrel10, om6, om10 = r
    print(f"  {name:<22} {nrel6:7.2f}% {om6:7.2f}% | {nrel10:7.2f}% {om10:7.2f}%")
if nv:
    a = agg(nv)
    print(f"  {'NV mean':<22} {a[0]:7.2f}% {a[1]:7.2f}% | {a[2]:7.2f}% {a[3]:7.2f}%")

print("\n=== ALL COMMON SITES ===")
a = agg(rows)
print(f"  n={a[4]}")
print(f"  6 GW solar :  NREL {a[0]:6.2f}%   |  Open-Meteo {a[1]:6.2f}%   (Δ {a[1]-a[0]:+.2f})")
print(f"  10 GW solar:  NREL {a[2]:6.2f}%*  |  Open-Meteo {a[3]:6.2f}%   (Δ {a[3]-a[2]:+.2f})")
print("  * NREL 10 GW = clamped 6 GW value (no data above 6 GW).")
