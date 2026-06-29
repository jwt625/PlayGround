#!/usr/bin/env python3
"""NREL vs Open-Meteo, wind=0, BESS=40,000 MWh (40 GWh, a real NREL grid point
-> no interpolation). Columns: solar 6 GW (NREL + OM) and solar 10 GW (OM only).
NREL has no data above 6 GW solar, so its 10 GW value would be a clamp artifact
and is intentionally OMITTED (no fake data).
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

BESS = 40000.0
ax = NREL["axes"]
nw, nb = len(ax["wind_mw"]), len(ax["bess_mwh"])
si6 = ax["solar_mw"].index(6000.0); wi0 = ax["wind_mw"].index(0.0); bi40 = ax["bess_mwh"].index(BESS)
nrel_slice = NREL["values_by_combo"][((si6 * nw) + wi0) * nb + bi40]  # exact grid point
nrel_by_id = {int(s["site_id"]): nrel_slice[i] for i, s in enumerate(NREL["sites"])}

solar_vals = np.array([6000.0, 10000.0]); wind_vals = np.array([0.0]); bess_vals = np.array([BESS])
NV_IDS = {24,55,239,333,371,374,390,413,434,455,464,503,505,512,524,526,535,553,559}

rows = []  # sid, name, state, nrel6, om6, om10
for site in sites:
    sid = int(site["site_id"])
    p = PROFILES / f"site_{sid:04d}.npz"
    if not p.exists() or sid not in nrel_by_id:
        continue
    sp, wp, _ = load_profile_cache(p)
    cube = vectorized_reliability_for_site(sp, wp, solar_vals, wind_vals, bess_vals, assumptions) * 100.0
    rows.append((sid, site["name"], site["state"], nrel_by_id[sid], float(cube[0,0,0]), float(cube[1,0,0])))

def agg(sub):
    n = len(sub)
    return (sum(r[3] for r in sub)/n, sum(r[4] for r in sub)/n, sum(r[5] for r in sub)/n, n)

print(f"Sites compared: {len(rows)}   |   wind=0, BESS=40,000 MWh (40 GWh, exact NREL grid point)\n")

print("=== NEVADA ===")
print(f"  {'site':<22} {'6 GW NREL':>10} {'6 GW OM':>9} {'10 GW OM':>9}")
nv = [r for r in rows if r[0] in NV_IDS]
for r in sorted(nv, key=lambda x: x[1]):
    _, name, _, nrel6, om6, om10 = r
    print(f"  {name:<22} {nrel6:9.2f}% {om6:8.2f}% {om10:8.2f}%")
if nv:
    a = agg(nv)
    print(f"  {'NV mean':<22} {a[0]:9.2f}% {a[1]:8.2f}% {a[2]:8.2f}%")

print("\n=== ALL COMMON SITES ===")
a = agg(rows)
print(f"  n={a[3]}")
print(f"  6 GW solar :  NREL {a[0]:6.2f}%   |  Open-Meteo {a[1]:6.2f}%   (Δ {a[1]-a[0]:+.2f})")
print(f"  10 GW solar:  (NREL n/a — no data above 6 GW)  |  Open-Meteo {a[2]:6.2f}%")
