#!/usr/bin/env python3
"""Compare NREL (old, cached) vs Open-Meteo (computed from profiles) reliability
at solar=10 GW / wind=0 / BESS=200 GWh, and at the old cache's true max 6 GW.

NREL: only the cached grid exists (solar max 6000), and the raw NREL profiles are
gone, so NREL at a *true* 10 GW cannot be recomputed -> the old map literally shows
its 6 GW value for any solar >= 6 GW. Open-Meteo profiles are present, so the true
10 GW value is computed here.
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

# NREL cached lookup: site_id -> reliability at (solar=6000, wind=0, bess=200000)
nrel_axes = NREL["axes"]
nw, nb = len(nrel_axes["wind_mw"]), len(nrel_axes["bess_mwh"])
def nrel_combo(si, wi, bi):
    return ((si * nw) + wi) * nb + bi
si6 = nrel_axes["solar_mw"].index(6000.0)
wi0 = nrel_axes["wind_mw"].index(0.0)
bi200 = nrel_axes["bess_mwh"].index(200000.0)
nrel_slice_6gw = NREL["values_by_combo"][nrel_combo(si6, wi0, bi200)]
nrel_by_id = {int(s["site_id"]): nrel_slice_6gw[i] for i, s in enumerate(NREL["sites"])}

# Open-Meteo: compute at solar in {6000, 10000}, wind=0, bess=200000 from profiles
solar_vals = np.array([6000.0, 10000.0])
wind_vals = np.array([0.0])
bess_vals = np.array([200000.0])

NV_IDS = {24, 55, 239, 333, 371, 374, 390, 413, 434, 455, 464, 503, 505, 512, 524, 526, 535, 553, 559}

rows = []  # (site_id, name, state, nrel6, om6, om10)
for site in sites:
    sid = int(site["site_id"])
    p = PROFILES / f"site_{sid:04d}.npz"
    if not p.exists() or sid not in nrel_by_id:
        continue
    solar_profile, wind_profile, _ = load_profile_cache(p)
    cube = vectorized_reliability_for_site(solar_profile, wind_profile, solar_vals, wind_vals, bess_vals, assumptions) * 100.0
    om6 = float(cube[0, 0, 0])
    om10 = float(cube[1, 0, 0])
    rows.append((sid, site["name"], site["state"], nrel_by_id[sid], om6, om10))

def agg(subset):
    if not subset:
        return None
    n = len(subset)
    return (
        sum(r[3] for r in subset) / n,
        sum(r[4] for r in subset) / n,
        sum(r[5] for r in subset) / n,
        n,
    )

print(f"Sites compared (profiles present AND in NREL cache): {len(rows)}\n")
print("Config: wind=0, BESS=200,000 MWh (200 GWh).  '10 GW' on the OLD map = its clamped 6 GW value.\n")

print("=== NEVADA ===")
print(f"  {'site':<22} {'NREL@6GW':>9} {'OM@6GW':>8} {'OM@10GW':>8} {'OM 10vs6':>9} {'OM-NREL@6':>10}")
nv = [r for r in rows if r[0] in NV_IDS]
for r in sorted(nv, key=lambda x: x[1]):
    _, name, _, nrel6, om6, om10 = r
    print(f"  {name:<22} {nrel6:8.2f}% {om6:7.2f}% {om10:7.2f}% {om10-om6:+8.2f} {om6-nrel6:+9.2f}")
a = agg(nv)
if a:
    print(f"  {'NV mean':<22} {a[0]:8.2f}% {a[1]:7.2f}% {a[2]:7.2f}% {a[2]-a[1]:+8.2f} {a[1]-a[0]:+9.2f}")

print("\n=== ALL COMMON SITES ===")
a = agg(rows)
print(f"  n={a[3]}")
print(f"  NREL @ 6 GW  (what old map shows at 'any solar >= 6 GW'): {a[0]:6.2f}%")
print(f"  Open-Meteo @ 6 GW:                                       {a[1]:6.2f}%   (vs NREL: {a[1]-a[0]:+.2f})")
print(f"  Open-Meteo @ 10 GW (true, new axis):                     {a[2]:6.2f}%   (gain over 6 GW: {a[2]-a[1]:+.2f})")
