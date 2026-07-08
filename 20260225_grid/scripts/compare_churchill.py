#!/usr/bin/env python3
"""NREL vs Open-Meteo for the sites nearest Churchill County, NV (seat: Fallon,
~39.4735 N, -118.7774 W). wind=0, BESS=40,000 MWh (exact NREL grid point).
NREL has no data above 6 GW solar, so no NREL 10 GW column (no fake data).
"""
from __future__ import annotations

import json
import math
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

FALLON_LAT, FALLON_LON = 39.4735, -118.7774
N_NEAREST = 6

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
nrel_slice = NREL["values_by_combo"][((si6 * nw) + wi0) * nb + bi40]
nrel_by_id = {int(s["site_id"]): nrel_slice[i] for i, s in enumerate(NREL["sites"])}


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# rank ALL sites by distance to Fallon (only those usable: profile + in NREL cache)
ranked = []
for site in sites:
    sid = int(site["site_id"])
    if not (PROFILES / f"site_{sid:04d}.npz").exists() or sid not in nrel_by_id:
        continue
    d = haversine_km(FALLON_LAT, FALLON_LON, float(site["lat"]), float(site["lon"]))
    ranked.append((d, site))
ranked.sort(key=lambda x: x[0])
nearest = ranked[:N_NEAREST]

solar_vals = np.array([6000.0, 10000.0]); wind_vals = np.array([0.0]); bess_vals = np.array([BESS])

print(f"Reference: Fallon, NV (Churchill County seat) {FALLON_LAT}, {FALLON_LON}")
print(f"Usable sites available: {len(ranked)}   |   wind=0, BESS=40,000 MWh (exact grid point)\n")
print(f"=== {N_NEAREST} SITES NEAREST CHURCHILL COUNTY ===")
print(f"  {'site':<22} {'state':>5} {'dist_km':>8} {'6 GW NREL':>10} {'6 GW OM':>9} {'10 GW OM':>9}")
om6s, om10s, nrel6s = [], [], []
for d, site in nearest:
    sid = int(site["site_id"])
    sp, wp, _ = load_profile_cache(PROFILES / f"site_{sid:04d}.npz")
    cube = vectorized_reliability_for_site(sp, wp, solar_vals, wind_vals, bess_vals, assumptions) * 100.0
    n6 = nrel_by_id[sid]; o6 = float(cube[0, 0, 0]); o10 = float(cube[1, 0, 0])
    nrel6s.append(n6); om6s.append(o6); om10s.append(o10)
    print(f"  {site['name']:<22} {site['state']:>5} {d:7.0f}  {n6:9.2f}% {o6:8.2f}% {o10:8.2f}%")
k = len(nearest)
print(f"  {'mean':<22} {'':>5} {'':>8} {sum(nrel6s)/k:9.2f}% {sum(om6s)/k:8.2f}% {sum(om10s)/k:8.2f}%")
