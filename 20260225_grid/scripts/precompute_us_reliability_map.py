#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HOURS_PER_YEAR = 8760
# CONUS reference bounds anchored to USGS extreme-point coordinates:
# west: Cape Alava, WA 124°44'W
# east: West Quoddy Head, ME 66°57'W
# north: Lake of the Woods Projection, MN 49°23'N
# south: Key West, FL 24°32'N
CONUS_MIN_LAT = 24.533
CONUS_MAX_LAT = 49.383
CONUS_MIN_LON = -124.733
CONUS_MAX_LON = -66.950
def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def mean_value(node: dict[str, Any]) -> float:
    value = node.get("value", node)
    return float(value["mean"])


def load_sites(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_points = payload["points"]
    sites: list[dict[str, Any]] = []
    for idx, point in enumerate(raw_points):
        sites.append(
            {
                "site_id": idx,
                "lat": round(float(point["lat"]), 3),
                "lon": round(float(point["lon"]), 3),
                "name": point.get("name", ""),
                "state": point.get("state", ""),
                "population": int(point.get("population", 0)),
            }
        )
    return sites


def solar_capacity_factor_from_location(lat: float, lon: float) -> float:
    sunbelt = np.clip((37.0 - lat) / 12.0, 0.0, 1.0)
    southwest = np.clip((-95.0 - lon) / 25.0, 0.0, 1.0)
    east_penalty = np.clip((lon + 90.0) / 18.0, 0.0, 1.0)
    cf = 0.18 + 0.10 * sunbelt + 0.06 * southwest - 0.03 * east_penalty
    return float(np.clip(cf, 0.14, 0.34))


def wind_capacity_factor_from_location(lat: float, lon: float) -> float:
    plains = np.exp(-((lon + 100.0) / 9.5) ** 2)
    texas = 0.8 * np.exp(-((lon + 99.0) / 7.5) ** 2) * np.exp(-((lat - 31.0) / 5.0) ** 2)
    northern = 0.5 * np.exp(-((lat - 44.0) / 6.0) ** 2)
    coastal = 0.25 * np.exp(-((lon + 73.0) / 5.0) ** 2)
    cf = 0.18 + 0.16 * plains + 0.10 * texas + 0.05 * northern + 0.03 * coastal
    return float(np.clip(cf, 0.16, 0.52))


def synthetic_solar_profile(hours: int, annual_cf: float, lat: float) -> np.ndarray:
    profile = np.zeros(hours, dtype=float)
    latitude_scale = np.clip((49.0 - lat) / 24.0, 0.6, 1.15)
    for hour in range(hours):
        day = hour // 24
        hod = hour % 24
        seasonal = 0.70 + 0.30 * np.sin(2.0 * np.pi * (day - 81) / 365.0)
        daylight = max(0.0, np.sin(np.pi * (hod - 6) / 12.0))
        profile[hour] = latitude_scale * seasonal * daylight
    return profile * (annual_cf / max(float(profile.mean()), 1e-9))


def synthetic_wind_profile(hours: int, annual_cf: float, lat: float, lon: float) -> np.ndarray:
    t = np.arange(hours, dtype=float)
    nocturnal = 0.10 * np.cos(2.0 * np.pi * ((t % 24.0) - 2.0) / 24.0)
    winter = 0.14 * np.cos(2.0 * np.pi * t / HOURS_PER_YEAR)
    plains_factor = np.exp(-((lon + 100.0) / 10.0) ** 2)
    northern_factor = np.clip((lat - 31.0) / 15.0, 0.0, 1.0)
    synoptic = 0.08 * np.sin(2.0 * np.pi * t / (24.0 * 6.5) + (lon + 100.0) / 8.0)
    raw = 1.0 + nocturnal + winter * (0.8 + 0.4 * northern_factor) + synoptic * (0.6 + 0.8 * plains_factor)
    raw = np.clip(raw, 0.05, None)
    raw = raw / raw.mean()
    return np.clip(raw * annual_cf, 0.0, 0.95)


@dataclass
class BatchAssumptions:
    workload_mw: float
    battery_round_trip_efficiency: float
    battery_usable_depth_of_discharge: float
    battery_reserve_fraction: float


def vectorized_reliability_for_site(
    solar_profile_per_mw: np.ndarray,
    wind_profile_per_mw: np.ndarray,
    solar_mw_values: np.ndarray,
    wind_mw_values: np.ndarray,
    bess_mwh_values: np.ndarray,
    assumptions: BatchAssumptions,
) -> np.ndarray:
    solar_grid, wind_grid, bess_grid = np.meshgrid(
        solar_mw_values,
        wind_mw_values,
        bess_mwh_values,
        indexing="ij",
    )
    solar_flat = solar_grid.reshape(-1)
    wind_flat = wind_grid.reshape(-1)
    bess_flat = bess_grid.reshape(-1)
    combo_count = solar_flat.size

    charge_eff = np.sqrt(assumptions.battery_round_trip_efficiency)
    discharge_eff = np.sqrt(assumptions.battery_round_trip_efficiency)
    min_soc = bess_flat * max(0.0, 1.0 - assumptions.battery_usable_depth_of_discharge)
    reserve_soc = np.maximum(min_soc, bess_flat * assumptions.battery_reserve_fraction)
    soc = reserve_soc.copy()

    served_hours = np.zeros(combo_count, dtype=np.float64)
    load = assumptions.workload_mw

    for hour in range(HOURS_PER_YEAR):
        generation = solar_profile_per_mw[hour] * solar_flat + wind_profile_per_mw[hour] * wind_flat
        served_from_generation = np.minimum(generation, load)
        deficit = load - served_from_generation
        surplus = generation - served_from_generation

        max_charge = np.maximum(0.0, bess_flat - soc) / max(charge_eff, 1e-9)
        charge = np.minimum(surplus, max_charge)
        soc += charge * charge_eff

        max_discharge = np.maximum(0.0, soc - min_soc) * discharge_eff
        discharge = np.minimum(deficit, max_discharge)
        soc -= discharge / max(discharge_eff, 1e-9)
        remaining_deficit = deficit - discharge

        served_hours += (remaining_deficit <= 1e-9).astype(np.float64)

    return (served_hours / HOURS_PER_YEAR).reshape(
        len(solar_mw_values),
        len(wind_mw_values),
        len(bess_mwh_values),
    )


def parse_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute deterministic U.S. reliability map cache.")
    parser.add_argument("--config", type=Path, default=Path("config/assumptions_2026_us_ai_datacenter.yaml"))
    parser.add_argument("--site-json", type=Path, default=Path("outputs/us_towns_dedup.json"))
    parser.add_argument("--workload-mw", type=float, default=1000.0)
    parser.add_argument("--solar-grid-mw", default="0,600,1200,1800,2400,3000,3600,4200,4800,5400,6000")
    parser.add_argument("--wind-grid-mw", default="0,600,1200,1800,2400,3000,3600,4200,4800,5400,6000")
    parser.add_argument("--bess-grid-mwh", default="0,20000,40000,60000,80000,100000,120000,140000,160000,180000,200000")
    parser.add_argument("--out", type=Path, default=Path("outputs/us_reliability_map_cache.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    sites = load_sites(args.site_json)
    solar_values = np.array(parse_list(args.solar_grid_mw), dtype=np.float64)
    wind_values = np.array(parse_list(args.wind_grid_mw), dtype=np.float64)
    bess_values = np.array(parse_list(args.bess_grid_mwh), dtype=np.float64)

    assumptions = BatchAssumptions(
        workload_mw=args.workload_mw,
        battery_round_trip_efficiency=mean_value(config["technology_parameters"]["bess_li_ion"]["round_trip_efficiency"]),
        battery_usable_depth_of_discharge=mean_value(config["technology_parameters"]["bess_li_ion"]["usable_depth_of_discharge"]),
        battery_reserve_fraction=0.0,
    )

    values_by_combo: list[list[float]] = [
        [0.0 for _ in range(len(sites))]
        for _ in range(len(solar_values) * len(wind_values) * len(bess_values))
    ]

    for site_index, site in enumerate(sites):
        lat = float(site["lat"])
        lon = float(site["lon"])
        solar_cf = solar_capacity_factor_from_location(lat, lon)
        wind_cf = wind_capacity_factor_from_location(lat, lon)
        solar_profile = synthetic_solar_profile(HOURS_PER_YEAR, solar_cf, lat)
        wind_profile = synthetic_wind_profile(HOURS_PER_YEAR, wind_cf, lat, lon)
        site_cube = vectorized_reliability_for_site(
            solar_profile,
            wind_profile,
            solar_values,
            wind_values,
            bess_values,
            assumptions,
        )
        combo_index = 0
        for s_idx in range(len(solar_values)):
            for w_idx in range(len(wind_values)):
                for b_idx in range(len(bess_values)):
                    values_by_combo[combo_index][site_index] = round(float(site_cube[s_idx, w_idx, b_idx] * 100.0), 4)
                    combo_index += 1

    payload = {
        "meta": {
            "model": "deterministic_synthetic_resource_v1",
            "workload_mw": args.workload_mw,
            "battery_interpretation": "energy_only",
            "battery_power_assumption": "not_binding_up_to_workload",
            "hours": HOURS_PER_YEAR,
            "site_source": str(args.site_json),
            "site_count": len(sites),
            "bounds": {
                "min_lat": CONUS_MIN_LAT,
                "max_lat": CONUS_MAX_LAT,
                "min_lon": CONUS_MIN_LON,
                "max_lon": CONUS_MAX_LON,
            },
            "sources": {
                "us_town_points": "https://download.geonames.org/export/dump/US.zip",
                "site_dedup_input": str(args.site_json),
            },
            "recommended_defaults": {
                "solar_mw": 3000.0,
                "wind_mw": 3000.0,
                "bess_mwh": 40000.0,
            },
        },
        "axes": {
            "solar_mw": solar_values.tolist(),
            "wind_mw": wind_values.tolist(),
            "bess_mwh": bess_values.tolist(),
        },
        "sites": sites,
        "values_by_combo": values_by_combo,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({"sites": len(sites), "combos": len(values_by_combo), "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
