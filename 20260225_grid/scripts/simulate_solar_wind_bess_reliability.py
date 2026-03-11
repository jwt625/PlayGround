#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


HOURS_PER_YEAR = 8760


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def mean_value(node: dict[str, Any]) -> float:
    value = node.get("value", node)
    return float(value["mean"])


def scale_nameplate_mw(config: dict[str, Any], scale_case: str) -> float:
    return mean_value(config["scale_cases_mw"][scale_case]["facility_nameplate_mw"])


def scale_it_fraction(config: dict[str, Any], scale_case: str) -> float:
    return mean_value(config["scale_cases_mw"][scale_case]["it_load_fraction_of_facility"])


@dataclass
class ReliabilityInputs:
    scale_case: str
    solar_mw: float
    wind_mw: float
    bess_power_mw: float
    bess_energy_mwh: float
    facility_mw: float
    it_load_fraction: float
    utilization: float
    pue: float
    critical_load_fraction: float
    deferrable_load_fraction: float
    interruptible_load_fraction: float
    solar_capacity_factor: float
    wind_capacity_factor: float
    bess_round_trip_efficiency: float
    bess_usable_depth_of_discharge: float
    bess_reserve_fraction: float
    grid_import_limit_mw: float
    grid_charge_limit_mw: float
    allow_grid_charging: bool
    outage_start_hour: int
    outage_duration_hours: int
    dispatch_reserve_for_outage: bool
    random_seed: int


@dataclass
class ReliabilityMetrics:
    hours_simulated: int
    uptime_fraction: float
    outage_hours: float
    loss_of_load_hours: float
    unserved_energy_mwh: float
    renewable_curtailment_mwh: float
    grid_import_mwh: float
    battery_charge_mwh: float
    battery_discharge_mwh: float
    battery_equivalent_cycles: float
    min_soc_mwh: float
    end_soc_mwh: float
    island_survival_hours: float
    solar_generation_mwh: float
    wind_generation_mwh: float
    solar_share_of_served_energy: float
    wind_share_of_served_energy: float
    battery_share_of_served_energy: float
    grid_share_of_served_energy: float
    served_energy_mwh: float
    critical_load_mw: float
    nominal_load_mw: float


def synthetic_solar_profile(hours: int, annual_cf: float) -> np.ndarray:
    profile = np.zeros(hours, dtype=float)
    for hour in range(hours):
        day = hour // 24
        hod = hour % 24
        seasonal = 0.75 + 0.25 * np.sin(2.0 * np.pi * (day - 81) / 365.0)
        daylight = max(0.0, np.sin(np.pi * (hod - 6) / 12.0))
        profile[hour] = seasonal * daylight
    mean_profile = float(profile.mean())
    if mean_profile <= 0.0:
        return profile
    return profile * (annual_cf / mean_profile)


def synthetic_wind_profile(hours: int, annual_cf: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(hours, dtype=float)
    daily = 0.10 * np.cos(2.0 * np.pi * ((t % 24.0) - 2.0) / 24.0)
    seasonal = 0.12 * np.cos(2.0 * np.pi * t / HOURS_PER_YEAR)
    weather = rng.normal(0.0, 0.18, size=hours)
    raw = 1.0 + daily + seasonal + weather
    raw = np.clip(raw, 0.05, None)
    raw = raw / raw.mean()
    return np.clip(raw * annual_cf, 0.0, 0.95)


def build_load_profile(hours: int, nominal_load_mw: float, critical_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    load = np.zeros(hours, dtype=float)
    critical = np.zeros(hours, dtype=float)
    for hour in range(hours):
        hod = hour % 24
        daily = 1.0 + 0.03 * np.sin(2.0 * np.pi * (hod - 15) / 24.0)
        load[hour] = nominal_load_mw * daily
        critical[hour] = load[hour] * critical_fraction
    return load, critical


def build_grid_availability(hours: int, outage_start_hour: int, outage_duration_hours: int) -> np.ndarray:
    availability = np.ones(hours, dtype=bool)
    if outage_duration_hours <= 0:
        return availability
    start = max(0, min(hours - 1, outage_start_hour))
    end = max(start, min(hours, start + outage_duration_hours))
    availability[start:end] = False
    return availability


def simulate(inputs: ReliabilityInputs) -> ReliabilityMetrics:
    solar_cf = synthetic_solar_profile(HOURS_PER_YEAR, inputs.solar_capacity_factor)
    wind_cf = synthetic_wind_profile(HOURS_PER_YEAR, inputs.wind_capacity_factor, inputs.random_seed)
    nominal_load_mw = inputs.facility_mw * inputs.it_load_fraction * inputs.utilization * inputs.pue
    load_mw, critical_load_mw = build_load_profile(HOURS_PER_YEAR, nominal_load_mw, inputs.critical_load_fraction)

    effective_critical_fraction = max(
        0.0,
        inputs.critical_load_fraction - inputs.deferrable_load_fraction - inputs.interruptible_load_fraction,
    )
    effective_critical_load_mw = load_mw * effective_critical_fraction
    grid_available = build_grid_availability(HOURS_PER_YEAR, inputs.outage_start_hour, inputs.outage_duration_hours)

    charge_eff = np.sqrt(inputs.bess_round_trip_efficiency)
    discharge_eff = np.sqrt(inputs.bess_round_trip_efficiency)
    min_soc_mwh = inputs.bess_energy_mwh * max(0.0, 1.0 - inputs.bess_usable_depth_of_discharge)
    reserve_soc_mwh = max(min_soc_mwh, inputs.bess_energy_mwh * inputs.bess_reserve_fraction)
    soc_mwh = max(reserve_soc_mwh, inputs.bess_energy_mwh * 0.5)

    outage_hours = 0.0
    unserved_energy_mwh = 0.0
    curtailment_mwh = 0.0
    grid_import_mwh = 0.0
    battery_charge_mwh = 0.0
    battery_discharge_mwh = 0.0
    served_energy_mwh = 0.0
    solar_generation_mwh = 0.0
    wind_generation_mwh = 0.0
    solar_served_mwh = 0.0
    wind_served_mwh = 0.0
    grid_served_mwh = 0.0
    battery_served_mwh = 0.0
    island_survival_hours = float(inputs.outage_duration_hours)
    outage_failed = False

    for hour in range(HOURS_PER_YEAR):
        solar_mwh = inputs.solar_mw * solar_cf[hour]
        wind_mwh = inputs.wind_mw * wind_cf[hour]
        solar_generation_mwh += solar_mwh
        wind_generation_mwh += wind_mwh

        remaining_load_mwh = effective_critical_load_mw[hour]
        served_from_solar = min(remaining_load_mwh, solar_mwh)
        remaining_load_mwh -= served_from_solar
        solar_surplus_mwh = solar_mwh - served_from_solar

        served_from_wind = min(remaining_load_mwh, wind_mwh)
        remaining_load_mwh -= served_from_wind
        wind_surplus_mwh = wind_mwh - served_from_wind

        grid_served_this_hour = 0.0
        if grid_available[hour] and inputs.grid_import_limit_mw > 0.0:
            grid_served_this_hour = min(remaining_load_mwh, inputs.grid_import_limit_mw)
            remaining_load_mwh -= grid_served_this_hour
            grid_import_mwh += grid_served_this_hour

        soc_floor = reserve_soc_mwh if (inputs.dispatch_reserve_for_outage and grid_available[hour]) else min_soc_mwh
        max_discharge_mwh = min(inputs.bess_power_mw, max(0.0, soc_mwh - soc_floor) * discharge_eff)
        served_from_battery = min(remaining_load_mwh, max_discharge_mwh)
        if served_from_battery > 0.0:
            soc_mwh -= served_from_battery / discharge_eff
            remaining_load_mwh -= served_from_battery
            battery_discharge_mwh += served_from_battery

        renewable_surplus_mwh = solar_surplus_mwh + wind_surplus_mwh
        max_charge_mwh = min(inputs.bess_power_mw, max(0.0, inputs.bess_energy_mwh - soc_mwh) / charge_eff)
        charge_from_renewables = min(renewable_surplus_mwh, max_charge_mwh)
        if charge_from_renewables > 0.0:
            soc_mwh += charge_from_renewables * charge_eff
            battery_charge_mwh += charge_from_renewables
            renewable_surplus_mwh -= charge_from_renewables

        if inputs.allow_grid_charging and grid_available[hour] and inputs.grid_charge_limit_mw > 0.0:
            extra_charge_cap_mwh = min(
                inputs.grid_charge_limit_mw,
                max(0.0, inputs.bess_energy_mwh - soc_mwh) / charge_eff,
            )
            if extra_charge_cap_mwh > 0.0:
                soc_mwh += extra_charge_cap_mwh * charge_eff
                battery_charge_mwh += extra_charge_cap_mwh
                grid_import_mwh += extra_charge_cap_mwh

        curtailment_mwh += max(0.0, renewable_surplus_mwh)
        served_this_hour = effective_critical_load_mw[hour] - remaining_load_mwh
        served_energy_mwh += served_this_hour
        solar_served_mwh += served_from_solar
        wind_served_mwh += served_from_wind
        grid_served_mwh += grid_served_this_hour
        battery_served_mwh += served_from_battery

        if remaining_load_mwh > 1e-9:
            outage_hours += 1.0
            unserved_energy_mwh += remaining_load_mwh
            if not grid_available[hour] and not outage_failed:
                island_survival_hours = float(max(0, hour - inputs.outage_start_hour))
                outage_failed = True

    if inputs.outage_duration_hours <= 0:
        island_survival_hours = 0.0
    elif not outage_failed:
        island_survival_hours = float(inputs.outage_duration_hours)

    throughput_mwh = battery_charge_mwh + battery_discharge_mwh
    equivalent_cycles = throughput_mwh / max(2.0 * inputs.bess_energy_mwh, 1e-9)
    uptime_fraction = 1.0 - outage_hours / HOURS_PER_YEAR

    return ReliabilityMetrics(
        hours_simulated=HOURS_PER_YEAR,
        uptime_fraction=uptime_fraction,
        outage_hours=outage_hours,
        loss_of_load_hours=outage_hours,
        unserved_energy_mwh=unserved_energy_mwh,
        renewable_curtailment_mwh=curtailment_mwh,
        grid_import_mwh=grid_import_mwh,
        battery_charge_mwh=battery_charge_mwh,
        battery_discharge_mwh=battery_discharge_mwh,
        battery_equivalent_cycles=equivalent_cycles,
        min_soc_mwh=min_soc_mwh,
        end_soc_mwh=soc_mwh,
        island_survival_hours=island_survival_hours,
        solar_generation_mwh=solar_generation_mwh,
        wind_generation_mwh=wind_generation_mwh,
        solar_share_of_served_energy=solar_served_mwh / max(served_energy_mwh, 1e-9),
        wind_share_of_served_energy=wind_served_mwh / max(served_energy_mwh, 1e-9),
        battery_share_of_served_energy=battery_served_mwh / max(served_energy_mwh, 1e-9),
        grid_share_of_served_energy=grid_served_mwh / max(served_energy_mwh, 1e-9),
        served_energy_mwh=served_energy_mwh,
        critical_load_mw=float(effective_critical_load_mw.mean()),
        nominal_load_mw=float(load_mw.mean()),
    )


def default_inputs_from_config(config: dict[str, Any], scale_case: str) -> ReliabilityInputs:
    facility_mw = scale_nameplate_mw(config, scale_case)
    it_fraction = scale_it_fraction(config, scale_case)
    utilization = mean_value(config["workload_and_compute"]["cluster_utilization"])
    pue = mean_value(config["thermal_and_efficiency"]["pue_grid_or_turbine"])
    solar_cf = mean_value(config["technology_parameters"]["solar_pv_utility"]["net_capacity_factor"])
    wind_cf = 0.38
    bess_rte = mean_value(config["technology_parameters"]["bess_li_ion"]["round_trip_efficiency"])
    bess_dod = mean_value(config["technology_parameters"]["bess_li_ion"]["usable_depth_of_discharge"])

    return ReliabilityInputs(
        scale_case=scale_case,
        solar_mw=facility_mw,
        wind_mw=facility_mw * 0.5,
        bess_power_mw=facility_mw * 0.5,
        bess_energy_mwh=facility_mw * 2.0,
        facility_mw=facility_mw,
        it_load_fraction=it_fraction,
        utilization=utilization,
        pue=pue,
        critical_load_fraction=0.90,
        deferrable_load_fraction=mean_value(config["workload_and_compute"]["deferrable_workload_fraction"]),
        interruptible_load_fraction=mean_value(config["workload_and_compute"]["interruptible_workload_fraction"]),
        solar_capacity_factor=solar_cf,
        wind_capacity_factor=wind_cf,
        bess_round_trip_efficiency=bess_rte,
        bess_usable_depth_of_discharge=bess_dod,
        bess_reserve_fraction=0.10,
        grid_import_limit_mw=0.0,
        grid_charge_limit_mw=0.0,
        allow_grid_charging=False,
        outage_start_hour=18 * 24,
        outage_duration_hours=72,
        dispatch_reserve_for_outage=True,
        random_seed=7,
    )


def write_metrics_json(path: Path, inputs: ReliabilityInputs, metrics: ReliabilityMetrics) -> None:
    payload = {
        "inputs": asdict(inputs),
        "metrics": asdict(metrics),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_metrics_csv(path: Path, inputs: ReliabilityInputs, metrics: ReliabilityMetrics) -> None:
    rows = []
    for key, value in asdict(inputs).items():
        rows.append({"section": "inputs", "name": key, "value": value})
    for key, value in asdict(metrics).items():
        rows.append({"section": "metrics", "name": key, "value": value})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "name", "value"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic solar-wind-BESS AIDC reliability simulation.")
    parser.add_argument("--config", type=Path, default=Path("config/assumptions_2026_us_ai_datacenter.yaml"))
    parser.add_argument("--scale-case", default="regional_100", choices=["edge_10", "regional_100", "hyperscale_1000"])
    parser.add_argument("--solar-mw", type=float)
    parser.add_argument("--wind-mw", type=float)
    parser.add_argument("--bess-power-mw", type=float)
    parser.add_argument("--bess-energy-mwh", type=float)
    parser.add_argument("--critical-load-fraction", type=float)
    parser.add_argument("--grid-import-limit-mw", type=float)
    parser.add_argument("--grid-charge-limit-mw", type=float)
    parser.add_argument("--allow-grid-charging", action="store_true")
    parser.add_argument("--solar-capacity-factor", type=float)
    parser.add_argument("--wind-capacity-factor", type=float)
    parser.add_argument("--outage-start-hour", type=int)
    parser.add_argument("--outage-duration-hours", type=int)
    parser.add_argument("--no-reserve", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-prefix", type=Path, default=Path("outputs/aidc_solar_wind_bess_reliability"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    inputs = default_inputs_from_config(config, args.scale_case)

    if args.solar_mw is not None:
        inputs.solar_mw = args.solar_mw
    if args.wind_mw is not None:
        inputs.wind_mw = args.wind_mw
    if args.bess_power_mw is not None:
        inputs.bess_power_mw = args.bess_power_mw
    if args.bess_energy_mwh is not None:
        inputs.bess_energy_mwh = args.bess_energy_mwh
    if args.critical_load_fraction is not None:
        inputs.critical_load_fraction = args.critical_load_fraction
    if args.grid_import_limit_mw is not None:
        inputs.grid_import_limit_mw = args.grid_import_limit_mw
    if args.grid_charge_limit_mw is not None:
        inputs.grid_charge_limit_mw = args.grid_charge_limit_mw
    if args.solar_capacity_factor is not None:
        inputs.solar_capacity_factor = args.solar_capacity_factor
    if args.wind_capacity_factor is not None:
        inputs.wind_capacity_factor = args.wind_capacity_factor
    if args.outage_start_hour is not None:
        inputs.outage_start_hour = args.outage_start_hour
    if args.outage_duration_hours is not None:
        inputs.outage_duration_hours = args.outage_duration_hours

    inputs.allow_grid_charging = args.allow_grid_charging
    inputs.dispatch_reserve_for_outage = not args.no_reserve
    inputs.random_seed = args.seed

    metrics = simulate(inputs)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_json(args.out_prefix.with_suffix(".json"), inputs, metrics)
    write_metrics_csv(args.out_prefix.with_suffix(".csv"), inputs, metrics)

    print(json.dumps(asdict(metrics), indent=2))


if __name__ == "__main__":
    main()
