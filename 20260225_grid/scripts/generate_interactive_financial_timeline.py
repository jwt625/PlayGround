#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import yaml


SCENARIO_MAP = {
    "legacy_grid": "legacy_grid",
    "offgrid_sofc": "offgrid_sofc",
    "offgrid_turbine": "offgrid_turbine",
    "green_microgrid": "green_microgrid",
}
SCENARIO_SEED_OFFSET = {
    "legacy_grid": 11,
    "offgrid_sofc": 23,
    "offgrid_turbine": 37,
    "green_microgrid": 53,
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tri(rng: np.random.Generator, r: Dict[str, float], n: int) -> np.ndarray:
    return rng.triangular(r["min"], r["mean"], r["max"], size=n)


def mean_of(r: Dict[str, float]) -> float:
    return float(r["mean"])


def arr_range(value: Dict[str, float], n: int, mean_only: bool, rng: np.random.Generator) -> np.ndarray:
    if mean_only:
        return np.full(n, mean_of(value), dtype=float)
    return tri(rng, value, n)


@dataclass
class Inputs:
    nameplate_mw: float
    it_fraction: np.ndarray
    utilization: np.ndarray
    cod_months: np.ndarray
    revenue_per_mw_month: np.ndarray
    grid_energy_usd_per_mwh: np.ndarray
    gas_usd_per_mmbtu: np.ndarray
    demand_charge_usd_per_kw_month: np.ndarray
    capacity_charge_usd_per_kw_month: np.ndarray
    sofc_eff: np.ndarray
    turbine_eff: np.ndarray
    sofc_forced_outage: np.ndarray
    turbine_forced_outage: np.ndarray
    sofc_capex_per_mw: np.ndarray
    turbine_capex_per_mw: np.ndarray
    solar_capex_per_mw: np.ndarray
    soec_capex_per_mw: np.ndarray
    bess_capex_per_kwh: np.ndarray
    bop_per_mw: np.ndarray
    fixed_opex_sofc_frac: np.ndarray
    fixed_opex_turbine_frac: np.ndarray
    fixed_opex_solar_frac: np.ndarray
    fixed_opex_bess_frac: np.ndarray
    var_opex_sofc_usd_per_mwh: np.ndarray
    var_opex_turbine_usd_per_mwh: np.ndarray
    compute_capex_usd_per_mw_it: np.ndarray
    compute_useful_life_years: np.ndarray
    compute_refresh_fraction: np.ndarray
    annual_compute_software_ops_fraction: np.ndarray
    annual_compute_hw_service_fraction: np.ndarray
    annual_compute_rma_repair_fraction: np.ndarray
    compute_spares_pool_fraction: np.ndarray


def gather_inputs(
    cfg: dict[str, Any],
    scale_case: str,
    design: str,
    n: int,
    mean_only: bool,
    seed: int,
    revenue_profile: str | None = None,
) -> Inputs:
    rng = np.random.default_rng(seed)
    sc = cfg["scale_cases_mw"][scale_case]
    tp = cfg["technology_parameters"]
    cap = cfg["capex_assumptions"]
    opex = cfg["opex_assumptions"]
    market = cfg["market_prices"]
    rev = cfg["revenue_and_sla"]
    comp = cfg["compute_economics"]
    sched = cfg["schedule_and_interconnection"]["cod_months_by_scenario_and_scale"][SCENARIO_MAP[design]][scale_case]

    # Revenue profile selection: CLI override > config selected profile > fallback gross_compute_revenue.
    selected_profile = revenue_profile or rev.get("selected_revenue_profile", {}).get("value")
    gross_rev_value = rev["gross_compute_revenue"]["value"]
    if selected_profile:
        profiles = rev.get("gross_compute_revenue_profiles", {})
        if selected_profile in profiles:
            gross_rev_value = profiles[selected_profile]["value"]

    return Inputs(
        nameplate_mw=mean_of(sc["facility_nameplate_mw"]),
        it_fraction=arr_range(sc["it_load_fraction_of_facility"], n, mean_only, rng),
        utilization=arr_range(cfg["workload_and_compute"]["cluster_utilization"]["value"], n, mean_only, rng),
        cod_months=arr_range(sched, n, mean_only, rng),
        revenue_per_mw_month=arr_range(gross_rev_value, n, mean_only, rng),
        grid_energy_usd_per_mwh=arr_range(market["grid_energy_price"]["value"], n, mean_only, rng),
        gas_usd_per_mmbtu=arr_range(market["delivered_natural_gas_cost"]["value"], n, mean_only, rng),
        demand_charge_usd_per_kw_month=arr_range(market["demand_charge"]["value"], n, mean_only, rng),
        capacity_charge_usd_per_kw_month=arr_range(market["capacity_charge_proxy"]["value"], n, mean_only, rng),
        sofc_eff=arr_range(tp["sofc"]["net_electrical_efficiency_lhv"]["value"], n, mean_only, rng),
        turbine_eff=arr_range(tp["turbine_aeroderivative"]["net_electrical_efficiency_lhv"]["value"], n, mean_only, rng),
        sofc_forced_outage=arr_range(tp["sofc"]["forced_outage_rate"]["value"], n, mean_only, rng),
        turbine_forced_outage=arr_range(tp["turbine_aeroderivative"]["forced_outage_rate"]["value"], n, mean_only, rng),
        sofc_capex_per_mw=arr_range(cap["generation_plant_capex"]["sofc_usd_per_mw"]["value"], n, mean_only, rng),
        turbine_capex_per_mw=arr_range(cap["generation_plant_capex"]["turbine_usd_per_mw"]["value"], n, mean_only, rng),
        solar_capex_per_mw=arr_range(cap["generation_plant_capex"]["solar_usd_per_mw_ac"]["value"], n, mean_only, rng),
        soec_capex_per_mw=arr_range(cap["generation_plant_capex"]["soec_usd_per_mw"]["value"], n, mean_only, rng),
        bess_capex_per_kwh=arr_range(cap["storage_capex"]["bess_usd_per_kwh_installed"]["value"], n, mean_only, rng),
        bop_per_mw=arr_range(cap["site_and_bop_capex"]["substation_switchyard_usd_per_mw"]["value"], n, mean_only, rng),
        fixed_opex_sofc_frac=arr_range(opex["fixed_opex_fraction_of_generation_capex"]["sofc"]["value"], n, mean_only, rng),
        fixed_opex_turbine_frac=arr_range(opex["fixed_opex_fraction_of_generation_capex"]["turbine"]["value"], n, mean_only, rng),
        fixed_opex_solar_frac=arr_range(opex["fixed_opex_fraction_of_generation_capex"]["solar"]["value"], n, mean_only, rng),
        fixed_opex_bess_frac=arr_range(opex["fixed_opex_fraction_of_generation_capex"]["bess"]["value"], n, mean_only, rng),
        var_opex_sofc_usd_per_mwh=arr_range(opex["variable_opex_excluding_fuel_usd_per_mwh"]["sofc"]["value"], n, mean_only, rng),
        var_opex_turbine_usd_per_mwh=arr_range(opex["variable_opex_excluding_fuel_usd_per_mwh"]["turbine"]["value"], n, mean_only, rng),
        compute_capex_usd_per_mw_it=arr_range(comp["compute_capex_usd_per_mw_it"]["value"], n, mean_only, rng),
        compute_useful_life_years=arr_range(comp["compute_asset_useful_life_years"]["value"], n, mean_only, rng),
        compute_refresh_fraction=arr_range(comp["compute_refresh_fraction_at_useful_life"]["value"], n, mean_only, rng),
        annual_compute_software_ops_fraction=arr_range(comp["annual_compute_software_ops_fraction"]["value"], n, mean_only, rng),
        annual_compute_hw_service_fraction=arr_range(comp["annual_compute_hw_service_fraction"]["value"], n, mean_only, rng),
        annual_compute_rma_repair_fraction=arr_range(comp["annual_compute_rma_repair_fraction"]["value"], n, mean_only, rng),
        compute_spares_pool_fraction=arr_range(comp["compute_spares_pool_fraction"]["value"], n, mean_only, rng),
    )


def simulate_design(
    cfg: dict[str, Any],
    scale_case: str,
    design: str,
    months: int,
    samples: int,
    mean_only: bool,
    seed: int,
    revenue_profile: str | None = None,
):
    n = max(samples, 1)
    inp = gather_inputs(cfg, scale_case, design, n=n, mean_only=mean_only, seed=seed, revenue_profile=revenue_profile)

    t = np.arange(months)
    cod = np.clip(np.rint(inp.cod_months).astype(int), 1, months - 1)
    hours_per_month = 730.0

    demand_mw = inp.nameplate_mw * inp.utilization
    demand_mwh_month = demand_mw * hours_per_month
    it_mw = inp.nameplate_mw * inp.it_fraction

    # capex rough composition by design
    base_bess_power_mw = inp.nameplate_mw * 0.25
    base_bess_energy_kwh = base_bess_power_mw * 2.0 * 1000.0

    if design == "legacy_grid":
        total_capex = inp.nameplate_mw * inp.bop_per_mw + base_bess_energy_kwh * inp.bess_capex_per_kwh * 0.15
        annual_fixed_opex = np.zeros(n, dtype=float)
    elif design == "offgrid_sofc":
        total_capex = (
            inp.nameplate_mw * inp.sofc_capex_per_mw
            + inp.nameplate_mw * inp.bop_per_mw
            + base_bess_energy_kwh * inp.bess_capex_per_kwh
        )
        annual_fixed_opex = inp.fixed_opex_sofc_frac * (inp.nameplate_mw * inp.sofc_capex_per_mw) + inp.fixed_opex_bess_frac * (base_bess_energy_kwh * inp.bess_capex_per_kwh)
    elif design == "offgrid_turbine":
        total_capex = (
            inp.nameplate_mw * inp.turbine_capex_per_mw
            + inp.nameplate_mw * inp.bop_per_mw
            + base_bess_energy_kwh * inp.bess_capex_per_kwh * 0.35
        )
        annual_fixed_opex = inp.fixed_opex_turbine_frac * (inp.nameplate_mw * inp.turbine_capex_per_mw)
    else:  # green_microgrid
        solar_mw = inp.nameplate_mw * 1.1
        total_capex = (
            solar_mw * inp.solar_capex_per_mw
            + inp.nameplate_mw * inp.sofc_capex_per_mw * 0.25
            + inp.nameplate_mw * inp.soec_capex_per_mw * 0.20
            + inp.nameplate_mw * inp.bop_per_mw
            + (inp.nameplate_mw * 0.5 * 4.0 * 1000.0) * inp.bess_capex_per_kwh
        )
        annual_fixed_opex = (
            inp.fixed_opex_solar_frac * (solar_mw * inp.solar_capex_per_mw)
            + inp.fixed_opex_bess_frac * ((inp.nameplate_mw * 0.5 * 4.0 * 1000.0) * inp.bess_capex_per_kwh)
            + inp.fixed_opex_sofc_frac * (inp.nameplate_mw * inp.sofc_capex_per_mw * 0.25)
        )

    # Compute CAPEX is modeled separately and is expected to be the dominant CAPEX component.
    compute_capex_total = it_mw * inp.compute_capex_usd_per_mw_it

    facility_capex_monthly = np.zeros((n, months), dtype=float)
    compute_capex_monthly = np.zeros((n, months), dtype=float)
    for i in range(n):
        build = max(cod[i], 1)
        # Slightly back-loaded facility spend.
        w = np.linspace(1.0, 1.8, build)
        w = w / w.sum()
        facility_capex_monthly[i, :build] = total_capex[i] * w

        # Compute spend is front-loaded to reflect server/network purchases ahead of go-live.
        c_build = max(int(np.ceil(build * 0.75)), 1)
        cw = np.linspace(1.4, 1.0, c_build)
        cw = cw / cw.sum()
        compute_capex_monthly[i, :c_build] = compute_capex_total[i] * cw

        # Refresh event based on compute useful life.
        refresh_month = int(np.rint(inp.compute_useful_life_years[i] * 12))
        if refresh_month < months:
            compute_capex_monthly[i, refresh_month] += compute_capex_total[i] * inp.compute_refresh_fraction[i]

    revenue_monthly = np.zeros((n, months), dtype=float)
    infra_opex_monthly = np.zeros((n, months), dtype=float)
    compute_opex_monthly = np.zeros((n, months), dtype=float)
    rma_repair_monthly = np.zeros((n, months), dtype=float)

    for i in range(n):
        in_service = t >= cod[i]
        availability = 1.0
        if design == "offgrid_sofc":
            availability -= inp.sofc_forced_outage[i]
        elif design == "offgrid_turbine":
            availability -= inp.turbine_forced_outage[i]
        elif design == "green_microgrid":
            availability -= inp.sofc_forced_outage[i] * 0.4

        revenue_monthly[i, in_service] = it_mw[i] * inp.revenue_per_mw_month[i] * max(availability, 0.0)

        fixed_opex_m = annual_fixed_opex[i] / 12.0
        if design == "legacy_grid":
            energy = demand_mwh_month[i] * inp.grid_energy_usd_per_mwh[i]
            demand = demand_mw[i] * 1000.0 * inp.demand_charge_usd_per_kw_month[i]
            cap = demand_mw[i] * 1000.0 * inp.capacity_charge_usd_per_kw_month[i]
            variable = energy + demand + cap
        elif design == "offgrid_sofc":
            fuel_mmbtu = demand_mwh_month[i] * 3.412 / max(inp.sofc_eff[i], 1e-3)
            fuel = fuel_mmbtu * inp.gas_usd_per_mmbtu[i]
            variable = fuel + demand_mwh_month[i] * inp.var_opex_sofc_usd_per_mwh[i]
        elif design == "offgrid_turbine":
            fuel_mmbtu = demand_mwh_month[i] * 3.412 / max(inp.turbine_eff[i], 1e-3)
            fuel = fuel_mmbtu * inp.gas_usd_per_mmbtu[i]
            variable = fuel + demand_mwh_month[i] * inp.var_opex_turbine_usd_per_mwh[i]
        else:
            # renewable-heavy with residual imports + some fuel
            residual_grid_mwh = demand_mwh_month[i] * 0.25
            fuel_mmbtu = demand_mwh_month[i] * 0.10 * 3.412 / max(inp.sofc_eff[i], 1e-3)
            variable = residual_grid_mwh * inp.grid_energy_usd_per_mwh[i] + fuel_mmbtu * inp.gas_usd_per_mmbtu[i]

        infra_opex_monthly[i, in_service] = fixed_opex_m + variable

        # Compute operations and maintenance.
        compute_ops_m = (inp.annual_compute_software_ops_fraction[i] + inp.annual_compute_hw_service_fraction[i]) * compute_capex_total[i] / 12.0
        compute_opex_monthly[i, in_service] = compute_ops_m

        # RMA/repair reserve; includes spare pool carrying cost as a simple multiplier.
        repair_m = (inp.annual_compute_rma_repair_fraction[i] * (1.0 + 0.5 * inp.compute_spares_pool_fraction[i])) * compute_capex_total[i] / 12.0
        rma_repair_monthly[i, in_service] = repair_m

    total_capex_monthly = facility_capex_monthly + compute_capex_monthly
    total_opex_monthly = infra_opex_monthly + compute_opex_monthly + rma_repair_monthly
    net_monthly = revenue_monthly - total_opex_monthly - total_capex_monthly
    cum_cash = np.cumsum(net_monthly, axis=1)

    def pct(a: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "p10": np.percentile(a, 10, axis=0),
            "p50": np.percentile(a, 50, axis=0),
            "p90": np.percentile(a, 90, axis=0),
            "mean_mc": np.mean(a, axis=0),
        }

    stats = {
        "months": t,
        "net": pct(np.cumsum(net_monthly, axis=1)),
        "revenue": pct(np.cumsum(revenue_monthly, axis=1)),
        "capex_total": pct(np.cumsum(total_capex_monthly, axis=1)),
        "capex_compute": pct(np.cumsum(compute_capex_monthly, axis=1)),
        "opex_total": pct(np.cumsum(total_opex_monthly, axis=1)),
        "opex_compute": pct(np.cumsum(compute_opex_monthly, axis=1)),
        "rma_repair": pct(np.cumsum(rma_repair_monthly, axis=1)),
    }
    if mean_only:
        for k in ["net", "revenue", "capex_total", "capex_compute", "opex_total", "opex_compute", "rma_repair"]:
            stats[k]["mean_param"] = stats[k]["mean_mc"]
    return stats


def build_figures(results: dict[str, dict[str, np.ndarray]], scale_case: str, months: int) -> tuple[go.Figure, go.Figure]:
    fig = go.Figure()
    fig_components = go.Figure()
    scenario_names = list(results.keys())

    colors = {
        "legacy_grid": "#1f77b4",
        "offgrid_sofc": "#2ca02c",
        "offgrid_turbine": "#d62728",
        "green_microgrid": "#ff7f0e",
    }
    band_colors = {
        "legacy_grid": "rgba(31,119,180,0.14)",
        "offgrid_sofc": "rgba(44,160,44,0.14)",
        "offgrid_turbine": "rgba(214,39,40,0.14)",
        "green_microgrid": "rgba(255,127,14,0.14)",
    }

    for idx, name in enumerate(scenario_names):
        r = results[name]
        x = r["months"]
        visible = True
        line_color = colors.get(name, "#333")
        band_fill = band_colors.get(name, "rgba(120,120,120,0.18)")

        fig.add_trace(
            go.Scatter(
                x=x,
                y=r["net"]["p90"],
                mode="lines",
                line=dict(width=0),
                name=f"{name} P90",
                showlegend=False,
                visible=visible,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=r["net"]["p10"],
                mode="lines",
                fill="tonexty",
                fillcolor=band_fill,
                line=dict(width=0),
                name=f"{name} P10-P90",
                visible=visible,
                hovertemplate="Month %{x}<br>P10: %{y:$,.0f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=r["net"]["mean_param"],
                mode="lines",
                line=dict(width=5, color=line_color),
                name=f"{name} mean-parameter line",
                visible=visible,
                customdata=np.stack([r["net"]["p10"], r["net"]["p50"], r["net"]["p90"], r["net"]["mean_mc"]], axis=1),
                hovertemplate=(
                    "Month %{x}<br>Mean Param: %{y:$,.0f}<br>"
                    "P10: %{customdata[0]:$,.0f}<br>"
                    "P50: %{customdata[1]:$,.0f}<br>"
                    "P90: %{customdata[2]:$,.0f}<br>"
                    "MC Mean: %{customdata[3]:$,.0f}<extra></extra>"
                ),
            )
        )

        # Component comparison curves (all shown simultaneously for selected scenario)
        comp_series = [
            ("Revenue (cum)", "revenue", "#2ca02c", 3),
            ("CAPEX Total (cum)", "capex_total", "#1f77b4", 3),
            ("CAPEX Compute (cum)", "capex_compute", "#003f8c", 2),
            ("OPEX Total (cum)", "opex_total", "#ff7f0e", 3),
            ("OPEX Compute (cum)", "opex_compute", "#cc7000", 2),
            ("RMA/Repair (cum)", "rma_repair", "#d62728", 3),
            ("Net Cash (cum)", "net", "#111111", 5),
        ]
        for label, key, c, w in comp_series:
            fig_components.add_trace(
                go.Scatter(
                    x=x,
                    y=r[key]["mean_param"],
                    mode="lines",
                    name=f"{name} {label}",
                    visible=visible,
                    line=dict(color=c, width=w),
                    customdata=np.stack([r[key]["p10"], r[key]["p50"], r[key]["p90"]], axis=1),
                    hovertemplate=(
                        "Month %{x}<br>"
                        + f"{label}: "
                        + "%{y:$,.0f}<br>P10: %{customdata[0]:$,.0f}<br>P50: %{customdata[1]:$,.0f}<br>P90: %{customdata[2]:$,.0f}<extra></extra>"
                    ),
                )
            )

    traces_per_scenario = 3
    comp_traces_per_scenario = 7
    total_comp = comp_traces_per_scenario * len(scenario_names)

    fig.update_layout(
        title=f"Cumulative Cashflow Timeline (all scenarios, {scale_case})",
        xaxis_title="Month",
        yaxis_title="Cumulative Cashflow (USD)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=20, t=90, b=50),
    )

    # Build scenario buttons for component figure.
    comp_buttons = []
    for idx, name in enumerate(scenario_names):
        vis_comp = [False] * total_comp
        cstart = idx * comp_traces_per_scenario
        vis_comp[cstart : cstart + comp_traces_per_scenario] = [True] * comp_traces_per_scenario
        comp_buttons.append(
            dict(
                label=name,
                method="update",
                args=[{"visible": vis_comp}, {"title": f"Cumulative Component Curves ({name}, {scale_case})"}],
            )
        )

    fig_components.update_layout(
        title=f"Cumulative Component Curves (legacy_grid, {scale_case})",
        xaxis_title="Month",
        yaxis_title="USD (cumulative)",
        template="plotly_white",
        hovermode="x unified",
        updatemenus=[dict(buttons=comp_buttons, direction="down", x=0.01, y=1.18, xanchor="left", yanchor="top")],
        margin=dict(l=60, r=20, t=90, b=50),
    )
    return fig, fig_components


def build_assumption_summary(cfg: dict[str, Any], scale_case: str, months: int, samples: int, revenue_profile: str | None) -> dict[str, Any]:
    rev_cfg = cfg["revenue_and_sla"]
    selected_profile = revenue_profile or rev_cfg.get("selected_revenue_profile", {}).get("value")
    selected_range = rev_cfg["gross_compute_revenue"]["value"]
    if selected_profile and selected_profile in rev_cfg.get("gross_compute_revenue_profiles", {}):
        selected_range = rev_cfg["gross_compute_revenue_profiles"][selected_profile]["value"]

    return {
        "meta": cfg.get("meta", {}),
        "run": {
            "scale_case": scale_case,
            "months": months,
            "monte_carlo_samples": samples,
            "scenarios": list(SCENARIO_MAP.keys()),
            "revenue_profile": selected_profile or "default_gross_compute_revenue",
        },
        "key_ranges": {
            "gross_compute_revenue_selected": selected_range,
            "gross_compute_revenue_profiles": rev_cfg.get("gross_compute_revenue_profiles", {}),
            "grid_energy_price": cfg["market_prices"]["grid_energy_price"]["value"],
            "delivered_natural_gas_cost": cfg["market_prices"]["delivered_natural_gas_cost"]["value"],
            "sofc_capex_per_mw": cfg["capex_assumptions"]["generation_plant_capex"]["sofc_usd_per_mw"]["value"],
            "turbine_capex_per_mw": cfg["capex_assumptions"]["generation_plant_capex"]["turbine_usd_per_mw"]["value"],
            "bess_capex_per_kwh": cfg["capex_assumptions"]["storage_capex"]["bess_usd_per_kwh_installed"]["value"],
            "compute_capex_usd_per_mw_it": cfg["compute_economics"]["compute_capex_usd_per_mw_it"]["value"],
            "compute_asset_useful_life_years": cfg["compute_economics"]["compute_asset_useful_life_years"]["value"],
            "compute_refresh_fraction_at_useful_life": cfg["compute_economics"]["compute_refresh_fraction_at_useful_life"]["value"],
            "annual_compute_software_ops_fraction": cfg["compute_economics"]["annual_compute_software_ops_fraction"]["value"],
            "annual_compute_hw_service_fraction": cfg["compute_economics"]["annual_compute_hw_service_fraction"]["value"],
            "annual_compute_rma_repair_fraction": cfg["compute_economics"]["annual_compute_rma_repair_fraction"]["value"],
            "compute_spares_pool_fraction": cfg["compute_economics"]["compute_spares_pool_fraction"]["value"],
        },
    }


def write_html(fig_cashflow: go.Figure, fig_components: go.Figure, summary: dict[str, Any], out_path: Path) -> None:
    fig_cashflow_html = fig_cashflow.to_html(full_html=False, include_plotlyjs="cdn")
    fig_components_html = fig_components.to_html(full_html=False, include_plotlyjs=False)
    summary_json = json.dumps(summary, indent=2, default=str)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Interactive Financial Timeline</title>
  <style>
    body {{ font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; background:#f7f9fb; }}
    .container {{ max-width: 1280px; margin: 0 auto; padding: 20px; }}
    .panel {{ background: white; border: 1px solid #d9e1ea; border-radius: 10px; padding: 14px; margin-top: 12px; }}
    .controls {{ display:flex; gap:10px; align-items:center; margin-bottom: 10px; }}
    button {{ border:1px solid #b7c4d3; background:#fff; border-radius:8px; padding:8px 12px; cursor:pointer; }}
    pre {{ white-space: pre-wrap; max-height: 380px; overflow: auto; background:#10151b; color:#e9f0f7; padding: 12px; border-radius:8px; }}
    .muted {{ color:#4f6275; font-size: 13px; }}
  </style>
</head>
<body>
  <div class=\"container\">
    <div class=\"panel\">
      <div class=\"controls\">
        <button id=\"toggleAssumptions\">Toggle Params and Assumptions</button>
        <span class=\"muted\">Band = P10–P90 from sampled ranges; thick line = mean-parameter run.</span>
      </div>
      {fig_cashflow_html}
    </div>
    <div class=\"panel\">
      {fig_components_html}
    </div>
    <div id=\"assumptions\" class=\"panel\" style=\"display:none\">
      <h3 style=\"margin-top:0\">Parameters and Assumptions Used</h3>
      <pre>{summary_json}</pre>
    </div>
  </div>
  <script>
    const btn = document.getElementById('toggleAssumptions');
    const panel = document.getElementById('assumptions');
    btn.addEventListener('click', () => {{
      panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    }});
  </script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interactive financial timeline HTML")
    parser.add_argument("--config", default="config/assumptions_2026_us_ai_datacenter.yaml")
    parser.add_argument("--scale", default="regional_100", choices=["edge_10", "regional_100", "hyperscale_1000"])
    parser.add_argument("--months", type=int, default=120)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revenue-profile", default=None, help="Optional revenue profile override from config revenue_and_sla.gross_compute_revenue_profiles")
    parser.add_argument("--out", default="outputs/interactive_financial_timeline.html")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))

    results: dict[str, dict[str, np.ndarray]] = {}
    for design in SCENARIO_MAP:
        results[design] = simulate_design(
            cfg,
            args.scale,
            design,
            args.months,
            args.samples,
            mean_only=False,
            seed=args.seed + SCENARIO_SEED_OFFSET[design],
            revenue_profile=args.revenue_profile,
        )
        mean_stats = simulate_design(
            cfg,
            args.scale,
            design,
            args.months,
            1,
            mean_only=True,
            seed=args.seed,
            revenue_profile=args.revenue_profile,
        )
        for k in ["net", "revenue", "capex_total", "capex_compute", "opex_total", "opex_compute", "rma_repair"]:
            results[design][k]["mean_param"] = mean_stats[k]["mean_param"]

    fig_cashflow, fig_components = build_figures(results, args.scale, args.months)
    summary = build_assumption_summary(cfg, args.scale, args.months, args.samples, args.revenue_profile)
    write_html(fig_cashflow, fig_components, summary, Path(args.out))

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
