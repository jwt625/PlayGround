---
title: Sub-technology Parameter Range Notes for Modeling
as_of_date: 2026-03-02
purpose: Traceable basis for min/mean/max assumptions in config/assumptions_2026_us_ai_datacenter.yaml
---

## SOFC / MCFC
- NETL Fuel Cell Handbook (7th ed.) reports SOFC efficiency from ~40% (small/simple cycle) to 50%+ (hybrid), with 60%+ potential in hybrid systems.
- NETL handbook reports MCFC system efficiency in the high 40s to low 50s, and high operating temperature/waste-heat integration can push system efficiency to high 50s/low 60s.
- NETL handbook explicitly notes MCFC slow startup and that large stationary/marine use-cases tolerate this.
- Bloom 2024 press release reports ~60% electrical efficiency on 100% H2 and up to ~90% CHP efficiency for high-temp heat utilization.

Captured files:
- `references/md/subtech/sofc__netl_fuel_cell_handbook7.md`
- `references/md/subtech/sofc__bloom_hydrogen_sofc_efficiency.md`
- `references/md/subtech/sofc__nrel_sam_fuel_cell.md`

## Turbines
- GE LM6000 Fast Start page specifies configurable startup schedules of 5, 8, or 10 minutes to full load (with life-impact caveat at 5-minute case).
- Mitsubishi M501J page lists simple-cycle efficiency around 44% LHV and combined-cycle efficiency >64% LHV; also includes high ramp values (e.g., 42 MW/min entry).
- Siemens Energy Q1 FY26 documents show strong turbine order pressure (102 turbines, 13 GW booked, 80 GW commitments), used as lead-time/risk proxy.

Captured files:
- `references/md/subtech/turbine__ge_lm6000_fast_start.md`
- `references/md/subtech/turbine__mhi_m501j_specs.md`
- `references/md/subtech/turbine__siemens_q1_fy26_analyst_pdf.md`
- `references/md/subtech/turbine__siemens_q1_fy26_earnings_pdf.md`

## BESS
- NREL 2025 utility-scale battery update abstract provides low/mid/high capex projections for 4-hour systems: 2035 = $152/$247/$349 per kWh; 2050 = $111/$184/$333 per kWh.
- NREL ATB utility-scale battery page indicates representative 4-hour capacity-factor logic (~16.7% at one cycle/day) and round-trip efficiency assumption around 85%.
- Fluence Q3 2025 release provides execution-risk signal: slower-than-expected U.S. production ramp and backlog/contract levels.
- Tesla references provide deployment/manufacturing scale proxies (Q4/FY deployments and megafactory output scale).

Captured files:
- `references/md/subtech/bess__nrel_battery_cost_proj_2025.md`
- `references/md/subtech/bess__nrel_atb_utility_battery_2024.md`
- `references/md/subtech/bess__fluence_q3_2025.md`
- `references/md/subtech/bess__tesla_q4_2025_deployments.md`
- `references/md/subtech/bess__tesla_megafactory.md`

## Solar
- NREL ATB utility-scale PV page gives capacity-factor class range around 21.4% to 34.0% (U.S. resource bins).
- NREL ATB utility-scale PV page describes degradation assumptions ranging roughly 0.2%/yr to 0.7%/yr depending on scenario.
- First Solar FY24/FY25-guidance release gives production/shipments guidance and booking scale (used for supply availability proxies).
- Canadian Solar 20-F provides module shipment and storage-manufacturing context for supplier diversity and scale checks.

Captured files:
- `references/md/subtech/solar__nrel_atb_utility_pv_2024.md`
- `references/md/subtech/solar__first_solar_2024_results_2025_guidance.md`
- `references/md/subtech/solar__canadian_solar_20f.md`

## Grid / Pricing Context
- EIA STEO and Electricity Monthly pages used for natural gas and electricity price range calibration.
- PJM BRA release used as an additional stress-price signal for capacity-tight regions.

Captured files:
- `references/md/subtech/grid__eia_steo.md`
- `references/md/subtech/grid__eia_electricity_monthly.md`
- `references/md/subtech/grid__pjm_2027_2028_bra.md`

## Manifests
- `references/manifest.subtech.json`
- `references/manifest.subtech.additional.json`
- `references/manifest.subtech.retries.json`
- `references/manifest.subtech.playwright_retries.json`
