---
key: paper__birk_jones_hybrid_hosting_capacity_2025
url: https://doi.org/10.1063/5.0264545
final_url: https://pubs.aip.org/jrse/article/17/6/066304/3371059/Hosting-capacity-considerations-for-the
retrieved_at_utc: 2026-03-11T07:13:21Z
source_type: pdf
source_method: local_cached_pdf_manual_review
http_status: 200
raw_path: references/066304_1_5.0264545.pdf
---

# Hosting capacity considerations for the combination of wind and solar on distribution electric power systems subject to different levels of coincident operations

Journal of Renewable and Sustainable Energy 17, 066304 (2025), Sandia National Laboratories.

## What the paper actually does

- Computes seasonal Pearson correlation between modeled PV and wind generation across the continental U.S.
- Clusters locations using four seasonal correlation features: `R_winter`, `R_spring`, `R_summer`, `R_fall`.
- Runs quasi-static time-series hosting-capacity simulations in OpenDSS on one fixed SMART-DS residential feeder.
- Compares voltage, line loading, and transformer loading across representative correlation regimes.

## Datasets and tools named in the paper

- WIND Toolkit / NREL wind resource data, accessed through HSDS / WRDB references.
- NSRDB hourly solar and temperature data.
- OpenEI End-Use Load Profiles for U.S. Building Stock.
- SMART-DS feeder model in OpenDSS.
- `windpowerlib`, `pvlib` / PVWatts, NumPy, scikit-learn.

## Findings relevant to this repo

- Positive PV-wind correlation tends to increase thermal stress on the feeder.
- Negative correlation can reduce thermal stress by spreading generation across more hours.
- Correlation is not a sufficient predictor of voltage problems.

## What is useful for an AIDC reliability model

- The correlation-map idea is directly useful for site screening.
- The time-series workflow is reusable: weather to generation to storage dispatch to reliability metric.

## What is not transferable without changes

- The target system is a residential distribution feeder, not a large AI data center.
- The study does not model battery storage, islanding, or survival during upstream grid outages.
- The load model is generic building-stock demand, not an AIDC workload with PUE and operational constraints.
