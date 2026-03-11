---
key: dataset__wind_integration_national_dataset_toolkit
url: https://www.nrel.gov/hpc/wind-dataset
final_url: https://www.nrel.gov/hpc/wind-dataset
retrieved_at_utc: 2026-03-11T07:13:21Z
source_type: html
source_method: manual_curl
http_status: 200
raw_path: references/raw/solar_wind/wind_integration_national_dataset_toolkit.html
---

# Wind Integration National Dataset Toolkit

## What it provides

- A national wind-resource dataset intended for high-performance access.
- NREL-hosted documentation and citation guidance for the WIND Toolkit.
- A starting point for retrieving wind speed time series for candidate sites.

## Fields relevant to the model

- Hourly wind resource time series.
- Site-specific resource variation across seasons and years.
- Inputs needed to convert hub-height wind speed into turbine power with a power curve.

## How to use it here

- Map each candidate AIDC site to the nearest wind grid point.
- Use the wind time series with an assumed turbine class, hub height, and power curve.
- Build annual and multi-year wind production traces, then combine them with PV and storage dispatch.

## Limitation for this use case

- This source provides resource data, not turbine availability, wake losses, icing losses, or site-specific curtailment. Those need explicit model parameters.
