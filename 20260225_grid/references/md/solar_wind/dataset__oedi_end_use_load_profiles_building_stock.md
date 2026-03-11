---
key: dataset__oedi_end_use_load_profiles_building_stock
url: https://data.openei.org/submissions/4520
final_url: https://data.openei.org/submissions/4520
retrieved_at_utc: 2026-03-11T07:13:21Z
source_type: html
source_method: manual_curl
http_status: 200
raw_path: references/raw/solar_wind/oedi_end_use_load_profiles_building_stock.html
---

# OEDI End-Use Load Profiles for the U.S. Building Stock

## What it provides

- Calibrated and validated `15-minute` load profiles.
- Residential and commercial building archetypes across U.S. climate regions.
- End-use level demand shapes derived from ResStock and ComStock models.

## How it helps

- Useful as a reference for weather sensitivity, cooling seasonality, and flexible-load logic.
- Useful if the project later needs proxy support loads for offices, warehouses, or mixed campus loads around an AIDC.

## Why it is not the main AIDC load dataset

- It is not a data-center-native dataset.
- It does not represent AI training or inference workload scheduling, UPS losses, PUE dynamics, or rack-level power density.
- For the AIDC uptime problem, the primary load should be a parameterized facility workload model rather than direct use of these profiles.
