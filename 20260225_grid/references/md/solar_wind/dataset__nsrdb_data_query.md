---
key: dataset__nsrdb_data_query
url: https://developer.nrel.gov/docs/solar/nsrdb/nsrdb_data_query/
final_url: https://developer.nlr.gov/docs/solar/nsrdb/nsrdb_data_query/
retrieved_at_utc: 2026-03-11T07:13:21Z
source_type: html
source_method: manual_curl
http_status: 200
raw_path: references/raw/solar_wind/nsrdb_data_query_api.html
---

# NSRDB Data Query API

## What it provides

- A location lookup endpoint for the nearest NSRDB datasets.
- Links to downstream NSRDB downloads for irradiance and weather data.
- Bulk-library pointer for the `nrel-pds-nsrdb` OpenEI bucket.

## Fields relevant to the model

- Latitude / longitude selection.
- Dataset availability by location and period.
- Access to irradiance and weather variables needed for PV simulation and temperature derating.

## How to use it here

- For each candidate AIDC location, call the query endpoint first.
- Choose an hourly dataset for multi-year reliability studies, not just a typical year.
- Feed the returned solar weather time series into `pvlib` or PVWatts-style production models.

## Important operational note

- The developer documentation page now warns users to migrate from `developer.nrel.gov` to `developer.nlr.gov` by April 30, 2026.
