---
key: dataset__smart_ds
url: https://www.nrel.gov/grid/smart-ds.html
final_url: https://www.nrel.gov/grid/smart-ds.html
retrieved_at_utc: 2026-03-11T07:13:21Z
source_type: html
source_method: manual_curl
http_status: 200
raw_path: references/raw/solar_wind/smart_ds.html
---

# SMART-DS

## What it provides

- Synthetic but realistic distribution network models and scenarios.
- Standardized feeder cases for distribution-system analysis.
- A way to test hosting-capacity and DER-interconnection questions in OpenDSS.

## How it helps

- Appropriate if the project later adds a feeder-side interconnection screen, voltage analysis, or export-limit study for an AIDC campus.
- Useful for studying whether a candidate site can interconnect large solar, wind, or storage without local upgrades.

## Why it is secondary for uptime modeling

- A behind-the-meter reliability model can be built without feeder simulation.
- Uptime under solar, wind, and BESS depends first on resource traces, dispatch policy, storage state of charge, and facility load.
- Feeder modeling becomes important when analyzing export constraints, import caps, or local voltage and thermal violations.
