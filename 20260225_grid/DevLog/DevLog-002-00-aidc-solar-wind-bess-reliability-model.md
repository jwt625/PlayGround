# DevLog-002-00: AIDC Solar-Wind-BESS Reliability Model Plan

## Progress Status

### Implemented so far
- Deterministic hourly adequacy simulator is implemented in `scripts/simulate_solar_wind_bess_reliability.py`.
- Cached U.S. map pipeline is implemented with:
  - cache builder in `scripts/precompute_us_reliability_map.py`,
  - interactive HTML generator in `scripts/generate_interactive_reliability_map.py`.
- The current dense map now uses a town-based site set instead of a synthetic latitude/longitude mesh.
- Site selection is built from GeoNames U.S. populated places, restricted to the lower 48 + DC, then greedily deduplicated by distance.
- Current town-point spacing is calibrated to approximately a `30 x 20` map density target.
- Current dense cache uses:
  - `562` sites,
  - `1331` cached parameter combinations,
  - source site file `outputs/us_towns_dedup.json`.
- The frontend no longer performs boundary masking or FE crop filtering. The cached sites themselves are now the display set.
- Real solar and wind resource adapters are now implemented in `scripts/resource_profiles.py`.
- The cache builder now supports both:
  - `synthetic` resource mode,
  - `real` resource mode.
- A live one-site validation path is implemented in `scripts/validate_real_resource_downloads.py`.
- A resumable overnight raw-data cache warmer is implemented in `scripts/cache_real_resource_api_data.py`.
- A derived per-site profile-cache builder is implemented in `scripts/build_site_profile_cache.py`.
- Parser and conversion tests for the raw NSRDB and WIND Toolkit CSV formats are implemented in `tests/test_resource_profiles.py`.
- Cache/profile merge tests are implemented in `tests/test_map_cache_pipeline.py`.

### Current artifacts
- Town-point builder:
  - `scripts/build_us_town_points.py`
- Town-point outputs:
  - `outputs/us_towns_dedup.json`
  - `outputs/us_towns_dedup.csv`
- Dense reliability cache:
  - `outputs/us_reliability_map_dense.json`
- Dense interactive map:
  - `outputs/interactive_reliability_map_dense.html`
- Real-data validation artifact:
  - `outputs/real_resource_validation.json`
- Real-data smoke cache:
  - `outputs/us_reliability_map_real_smoke2.json`
- Real-data smoke map:
  - `outputs/interactive_reliability_map_real_smoke2.html`
- Derived profile-cache summary:
  - `outputs/site_profile_cache_summary.json`
- Partial real-data dense cache:
  - `outputs/us_reliability_map_real_partial.json`
- Raw-cache smoke summary:
  - `outputs/real_resource_cache_smoke_summary.json`
- Overnight batch progress:
  - `outputs/real_resource_cache_night1_manifest.jsonl`
  - `outputs/real_resource_cache_night1_summary.json`

### Current product behavior
- Sliders:
  - solar capacity up to `6000 MW`,
  - wind capacity up to `6000 MW`,
  - BESS energy up to `200000 MWh`.
- Cache lattice:
  - solar in `600 MW` steps,
  - wind in `600 MW` steps,
  - BESS in `20000 MWh` steps.
- UI slider granularity remains finer than the cache and is handled by interpolation in-browser.
- Reliability tooltip formatting is implemented to three decimal places.
- The method section and formula description are rendered below the map.
- The main dense HTML now supports a merged-cache mode:
  - real cached site results are used where available,
  - synthetic dense-cache values remain as fallback for sites whose real raw data has not been cached yet.

### Real-data ingestion status
- Official API probes to NSRDB and WIND Toolkit succeeded with the current `.env` key and email.
- The raw downloaded formats were validated against live data:
  - NSRDB returns:
    - metadata header rows,
    - hourly `GHI`, `DNI`, `DHI`, `Temperature`, `Wind Speed`
  - WIND Toolkit returns:
    - metadata header row,
    - hourly `wind speed at 100m`, `wind direction at 100m`, `air temperature at 100m`, `air pressure at 100m`
- The current real-data conversion layer is intentionally simple:
  - solar: `GHI` plus fixed losses and temperature derate,
  - wind: generic turbine-style power curve from `windspeed_100m` plus fixed losses.
- This is already materially better than the synthetic resource field for adequacy timing, but it is not yet a high-fidelity PV or turbine plant model.
- Raw API responses are now being transformed into derived per-site profile cache files under:
  - `references/data/solar_wind_profiles/`
- Each derived site profile stores:
  - hourly solar per-MW profile,
  - hourly wind per-MW profile,
  - source metadata from both raw datasets.
- The first full national raw-data batch for the current map definition has now completed.
- Current batch scope:
  - `562` town sites,
  - `1` NSRDB solar year (`2020`),
  - `1` WIND Toolkit wind year (`2014`).

### Real-data validation results
- Fixture-based parser/conversion tests are passing.
- Profile-cache round-trip and cache-merge tests are passing.
- Live one-site validation succeeded with full-year hourly series for both datasets.
- End-to-end smoke path succeeded:
  - raw fetch,
  - real-resource cache build,
  - interactive HTML render.
- Local derived profile-cache build succeeded for the currently cached raw files.
- Current local real-data status:
  - `561` site profiles built and usable,
  - `561 / 562` sites currently represented with real derived profiles,
  - `1` site still falls back to the synthetic dense cache in the main frontend artifact.
- The single failed site in the first national batch is:
  - `Fort Jefferson, FL`
  - NSRDB response: `No data available at the provided location`
- The main dense frontend artifact has been refreshed against the updated partial real cache, so it is now effectively real-backed everywhere except that one fallback point.
- Current blocker for full national real-data rebuild:
  - no longer the first single-year national batch,
  - next blockers are multi-year expansion, better plant-conversion fidelity, and handling the one no-data site cleanly.

## Objective
Build a location-based adequacy and reliability model for an AI data center campus powered by solar, wind, battery energy storage, and optionally the grid. The primary product is an interactive 2D U.S. reliability map with three controls:

- solar capacity,
- wind capacity,
- BESS energy capacity.

The map color at each location should represent reliability percentage for a fixed 1 GW workload. The end state is a map-driven simulator that can rank locations and size resource mixes for target reliability.

## Problem Reframing
The paper summarized in `DevLog-002-solar-wind.md` is useful for site screening, but it is not the target model. That paper studies feeder hosting capacity on one residential feeder. The target here is a behind-the-meter AIDC survival and uptime model.

The first-order question is:

`Can the site serve critical AIDC load hour by hour under a given solar + wind + BESS buildout and outage regime?`

That requires:

1. location-indexed resource traces,
2. a facility load model,
3. BESS dispatch logic,
4. outage and import assumptions,
5. reliability metrics.

## Core Modeling Structure

### State variables
- `load_mw[t]`
- `solar_mw_available[t]`
- `wind_mw_available[t]`
- `battery_soc_mwh[t]`
- `grid_available[t]`
- `grid_import_limit_mw[t]`

### Power balance
For each time step:

`served_load = solar + wind + battery_discharge + grid_import - battery_charge - curtailment`

Unserved load occurs when available supply plus permitted imports cannot meet critical demand.

### Reliability outputs
- uptime fraction
- outage hours
- expected unserved energy (`EUE`, MWh)
- loss-of-load hours (`LOLH`)
- renewable curtailment
- battery cycles
- survival duration during grid outage windows

## Required Inputs

### Site and resource inputs
- latitude / longitude
- timezone
- solar weather time series from NSRDB
- wind weather time series from WIND Toolkit
- optional multi-year traces for interannual variability

### Plant sizing inputs
- solar AC MW
- wind MW
- BESS MWh
- grid import cap MW

### AIDC load inputs
- facility nameplate MW
- IT load fraction
- utilization
- PUE
- critical load fraction
- deferrable workload fraction
- interruptible workload fraction

### Technology parameters
- solar degradation
- wind availability and losses
- BESS round-trip efficiency
- BESS usable depth of discharge
- BESS reserve floor
- BESS augmentation/degradation

### Reliability scenario inputs
- islanded vs grid-connected mode
- grid outage event distribution
- maintenance / forced outage assumptions for generation assets
- dispatch policy during normal operation and during outage operation

## Phased Implementation Plan

## Phase 1: Deterministic adequacy engine
Goal:
- Produce a runnable hourly simulator using synthetic solar and wind traces.

Scope:
- Use existing YAML assumptions for AIDC load, solar, and BESS defaults.
- Generate synthetic hourly traces from annual capacity factors plus seasonal and diurnal structure.
- Simulate battery charging/discharging against net load.
- Support both islanded and grid-connected modes.
- Emit uptime, unserved energy, and battery-cycle metrics.

Reason:
- This validates model state transitions, metric definitions, and CLI shape before adding dataset plumbing.

## Phase 2: Cached U.S. map product
Goal:
- Precompute reliability slices for a U.S. location mesh and expose them through an interactive map with three independent controls.

Scope:
- Fix workload at `1 GW`.
- Treat BESS as an energy-only slider.
- Assume BESS inverter power is not the binding constraint in V1; battery discharge can support up to the full workload while energy lasts.
- Precompute deterministic reliability on a parameter lattice:
  - solar MW
  - wind MW
  - BESS MWh
- Use a town-based lower-48 point set rather than a naive regular map mesh.
- Cache the results and render them directly in the browser.

Reason:
- This is the only practical way to get instant slider response over the full U.S. map.
- Monte Carlo should not run on slider movement.

Status:
- Implemented in synthetic-resource form.
- The current site set uses deduplicated U.S. populated places rather than polygon-filtered regular grid points.
- This resolved the earlier map coverage and masking problems from FE boundary filtering.

## Phase 3: Real weather ingestion
Goal:
- Replace synthetic traces with site-specific NSRDB and WIND traces.

Scope:
- Add data adapters for NSRDB and WIND Toolkit.
- Normalize traces to installed MW.
- Produce site-level annual summaries and joint correlation metrics.

Status:
- Adapters implemented.
- Parsers tested.
- Live validation completed.
- End-to-end smoke build completed.
- First full national raw-cache batch completed.
- Derived local per-site profile cache is now implemented and working.
- The frontend can already consume a partial real-data cache and fall back to the old dense synthetic cache for incomplete coverage.
- For the current one-year map definition, the dense map is now effectively complete.
- Next expansion should be additional weather years rather than more sites.

## Phase 4: Monte Carlo reliability
Goal:
- Estimate reliability distributions instead of only deterministic outcomes.

Scope:
- Sample weather years, grid outage windows, asset forced outages, and load perturbations.
- Estimate `P50`, `P90`, and failure probabilities for target outage thresholds.

Reason:
- Monte Carlo is valuable as an offline batch layer, not as an interactive slider-time computation.
- The recommended product architecture is:
  - deterministic cache for the interactive map,
  - optional Monte Carlo cache for a second view or uncertainty overlay.

## Phase 5: Interconnection and feeder constraints
Goal:
- Add local import/export and feeder effects for sites where the distribution system matters.

Scope:
- Use SMART-DS or similar feeders only when studying interconnection bottlenecks.
- Keep this downstream of the behind-the-meter adequacy engine.

## Parameters to keep explicit from day one
- `critical_load_fraction`
- `deferrable_workload_fraction`
- `interruptible_workload_fraction`
- `bess_reserve_fraction`
- `grid_import_limit_mw`
- `grid_outage_hours`
- `dispatch_charge_from_grid`
- `dispatch_reserve_for_outage`

These parameters materially change uptime and should not be hidden inside fixed assumptions.

## Implementation Decisions For V1

### Load model
Use:
- fixed `1 GW` workload for the map product.

For V1 map generation, load is constant in time. This matches the intended use: compare resource adequacy across locations while keeping the compute demand fixed.

### Resource model
For V1, use synthetic traces with:
- solar: strong diurnal cycle and seasonal amplitude,
- wind: weaker diurnal cycle, larger stochastic variation, optional nocturnal complementarity.

The synthetic traces are placeholders only; they let the adequacy math stabilize before real data ingestion.

### Dispatch policy
Use a simple reliability-first policy:
- serve load from solar and wind first,
- charge BESS from surplus renewables,
- import from grid if allowed and available,
- discharge BESS when renewables plus allowed imports are insufficient,
- count remaining deficit as unserved load.

Optional reserve logic:
- maintain a minimum state-of-charge floor for resilience.

### BESS interpretation for the map UI
In the map product, BESS is controlled as energy only (`MWh`).

V1 assumption:
- BESS power is not slider-controlled.
- BESS can discharge at up to the full 1 GW workload when energy is available.

This is a simplifying assumption chosen to match the user-facing product requirement. If needed later, a second advanced control can add an explicit battery power slider.

## Compute strategy decision
For the interactive map:

- do not run Monte Carlo on slider movement,
- do not recompute the full U.S. map on the fly,
- precompute deterministic slices and cache them,
- update the map by indexed lookup into the cache.

This is the correct architecture for responsiveness.

## Deliverables

### Added in this step
- this dev log
- deterministic hourly simulator script
- sample output artifact in `outputs/`
- cached-map implementation scaffold
- interactive U.S. reliability map scaffold
- town-point site builder from GeoNames
- dense map backed by deduplicated town points
- standalone U.S. boundary debug HTML used during geometry debugging

### Near-term next work
- finish warming the raw NSRDB and WIND Toolkit cache for the town-based site set
- rebuild the full dense reliability cache in `real` resource mode once enough site data is local
- upgrade the resource conversion layer:
  - better PV modeling,
  - better wind turbine assumptions
- keep the town-based site set as the dense map backbone unless a better national site sample is needed later
- add offline Monte Carlo cache only after deterministic real-weather map validation

## Clarifications
No blocking clarification is required for V1. I am proceeding with:

- hourly time steps,
- deterministic first pass,
- synthetic traces before real dataset adapters,
- reliability-first BESS dispatch,
- support for both islanded and grid-connected operation.
