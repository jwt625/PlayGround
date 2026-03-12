# DevLog-002-01: Interactive Map Overlay Layers

## Objective

Add toggle-able infrastructure overlay layers to the existing AIDC reliability map (`outputs/interactive_reliability_map_dense.html`). Each layer renders as an additional Plotly trace that can be shown/hidden via a checkbox in the left control panel, alongside the existing solar/wind/BESS sliders.

## Target Overlays

### Phase A — Line/Polygon Overlays

#### 1. Natural Gas Pipelines

**Why it matters:** Proximity to gas pipelines indicates fuel availability for on-site backup generation or gas turbine peakers — critical for data center resilience and grid-independent operation.

**Data source:** EIA Natural Gas Interstate and Intrastate Pipelines
- Portal: `https://atlas.eia.gov/datasets/eia::natural-gas-interstate-and-intrastate-pipelines/about`
- ArcGIS REST endpoint: `https://services7.arcgis.com/FGr1D95XCGALKXqM/arcgis/rest/services/`
- Formats available: GeoJSON, Shapefile, CSV, KML via ArcGIS REST API (`f=geojson`)
- Coverage: CONUS interstate + major intrastate pipelines
- License: U.S. government public domain
- Update frequency: monthly

**Alternative source:** HIFLD Natural Gas Pipelines
- Portal: `https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::natural-gas-pipelines/about`
- Same coverage, public domain, available via ArcGIS Hub

**Processing for web:**
- Raw GeoJSON is 50–200 MB (hundreds of thousands of vertices)
- Simplify with Mapshaper (Douglas-Peucker, ~0.01° tolerance) → target 1–3 MB
- Convert polyline features to arrays of `(lat, lon)` segments for Plotly `scattergeo` lines trace

#### 2. EPA Nonattainment Zones

**Why it matters:** Nonattainment areas trigger stricter air quality permitting for backup diesel generators. Major source permitting threshold drops from 250 tons/year (attainment, PSD) to 100 tons/year (nonattainment, NNSR). Additional requirements include emissions offsets, BACT analysis, and public notice. Nonattainment designation can delay projects 12–24 months for permitting.

**Data source:** EPA Green Book GIS Shapefiles
- GIS download: `https://www.epa.gov/green-book/green-book-gis-download`
- Formats: ESRI Shapefiles (ZIP), XLS, DBF
- Pollutants: 8-hr Ozone (2015 NAAQS), PM2.5 annual (2012 NAAQS), SO2 1-hr (2010), Lead, PM10, CO, NO2
- Coverage: polygon boundaries (finer than county-level)
- License: U.S. government public domain
- Current as of: February 28, 2026
- Individual shapefiles: 30 KB – 12.3 MB

**Alternative source:** EPA ArcGIS REST MapServer
- Endpoint: `https://gispub.epa.gov/arcgis/rest/services/OAR_OAQPS/NonattainmentAreas/MapServer`
- Layer IDs: Ozone 8-hr 2015 = 2, PM2.5 Annual 2012 = 7, SO2 1-hr 2010 = 4
- Supports `f=geojson` export directly
- Updated weekly from OAQPS database

**Processing for web:**
- Focus on Ozone 8-hr (2015) and PM2.5 Annual (2012) — these cover >90% of nonattainment designations
- Polygon data is moderate complexity; simplify to ~500 KB–1 MB GeoJSON
- Render as semi-transparent filled polygons or boundary outlines via `scattergeo`

#### 3. Fiber Backbone / Broadband Infrastructure

**Why it matters:** Data centers require high-bandwidth, low-latency connectivity. Distance from lit fiber backbone directly affects build-out cost and time-to-service.

**Data source (backbone topology):** FCC National Broadband Map (BDC)
- Portal: `https://broadbandmap.fcc.gov/data-download`
- API: `https://broadbandmap.fcc.gov/api`
- Coverage: nationwide, address-level fiber availability by provider and technology type (fiber = code 50)
- License: U.S. government public domain
- Updated biannually (8th collection window opened January 2, 2026)

**Data source (long-haul routes):** Internet Infrastructure Map
- Portal: `https://map.kmcd.dev/`
- Format: GeoJSON FeatureCollection
- Coverage: submarine cables, IXPs, backbone routes
- Sources: TeleGeography, PeeringDB

**Data source (exchange points):** PeeringDB
- API: `https://docs.peeringdb.com/api_specs/`
- Format: REST API → JSON with geographic coordinates
- Coverage: all US Internet Exchange Points, peering facilities
- License: free, CC0

**Practical approach:**
- For the map overlay, the most tractable approach is:
  1. Precompute a per-site "fiber proximity score" from FCC BDC data (offline), stored as a site attribute
  2. For a visual backbone layer, use long-haul route GeoJSON (1–3 MB simplified)
  3. Overlay IXP/peering facility point markers from PeeringDB (~0.5 MB)
- Display backbone routes as line traces and IXP points as markers

### Phase B — Additional Overlays (future)

#### 4. Electric Transmission Lines
- Source: HIFLD Electric Power Transmission Lines (`https://catalog.data.gov/dataset/electric-power-transmission-lines`)
- Format: Shapefile, 69–765 kV lines
- Processing: heavy simplification needed (~5–10 MB simplified)

#### 5. Water Stress
- Source: WRI Aqueduct 4.0 (`https://www.wri.org/applications/aqueduct/water-risk-atlas/`)
- Format: GeoTIFF raster or pre-aggregated watershed polygons
- Relevance: evaporative cooling is dominant DC cooling method

#### 6. Electricity Prices by Region
- Source: EIA-861 utility rate data + LBL ReWEP nodal pricing
- Format: CSV, state/utility level
- Processing: precompute as per-site attribute, no extra geometry

#### 7. Natural Hazard Risk
- Source: FEMA National Risk Index (county-level composite scores)
- Format: CSV/Shapefile
- Covers: earthquakes, floods, tornadoes, hurricanes

## Implementation Architecture

### Data Pipeline

```
[Public API / Shapefile download]
  → scripts/fetch_overlay_data.py        # download + cache raw data
  → scripts/build_overlay_geojson.py     # simplify, convert, filter
  → outputs/overlays/*.json              # browser-ready GeoJSON per layer
  → generate_interactive_reliability_map.py  # embed in HTML as additional Plotly traces
```

### Frontend Integration

Each overlay is an additional Plotly trace added to the `traces` array during `Plotly.newPlot()`. Visibility is controlled by checkboxes in the left panel.

**Trace types:**
- Pipelines → `scattergeo` with `mode: "lines"`, thin semi-transparent lines
- Nonattainment zones → `scattergeo` with `mode: "lines"` (boundary outlines) or `fill: "toself"` polygons
- Fiber backbone → `scattergeo` with `mode: "lines"` for routes + `mode: "markers"` for IXP points

**UI addition (in left `.panel`, after color mode dropdown):**
```html
<div class="control">
  <label><strong>Map Overlays</strong></label>
  <div class="overlay-toggles">
    <label><input type="checkbox" id="overlayPipelines"> Natural Gas Pipelines</label>
    <label><input type="checkbox" id="overlayEPA"> EPA Nonattainment</label>
    <label><input type="checkbox" id="overlayFiber"> Fiber Backbone</label>
  </div>
</div>
```

**JavaScript toggle:**
```javascript
document.getElementById("overlayPipelines").addEventListener("change", (e) => {
  Plotly.restyle("map", {visible: e.target.checked}, [PIPELINE_TRACE_INDEX]);
});
```

### Size Budget

| Layer | Raw size | Simplified | Notes |
|-------|----------|-----------|-------|
| Current HTML | — | 6.3 MB | reliability cache JSON |
| Gas pipelines | 50–200 MB | 1–3 MB | Mapshaper simplification |
| EPA nonattainment | 1–12 MB | 0.3–1 MB | Ozone + PM2.5 only |
| Fiber backbone | 10–50 MB | 1–3 MB | Long-haul routes only |
| **Total estimated** | — | **~10–13 MB** | Acceptable for local tool |

## Data Caching Plan

All raw downloads go under `references/data/overlays/` (gitignored).
Processed browser-ready files go under `outputs/overlays/` (tracked selectively).

```
references/data/overlays/
  eia_natural_gas_pipelines_raw.geojson    (16 MB, 33806 features from HIFLD)
  epa_nonattainment_ozone_8hr_2015_raw.geojson  (29 MB, 66 features from EPA)
  epa_nonattainment_pm25_annual_2012_raw.geojson (1.1 MB, 9 features from EPA)
  peeringdb_facilities_us.json             (1.7 MB, 1383 facilities from PeeringDB)

outputs/overlays/
  natural_gas_pipelines.json    (4.4 MB, simplified with Douglas-Peucker @ 0.02°)
  epa_nonattainment_zones.json  (0.1 MB, simplified with Douglas-Peucker @ 0.01°)
  fiber_infrastructure.json     (0.3 MB, 1347 CONUS points)
```

## Current Status

- [x] DevLog written
- [x] Data fetch script implemented (`scripts/fetch_overlay_data.py`)
- [x] Overlay GeoJSON processing script implemented (`scripts/build_overlay_geojson.py`)
- [x] Overlay data cached locally (raw under `references/data/overlays/`, processed under `outputs/overlays/`)
- [x] HTML generator updated with overlay traces and toggles
- [x] Dense map HTML regenerated with overlays (11.4 MB)
- [x] All overlays enabled by default
- [x] EPA nonattainment rendered as boundary outlines only (fill caused rendering artifacts)
- [x] Fullscreen button added to map panel (Escape to exit)
- [x] Enriched hover metadata: pipeline operator/type, DC/IXP operator/network count/IX count
