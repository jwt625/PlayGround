#!/usr/bin/env python3
"""Fetch and cache raw overlay data for the interactive reliability map.

Downloads:
  1. EIA Natural Gas Pipelines (ArcGIS REST → GeoJSON)
  2. EPA Nonattainment Zones (ArcGIS REST → GeoJSON)
  3. PeeringDB IXP Facilities (REST API → JSON)

All raw downloads go under references/data/overlays/.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAW_DIR = Path("references/data/overlays")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_json(url: str, timeout: int = 120, retries: int = 3) -> dict | list:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AIDC-Overlay-Fetcher/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                wait = 5 * (attempt + 1)
                print(f"  Retry {attempt + 1}/{retries} after error: {exc} (waiting {wait}s)")
                time.sleep(wait)
    raise RuntimeError(f"Failed after {retries + 1} attempts: {last_error}")


def fetch_arcgis_geojson_paged(
    base_url: str,
    where: str = "1=1",
    out_fields: str = "*",
    max_record_count: int = 2000,
) -> dict:
    """Page through an ArcGIS REST FeatureServer/MapServer and collect all features as GeoJSON."""
    all_features: list[dict] = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": out_fields,
            "f": "geojson",
            "resultOffset": str(offset),
            "resultRecordCount": str(max_record_count),
        }
        url = base_url + "/query?" + urllib.parse.urlencode(params)
        print(f"  Fetching offset={offset} …")
        data = fetch_json(url, timeout=180)
        features = data.get("features", [])
        if not features:
            break
        all_features.extend(features)
        print(f"    got {len(features)} features (total {len(all_features)})")
        if len(features) < max_record_count:
            break
        offset += len(features)
    return {
        "type": "FeatureCollection",
        "features": all_features,
    }


# ---------------------------------------------------------------------------
# 1. EIA Natural Gas Pipelines
# ---------------------------------------------------------------------------

PIPELINE_URL = (
    "https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services"
    "/Natural_Gas_Pipelines___Copy_shp/FeatureServer/0"
)


def fetch_eia_pipelines() -> Path:
    out = RAW_DIR / "eia_natural_gas_pipelines_raw.geojson"
    if out.exists():
        size_mb = out.stat().st_size / 1e6
        print(f"[pipelines] Already cached ({size_mb:.1f} MB): {out}")
        return out
    print("[pipelines] Fetching HIFLD Natural Gas Pipelines …")
    geojson = fetch_arcgis_geojson_paged(PIPELINE_URL)
    out.write_text(json.dumps(geojson), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"[pipelines] Saved {len(geojson['features'])} features ({size_mb:.1f} MB): {out}")
    return out


# ---------------------------------------------------------------------------
# 2. EPA Nonattainment Zones
# ---------------------------------------------------------------------------

EPA_MAPSERVER_BASE = (
    "https://gispub.epa.gov/arcgis/rest/services/OAR_OAQPS/NonattainmentAreas/MapServer"
)

EPA_LAYERS = {
    "ozone_8hr_2015": 2,
    "pm25_annual_2012": 7,
}


def fetch_epa_nonattainment() -> list[Path]:
    paths = []
    for name, layer_id in EPA_LAYERS.items():
        out = RAW_DIR / f"epa_nonattainment_{name}_raw.geojson"
        if out.exists():
            size_mb = out.stat().st_size / 1e6
            print(f"[epa/{name}] Already cached ({size_mb:.1f} MB): {out}")
            paths.append(out)
            continue
        url = f"{EPA_MAPSERVER_BASE}/{layer_id}"
        print(f"[epa/{name}] Fetching EPA layer {layer_id} …")
        geojson = fetch_arcgis_geojson_paged(url)
        out.write_text(json.dumps(geojson), encoding="utf-8")
        size_mb = out.stat().st_size / 1e6
        print(f"[epa/{name}] Saved {len(geojson['features'])} features ({size_mb:.1f} MB): {out}")
        paths.append(out)
    return paths


# ---------------------------------------------------------------------------
# 3. PeeringDB IXP Facilities
# ---------------------------------------------------------------------------

PEERINGDB_FAC_URL = "https://www.peeringdb.com/api/fac?country=US&status=ok"


def fetch_peeringdb_facilities() -> Path:
    out = RAW_DIR / "peeringdb_facilities_us.json"
    if out.exists():
        size_mb = out.stat().st_size / 1e6
        print(f"[peeringdb] Already cached ({size_mb:.1f} MB): {out}")
        return out
    print("[peeringdb] Fetching PeeringDB US facilities …")
    data = fetch_json(PEERINGDB_FAC_URL, timeout=60)
    facilities = data.get("data", [])
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"[peeringdb] Saved {len(facilities)} facilities ({size_mb:.1f} MB): {out}")
    return out


# ---------------------------------------------------------------------------
# 4. Fiber Route Datasets (real carrier networks)
# ---------------------------------------------------------------------------

FIBER_ROUTE_SOURCES = {
    "uniti_national_core": {
        "url": "https://services.arcgis.com/QO7AhQX53d0m9M2s/arcgis/rest/services/Uniti_Wholesale_Map_Demo_4_WFL1/FeatureServer/70",
        "description": "Uniti/Windstream National Core Network",
        "filename": "fiber_uniti_national_core_raw.geojson",
    },
    "segra_consolidated": {
        "url": "https://services2.arcgis.com/idZndEogp4LoOmcv/arcgis/rest/services/SegraFiberNetwork_Consolidated/FeatureServer/167",
        "description": "Segra Fiber Network (East Coast GA-NY)",
        "filename": "fiber_segra_consolidated_raw.geojson",
    },
    "nwpa_geotel": {
        "url": "https://services7.arcgis.com/6h8O0NpKpUyHiCti/arcgis/rest/services/AGOL_Publish4/FeatureServer/13",
        "description": "NWPA/GeoTel Multi-Carrier Fiber (PA+OH)",
        "filename": "fiber_nwpa_geotel_raw.geojson",
        "max_record_count": 1000,
    },
}


def fetch_fiber_routes() -> list[Path]:
    paths = []
    for key, src in FIBER_ROUTE_SOURCES.items():
        out = RAW_DIR / src["filename"]
        if out.exists():
            size_mb = out.stat().st_size / 1e6
            print(f"[fiber/{key}] Already cached ({size_mb:.1f} MB): {out}")
            paths.append(out)
            continue
        print(f"[fiber/{key}] Fetching {src['description']} …")
        max_rc = src.get("max_record_count", 2000)
        geojson = fetch_arcgis_geojson_paged(src["url"], max_record_count=max_rc)
        out.write_text(json.dumps(geojson), encoding="utf-8")
        size_mb = out.stat().st_size / 1e6
        print(f"[fiber/{key}] Saved {len(geojson['features'])} features ({size_mb:.1f} MB): {out}")
        paths.append(out)
    return paths


# ---------------------------------------------------------------------------
# 5. Submarine Cable Routes (TeleGeography)
# ---------------------------------------------------------------------------

SUBMARINE_CABLE_URL = "https://www.submarinecablemap.com/api/v3/cable/cable-geo.json"


def fetch_submarine_cables() -> Path:
    out = RAW_DIR / "submarine_cables_raw.geojson"
    if out.exists():
        size_mb = out.stat().st_size / 1e6
        print(f"[submarine] Already cached ({size_mb:.1f} MB): {out}")
        return out
    print("[submarine] Fetching TeleGeography submarine cable routes …")
    data = fetch_json(SUBMARINE_CABLE_URL, timeout=120)
    out.write_text(json.dumps(data), encoding="utf-8")
    features = data.get("features", [])
    size_mb = out.stat().st_size / 1e6
    print(f"[submarine] Saved {len(features)} cable routes ({size_mb:.1f} MB): {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Overlay Data Fetcher")
    print("=" * 60)

    errors: list[str] = []

    # 1. Pipelines
    try:
        fetch_eia_pipelines()
    except Exception as exc:
        print(f"[pipelines] ERROR: {exc}")
        errors.append(f"pipelines: {exc}")

    # 2. EPA
    try:
        fetch_epa_nonattainment()
    except Exception as exc:
        print(f"[epa] ERROR: {exc}")
        errors.append(f"epa: {exc}")

    # 3. PeeringDB
    try:
        fetch_peeringdb_facilities()
    except Exception as exc:
        print(f"[peeringdb] ERROR: {exc}")
        errors.append(f"peeringdb: {exc}")

    # 4. Fiber route datasets
    try:
        fetch_fiber_routes()
    except Exception as exc:
        print(f"[fiber-routes] ERROR: {exc}")
        errors.append(f"fiber-routes: {exc}")

    # 5. Submarine cables
    try:
        fetch_submarine_cables()
    except Exception as exc:
        print(f"[submarine] ERROR: {exc}")
        errors.append(f"submarine: {exc}")

    print()
    if errors:
        print(f"Completed with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All overlay data fetched successfully.")


if __name__ == "__main__":
    main()
