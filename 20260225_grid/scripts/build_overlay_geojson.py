#!/usr/bin/env python3
"""Process raw overlay data into simplified browser-ready JSON for Plotly scattergeo.

Reads from references/data/overlays/ and writes to outputs/overlays/.

Each output is a compact JSON structure optimized for embedding in the HTML map:
  - pipelines: arrays of line segments (lat/lon pairs)
  - epa_zones: arrays of polygon boundaries (lat/lon rings)
  - fiber/ixp: arrays of point markers (lat/lon + metadata)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RAW_DIR = Path("references/data/overlays")
OUT_DIR = Path("outputs/overlays")

# ---------------------------------------------------------------------------
# Geometry simplification (Douglas-Peucker)
# ---------------------------------------------------------------------------

def _haversine_approx(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in degrees (fast, for simplification threshold)."""
    dlat = lat2 - lat1
    dlon = (lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat * dlat + dlon * dlon)


def _point_line_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """Perpendicular distance from point (px,py) to line segment (a→b), in degree-space."""
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-18:
        return _haversine_approx(px, py, ax, ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return _haversine_approx(px, py, proj_x, proj_y)


def douglas_peucker(coords: list[list[float]], tolerance: float) -> list[list[float]]:
    """Simplify a coordinate list using Douglas-Peucker algorithm."""
    if len(coords) <= 2:
        return coords
    max_dist = 0.0
    max_idx = 0
    ax, ay = coords[0][0], coords[0][1]
    bx, by = coords[-1][0], coords[-1][1]
    for i in range(1, len(coords) - 1):
        d = _point_line_distance(coords[i][0], coords[i][1], ax, ay, bx, by)
        if d > max_dist:
            max_dist = d
            max_idx = i
    if max_dist > tolerance:
        left = douglas_peucker(coords[: max_idx + 1], tolerance)
        right = douglas_peucker(coords[max_idx:], tolerance)
        return left[:-1] + right
    return [coords[0], coords[-1]]


# ---------------------------------------------------------------------------
# 1. Natural Gas Pipelines
# ---------------------------------------------------------------------------

def build_pipelines(tolerance: float = 0.02) -> Path:
    """Simplify pipeline polylines into compact line-segment arrays."""
    raw_path = RAW_DIR / "eia_natural_gas_pipelines_raw.geojson"
    out_path = OUT_DIR / "natural_gas_pipelines.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw pipeline data not found: {raw_path}")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    features = raw["features"]

    segments: list[dict] = []
    total_points_before = 0
    total_points_after = 0

    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        geom_type = geom["type"]
        props = feat.get("properties", {})

        if geom_type == "LineString":
            coord_lists = [geom["coordinates"]]
        elif geom_type == "MultiLineString":
            coord_lists = geom["coordinates"]
        else:
            continue

        for coords in coord_lists:
            total_points_before += len(coords)
            simplified = douglas_peucker(coords, tolerance)
            total_points_after += len(simplified)
            if len(simplified) < 2:
                continue
            lats = [round(c[1], 3) for c in simplified]
            lons = [round(c[0], 3) for c in simplified]
            segments.append({
                "lat": lats,
                "lon": lons,
                "name": props.get("Pipename") or props.get("PIPENAME") or props.get("Name") or "",
                "operator": props.get("Operator") or props.get("OPERATOR") or "",
                "type": props.get("TYPEPIPE") or props.get("Typepipe") or "",
            })

    payload = {
        "type": "pipelines",
        "segments": segments,
        "meta": {
            "source": "HIFLD Natural Gas Pipelines (via ArcGIS)",
            "feature_count": len(features),
            "segment_count": len(segments),
            "points_before": total_points_before,
            "points_after": total_points_after,
            "tolerance_deg": tolerance,
        },
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"[pipelines] {len(segments)} segments, {total_points_before}→{total_points_after} points, {size_mb:.1f} MB: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 2. EPA Nonattainment Zones
# ---------------------------------------------------------------------------

def _extract_polygon_rings(geom: dict, tolerance: float) -> list[dict]:
    """Extract and simplify polygon rings from a GeoJSON geometry."""
    rings = []
    geom_type = geom["type"]
    if geom_type == "Polygon":
        polygon_list = [geom["coordinates"]]
    elif geom_type == "MultiPolygon":
        polygon_list = geom["coordinates"]
    else:
        return rings

    for polygon_coords in polygon_list:
        # Only take the outer ring (index 0), skip holes for simplicity
        outer = polygon_coords[0]
        simplified = douglas_peucker(outer, tolerance)
        if len(simplified) < 4:
            continue
        lats = [round(c[1], 3) for c in simplified]
        lons = [round(c[0], 3) for c in simplified]
        rings.append({"lat": lats, "lon": lons})
    return rings


def build_epa_zones(tolerance: float = 0.01) -> Path:
    """Merge and simplify EPA nonattainment zone polygons."""
    out_path = OUT_DIR / "epa_nonattainment_zones.json"

    all_zones: list[dict] = []
    for layer_name, filename in [
        ("ozone_8hr_2015", "epa_nonattainment_ozone_8hr_2015_raw.geojson"),
        ("pm25_annual_2012", "epa_nonattainment_pm25_annual_2012_raw.geojson"),
    ]:
        raw_path = RAW_DIR / filename
        if not raw_path.exists():
            print(f"[epa] Skipping {layer_name}: {raw_path} not found")
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        features = raw.get("features", [])
        print(f"[epa/{layer_name}] Processing {len(features)} features …")

        for feat in features:
            geom = feat.get("geometry")
            if not geom:
                continue
            props = feat.get("properties", {})
            rings = _extract_polygon_rings(geom, tolerance)
            if not rings:
                continue
            # Try various property name patterns
            area_name = (
                props.get("area_name")
                or props.get("AREA_NAME")
                or props.get("AreaName")
                or props.get("name")
                or ""
            )
            classification = (
                props.get("classification")
                or props.get("CLASSIFICATION")
                or props.get("Classification")
                or ""
            )
            all_zones.append({
                "pollutant": layer_name,
                "name": area_name,
                "classification": classification,
                "rings": rings,
            })

    payload = {
        "type": "epa_nonattainment",
        "zones": all_zones,
        "meta": {
            "source": "EPA Green Book / ArcGIS MapServer",
            "layers": ["ozone_8hr_2015", "pm25_annual_2012"],
            "zone_count": len(all_zones),
            "tolerance_deg": tolerance,
        },
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"[epa] {len(all_zones)} zones, {size_mb:.1f} MB: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 3. PeeringDB / Fiber Infrastructure Points
# ---------------------------------------------------------------------------

def build_fiber_points() -> Path:
    """Extract US data center / IXP facility locations from PeeringDB."""
    raw_path = RAW_DIR / "peeringdb_facilities_us.json"
    out_path = OUT_DIR / "fiber_infrastructure.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"PeeringDB data not found: {raw_path}")

    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    facilities = raw.get("data", [])

    points: list[dict] = []
    skipped = 0
    for fac in facilities:
        lat = fac.get("latitude")
        lon = fac.get("longitude")
        if lat is None or lon is None or (lat == 0 and lon == 0):
            skipped += 1
            continue
        # Filter to CONUS
        if not (24.0 <= lat <= 50.0 and -125.0 <= lon <= -66.0):
            skipped += 1
            continue
        org_name = fac.get("org_name", "")
        if not org_name and isinstance(fac.get("org"), dict):
            org_name = fac["org"].get("name", "")
        points.append({
            "lat": round(float(lat), 4),
            "lon": round(float(lon), 4),
            "name": fac.get("name", ""),
            "city": fac.get("city", ""),
            "state": fac.get("state", ""),
            "org": org_name,
            "net_count": int(fac.get("net_count", 0)),
            "ix_count": int(fac.get("ix_count", 0)),
            "website": fac.get("website", ""),
        })

    payload = {
        "type": "fiber_infrastructure",
        "points": points,
        "meta": {
            "source": "PeeringDB US Facilities",
            "total_facilities": len(facilities),
            "conus_points": len(points),
            "skipped": skipped,
        },
    }
    out_path.write_text(json.dumps(payload, indent=None), encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"[fiber] {len(points)} CONUS facilities, {size_mb:.1f} MB: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Overlay GeoJSON Builder")
    print("=" * 60)

    build_pipelines(tolerance=0.02)
    build_epa_zones(tolerance=0.01)
    build_fiber_points()

    print()
    print("All overlay data processed.")
    total_size = sum(f.stat().st_size for f in OUT_DIR.glob("*.json"))
    print(f"Total overlay size: {total_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
