#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


GEONAMES_US_URL = "https://download.geonames.org/export/dump/US.zip"
GEONAMES_US_ZIP = Path("references/raw/geo/geonames_us.zip")
GEONAMES_US_TXT = "US.txt"

# Lower 48 + DC
EXCLUDED_ADMIN1 = {"AK", "HI", "PR", "GU", "VI", "MP", "AS"}


def ensure_geonames_zip() -> Path:
    GEONAMES_US_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not GEONAMES_US_ZIP.exists():
        urlretrieve(GEONAMES_US_URL, GEONAMES_US_ZIP)
    return GEONAMES_US_ZIP


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))


def load_us_populated_places() -> list[dict[str, object]]:
    zip_path = ensure_geonames_zip()
    towns: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(GEONAMES_US_TXT) as handle:
            for raw_line in handle:
                parts = raw_line.decode("utf-8").rstrip("\n").split("\t")
                if len(parts) < 19:
                    continue
                feature_class = parts[6]
                feature_code = parts[7]
                admin1 = parts[10]
                if feature_class != "P":
                    continue
                if not feature_code.startswith("PPL"):
                    continue
                if admin1 in EXCLUDED_ADMIN1:
                    continue
                population = int(parts[14] or "0")
                towns.append(
                    {
                        "geonameid": int(parts[0]),
                        "name": parts[1],
                        "asciiname": parts[2],
                        "lat": float(parts[4]),
                        "lon": float(parts[5]),
                        "feature_code": feature_code,
                        "state": admin1,
                        "county": parts[11],
                        "population": population,
                        "timezone": parts[17],
                    }
                )
    return towns


def dedup_by_distance(
    towns: list[dict[str, object]],
    min_separation_km: float,
) -> list[dict[str, object]]:
    # Keep larger population centers first, then greedily reject nearby points.
    sorted_towns = sorted(
        towns,
        key=lambda row: (int(row["population"]), str(row["feature_code"]).startswith("PPLA"), str(row["name"])),
        reverse=True,
    )

    kept: list[dict[str, object]] = []
    cell_size_deg = max(min_separation_km / 111.0, 0.01)
    buckets: dict[tuple[int, int], list[int]] = {}

    for town in sorted_towns:
        lat = float(town["lat"])
        lon = float(town["lon"])
        lat_bucket = int(math.floor(lat / cell_size_deg))
        lon_bucket = int(math.floor(lon / cell_size_deg))

        too_close = False
        for dlat in (-1, 0, 1):
            for dlon in (-1, 0, 1):
                key = (lat_bucket + dlat, lon_bucket + dlon)
                for kept_index in buckets.get(key, []):
                    other = kept[kept_index]
                    if haversine_km(lat, lon, float(other["lat"]), float(other["lon"])) < min_separation_km:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break

        if too_close:
            continue

        kept.append(town)
        buckets.setdefault((lat_bucket, lon_bucket), []).append(len(kept) - 1)

    return kept


def write_json(path: Path, points: list[dict[str, object]], min_separation_km: float) -> None:
    payload = {
        "meta": {
            "source": GEONAMES_US_URL,
            "feature_class": "P",
            "feature_code_filter": "PPL*",
            "excluded_admin1": sorted(EXCLUDED_ADMIN1),
            "min_separation_km": min_separation_km,
            "count": len(points),
        },
        "points": points,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path: Path, points: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["geonameid", "name", "asciiname", "state", "county", "lat", "lon", "population", "feature_code", "timezone"],
        )
        writer.writeheader()
        writer.writerows(points)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lower-48 U.S. town points from GeoNames and dedup nearby points.")
    parser.add_argument("--min-separation-km", type=float, default=100.0)
    parser.add_argument("--json-out", type=Path, default=Path("outputs/us_towns_dedup.json"))
    parser.add_argument("--csv-out", type=Path, default=Path("outputs/us_towns_dedup.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    towns = load_us_populated_places()
    deduped = dedup_by_distance(towns, args.min_separation_km)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.json_out, deduped, args.min_separation_km)
    write_csv(args.csv_out, deduped)
    print(json.dumps({"raw": len(towns), "deduped": len(deduped), "json_out": str(args.json_out), "csv_out": str(args.csv_out)}))


if __name__ == "__main__":
    main()
