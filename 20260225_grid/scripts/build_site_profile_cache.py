#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from resource_profiles import (
    RealResourceSettings,
    cache_path_for_site,
    load_profile_cache,
    parse_nsrdb_csv,
    parse_wtk_csv,
    save_profile_cache,
    solar_generation_profile_from_nsrdb,
    wind_generation_profile_from_wtk,
)


def load_sites(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sites: list[dict[str, Any]] = []
    for idx, point in enumerate(payload["points"]):
        site = dict(point)
        site.setdefault("site_id", idx)
        sites.append(site)
    return sites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-site derived profile cache from raw NSRDB and WTK CSVs.")
    parser.add_argument("--site-json", type=Path, default=Path("outputs/us_towns_dedup.json"))
    parser.add_argument("--raw-cache-root", type=Path, default=Path("references/data/solar_wind_api"))
    parser.add_argument("--profile-cache-root", type=Path, default=Path("references/data/solar_wind_profiles"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/site_profile_cache_summary.json"))
    parser.add_argument("--solar-year", type=int, default=2020)
    parser.add_argument("--wind-year", type=int, default=2014)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-sites", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = RealResourceSettings(solar_year=args.solar_year, wind_year=args.wind_year)
    sites = load_sites(args.site_json)
    end_index = len(sites) if args.max_sites <= 0 else min(len(sites), args.start_index + args.max_sites)
    selected = sites[args.start_index:end_index]

    summary = {
        "site_count_total": len(sites),
        "site_count_selected": len(selected),
        "start_index": args.start_index,
        "end_index": end_index,
        "solar_year": args.solar_year,
        "wind_year": args.wind_year,
        "profiles_built": 0,
        "profiles_already_present": 0,
        "sites_missing_raw_csv": 0,
        "site_ids_with_profiles": [],
    }

    for offset, site in enumerate(selected):
        site_id = int(site.get("site_id", args.start_index + offset))
        lat = float(site["lat"])
        lon = float(site["lon"])
        nsrdb_path = cache_path_for_site(args.raw_cache_root / "nsrdb", "nsrdb", lat, lon, args.solar_year)
        wtk_path = cache_path_for_site(args.raw_cache_root / "wtk", "wtk", lat, lon, args.wind_year)
        profile_path = args.profile_cache_root / f"site_{site_id:04d}.npz"
        if profile_path.exists() and not args.rebuild:
            summary["profiles_already_present"] += 1
            summary["site_ids_with_profiles"].append(site_id)
            continue
        if not nsrdb_path.exists() or not wtk_path.exists():
            summary["sites_missing_raw_csv"] += 1
            continue
        solar_meta, solar_rows = parse_nsrdb_csv(nsrdb_path.read_text(encoding="utf-8"))
        wind_meta, wind_rows = parse_wtk_csv(wtk_path.read_text(encoding="utf-8"))
        solar_profile = solar_generation_profile_from_nsrdb(solar_rows, settings)
        wind_profile = wind_generation_profile_from_wtk(wind_rows, settings)
        metadata = {
            "site_id": site_id,
            "site": {
                "name": site.get("name", ""),
                "state": site.get("state", ""),
                "lat": lat,
                "lon": lon,
            },
            "solar_metadata": solar_meta,
            "wind_metadata": wind_meta,
            "settings": settings.__dict__,
        }
        save_profile_cache(profile_path, solar_profile, wind_profile, metadata)
        # re-open once to validate the file shape
        solar_check, wind_check, _ = load_profile_cache(profile_path)
        if len(solar_check) != 8760 or len(wind_check) != 8760:
            raise RuntimeError(f"Profile cache validation failed for site_id={site_id}")
        summary["profiles_built"] += 1
        summary["site_ids_with_profiles"].append(site_id)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
