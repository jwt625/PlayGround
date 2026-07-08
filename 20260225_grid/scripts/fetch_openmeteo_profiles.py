#!/usr/bin/env python3
"""Fetch per-site solar+wind generation profiles from the Open-Meteo Historical
Weather API (ERA5 reanalysis, no API key required) and write them in the same
.npz format the NREL pipeline produces, so precompute_us_reliability_map.py can
consume them unchanged in --resource-mode real.

Open-Meteo fields used:
  shortwave_radiation (W/m^2) -> treated as GHI for solar_generation_profile_from_nsrdb
  temperature_2m (C)          -> temperature for the PV temperature derate
  wind_speed_100m (m/s)       -> wind_generation_profile_from_wtk power curve

Leap years (e.g. 2020) are reduced to 8760 hours by dropping Feb 29, matching
the NREL queries that were run with leap_day=false.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from resource_profiles import (
    RealResourceSettings,
    solar_generation_profile_from_nsrdb,
    wind_generation_profile_from_wtk,
    save_profile_cache,
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURS_PER_YEAR = 8760


def load_points(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["points"]


def _get_json(params: dict[str, str], retries: int = 5, base_delay: float = 3.0) -> dict[str, Any]:
    url = ARCHIVE_URL + "?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            # 429 = rate limited, 5xx = transient
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(base_delay * (attempt + 1))
                last_error = RuntimeError(f"HTTP {exc.code}: {body}")
                continue
            raise RuntimeError(f"HTTP {exc.code} from Open-Meteo: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < retries:
                time.sleep(base_delay * (attempt + 1))
                last_error = exc
                continue
            raise
    assert last_error is not None
    raise last_error


def _drop_feb29(times: list[str], *series: list) -> tuple[list, ...]:
    keep = [i for i, t in enumerate(times) if t[5:10] != "02-29"]
    return tuple([s[i] for i in keep] for s in series)


def fetch_solar_rows(lat: float, lon: float, year: int) -> list[dict[str, float]]:
    data = _get_json(
        {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": "shortwave_radiation,temperature_2m",
            "timezone": "GMT",
        }
    )
    h = data["hourly"]
    ghi, temp = _drop_feb29(h["time"], h["shortwave_radiation"], h["temperature_2m"])
    rows = [
        {
            "ghi": float(g) if g is not None else 0.0,
            "temperature_c": float(t) if t is not None else 15.0,
        }
        for g, t in zip(ghi, temp)
    ]
    return rows


def fetch_wind_rows(lat: float, lon: float, year: int) -> list[dict[str, float]]:
    data = _get_json(
        {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": "wind_speed_100m",
            "wind_speed_unit": "ms",
            "timezone": "GMT",
        }
    )
    h = data["hourly"]
    (ws,) = _drop_feb29(h["time"], h["wind_speed_100m"])
    return [{"wind_speed_ms": float(w) if w is not None else 0.0} for w in ws]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Open-Meteo ERA5 profiles -> .npz")
    p.add_argument("--site-json", type=Path, default=Path("outputs/us_towns_dedup.json"))
    p.add_argument("--profile-cache-root", type=Path, default=Path("references/data/solar_wind_profiles"))
    p.add_argument("--solar-year", type=int, default=2020)
    p.add_argument("--wind-year", type=int, default=2014)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-sites", type=int, default=0)
    p.add_argument("--site-ids", default="", help="comma-separated site indices to fetch (overrides start/max range)")
    p.add_argument("--sleep", type=float, default=0.4, help="seconds between sites (politeness)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.site_json)
    end = len(points) if args.max_sites <= 0 else min(len(points), args.start_index + args.max_sites)
    settings = RealResourceSettings(solar_year=args.solar_year, wind_year=args.wind_year)
    args.profile_cache_root.mkdir(parents=True, exist_ok=True)

    if args.site_ids.strip():
        indices = [int(x) for x in args.site_ids.split(",") if x.strip() != ""]
    else:
        indices = list(range(args.start_index, end))

    done = skipped = failed = 0
    for idx in indices:
        point = points[idx]
        out_path = args.profile_cache_root / f"site_{idx:04d}.npz"
        if out_path.exists():
            skipped += 1
            continue
        lat = round(float(point["lat"]), 3)
        lon = round(float(point["lon"]), 3)
        try:
            solar_rows = fetch_solar_rows(lat, lon, args.solar_year)
            wind_rows = fetch_wind_rows(lat, lon, args.wind_year)
            solar_profile = solar_generation_profile_from_nsrdb(solar_rows, settings)
            wind_profile = wind_generation_profile_from_wtk(wind_rows, settings)
            if len(solar_profile) != HOURS_PER_YEAR or len(wind_profile) != HOURS_PER_YEAR:
                raise ValueError(
                    f"profile length {len(solar_profile)}/{len(wind_profile)} != {HOURS_PER_YEAR}"
                )
            metadata = {
                "source": "open-meteo",
                "dataset": "ERA5_reanalysis",
                "provider_label": "Real ERA5 (Open-Meteo)",
                "lat": lat,
                "lon": lon,
                "solar_year": args.solar_year,
                "wind_year": args.wind_year,
                "settings": json.loads(json.dumps(settings.__dict__)),
                "solar_cf": round(float(np.mean(solar_profile)), 4),
                "wind_cf": round(float(np.mean(wind_profile)), 4),
            }
            save_profile_cache(out_path, solar_profile, wind_profile, metadata)
            done += 1
            if done % 25 == 0 or idx == end - 1:
                print(
                    f"[{idx+1}/{end}] {point.get('name','')}, {point.get('state','')}: "
                    f"solarCF={metadata['solar_cf']:.3f} windCF={metadata['wind_cf']:.3f} "
                    f"(done={done} skip={skipped} fail={failed})",
                    flush=True,
                )
        except Exception as exc:  # keep going; rerun fills gaps (resumable)
            failed += 1
            print(f"[{idx+1}/{end}] FAILED {point.get('name','')}: {exc}", flush=True)
        time.sleep(args.sleep)

    print(json.dumps({"done": done, "skipped": skipped, "failed": failed, "range": [args.start_index, end]}, indent=2))


if __name__ == "__main__":
    main()
