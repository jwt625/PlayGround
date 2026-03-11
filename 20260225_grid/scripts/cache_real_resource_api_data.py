#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from resource_profiles import (
    cache_path_for_site,
    fetch_nsrdb_csv_text,
    fetch_wtk_csv_text,
    load_api_credentials_from_env,
)


def load_sites(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["points"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overnight/resumable cache warmer for NSRDB + WIND Toolkit point downloads.")
    parser.add_argument("--site-json", type=Path, default=Path("outputs/us_towns_dedup.json"))
    parser.add_argument("--cache-root", type=Path, default=Path("references/data/solar_wind_api"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/real_resource_cache_manifest.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/real_resource_cache_summary.json"))
    parser.add_argument("--solar-year", type=int, default=2020)
    parser.add_argument("--wind-year", type=int, default=2014)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-sites", type=int, default=0)
    parser.add_argument("--min-call-spacing-seconds", type=float, default=45.0)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--batch-sleep-seconds", type=float, default=600.0)
    parser.add_argument("--cooldown-on-rate-limit-seconds", type=float, default=3600.0)
    parser.add_argument("--stop-on-rate-limit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def append_manifest(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def main() -> None:
    args = parse_args()
    credentials = load_api_credentials_from_env()
    sites = load_sites(args.site_json)
    end_index = len(sites) if args.max_sites <= 0 else min(len(sites), args.start_index + args.max_sites)
    selected = sites[args.start_index:end_index]

    summary = {
        "site_json": str(args.site_json),
        "site_count_total": len(sites),
        "site_count_selected": len(selected),
        "start_index": args.start_index,
        "end_index": end_index,
        "solar_year": args.solar_year,
        "wind_year": args.wind_year,
        "min_call_spacing_seconds": args.min_call_spacing_seconds,
        "batch_size": args.batch_size,
        "batch_sleep_seconds": args.batch_sleep_seconds,
        "cooldown_on_rate_limit_seconds": args.cooldown_on_rate_limit_seconds,
        "dry_run": args.dry_run,
        "solar_calls_needed": 0,
        "wind_calls_needed": 0,
        "total_api_calls_needed": 0,
        "sites_completed": 0,
        "sites_failed": 0,
        "rate_limit_events": 0,
        "estimated_runtime_hours_at_configured_spacing": 0.0,
    }

    nsrdb_dir = args.cache_root / "nsrdb"
    wtk_dir = args.cache_root / "wtk"
    queue: list[dict[str, Any]] = []
    for offset, site in enumerate(selected):
        index = args.start_index + offset
        lat = float(site["lat"])
        lon = float(site["lon"])
        nsrdb_path = cache_path_for_site(nsrdb_dir, "nsrdb", lat, lon, args.solar_year)
        wtk_path = cache_path_for_site(wtk_dir, "wtk", lat, lon, args.wind_year)
        need_solar = not nsrdb_path.exists()
        need_wind = not wtk_path.exists()
        summary["solar_calls_needed"] += int(need_solar)
        summary["wind_calls_needed"] += int(need_wind)
        queue.append(
            {
                "index": index,
                "site": site,
                "need_solar": need_solar,
                "need_wind": need_wind,
                "nsrdb_path": str(nsrdb_path),
                "wtk_path": str(wtk_path),
            }
        )

    summary["total_api_calls_needed"] = summary["solar_calls_needed"] + summary["wind_calls_needed"]
    summary["estimated_runtime_hours_at_configured_spacing"] = round(
        summary["total_api_calls_needed"] * args.min_call_spacing_seconds / 3600.0, 2
    )

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    last_live_call_at = 0.0
    live_call_counter = 0

    for entry in queue:
        site = entry["site"]
        lat = float(site["lat"])
        lon = float(site["lon"])
        site_record = {
            "index": entry["index"],
            "name": site.get("name", ""),
            "state": site.get("state", ""),
            "lat": lat,
            "lon": lon,
            "solar_year": args.solar_year,
            "wind_year": args.wind_year,
            "need_solar": entry["need_solar"],
            "need_wind": entry["need_wind"],
        }
        try:
            for dataset_name, needed, fn in (
                ("solar", entry["need_solar"], fetch_nsrdb_csv_text),
                ("wind", entry["need_wind"], fetch_wtk_csv_text),
            ):
                if not needed:
                    continue
                elapsed = time.time() - last_live_call_at
                if live_call_counter > 0 and elapsed < args.min_call_spacing_seconds:
                    time.sleep(args.min_call_spacing_seconds - elapsed)

                try:
                    fn(
                        lat=lat,
                        lon=lon,
                        year=args.solar_year if dataset_name == "solar" else args.wind_year,
                        credentials=credentials,
                        cache_dir=nsrdb_dir if dataset_name == "solar" else wtk_dir,
                    )
                except Exception as exc:  # keep batch alive and log status
                    body = str(exc)
                    if "HTTP 429" in body or "OVER_RATE_LIMIT" in body:
                        summary["rate_limit_events"] += 1
                        append_manifest(
                            args.manifest,
                            {
                                **site_record,
                                "status": "rate_limited",
                                "dataset": dataset_name,
                                "error": body,
                                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            },
                        )
                        if args.stop_on_rate_limit:
                            summary["sites_failed"] += 1
                            args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                            raise
                        time.sleep(args.cooldown_on_rate_limit_seconds)
                        raise
                    raise

                last_live_call_at = time.time()
                live_call_counter += 1
                if args.batch_size > 0 and live_call_counter % args.batch_size == 0:
                    time.sleep(args.batch_sleep_seconds)

            summary["sites_completed"] += 1
            append_manifest(
                args.manifest,
                {
                    **site_record,
                    "status": "ok",
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        except Exception as exc:
            summary["sites_failed"] += 1
            append_manifest(
                args.manifest,
                {
                    **site_record,
                    "status": "failed",
                    "error": str(exc),
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
            if args.stop_on_rate_limit and ("HTTP 429" in str(exc) or "OVER_RATE_LIMIT" in str(exc)):
                break

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
