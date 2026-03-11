#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from resource_profiles import (
    RealResourceSettings,
    fetch_real_resource_profiles,
    load_api_credentials_from_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and summarize one site of NSRDB + WIND Toolkit data.")
    parser.add_argument("--lat", type=float, default=40.130)
    parser.add_argument("--lon", type=float, default=-105.240)
    parser.add_argument("--cache-root", type=Path, default=Path("references/data/solar_wind_api"))
    parser.add_argument("--out", type=Path, default=Path("outputs/real_resource_validation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    credentials = load_api_credentials_from_env()
    settings = RealResourceSettings()
    solar_profile, wind_profile, metadata = fetch_real_resource_profiles(
        lat=args.lat,
        lon=args.lon,
        credentials=credentials,
        cache_root=args.cache_root,
        settings=settings,
    )
    payload = {
        "site": {"lat": args.lat, "lon": args.lon},
        "solar": {
            "rows": len(solar_profile),
            "mean": float(np.mean(solar_profile)),
            "max": float(np.max(solar_profile)),
            "nonzero_hours": int(np.sum(solar_profile > 0)),
            "metadata": metadata["solar"],
        },
        "wind": {
            "rows": len(wind_profile),
            "mean": float(np.mean(wind_profile)),
            "max": float(np.max(wind_profile)),
            "nonzero_hours": int(np.sum(wind_profile > 0)),
            "metadata": metadata["wind"],
        },
        "settings": metadata["settings"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
