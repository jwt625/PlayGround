#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


NSRDB_BASE_URL = "https://developer.nlr.gov/api/nsrdb/v2/solar/nsrdb-GOES-conus-v4-0-0-download.csv"
WTK_BASE_URL = "https://developer.nlr.gov/api/wind-toolkit/v2/wind/wtk-download.csv"
SOLAR_ATTRIBUTES = ("ghi", "dni", "dhi", "air_temperature", "wind_speed")
WIND_ATTRIBUTES = ("windspeed_100m", "winddirection_100m", "temperature_100m", "pressure_100m")


@dataclass
class ApiCredentials:
    api_key: str
    email: str
    full_name: str
    affiliation: str
    reason: str = "AIDC renewable reliability modeling"
    mailing_list: bool = False


@dataclass
class RealResourceSettings:
    solar_year: int = 2020
    wind_year: int = 2014
    solar_losses_fraction: float = 0.14
    solar_temperature_coefficient_per_c: float = -0.004
    solar_reference_temperature_c: float = 25.0
    wind_cut_in_ms: float = 3.0
    wind_rated_ms: float = 12.0
    wind_cut_out_ms: float = 25.0
    wind_losses_fraction: float = 0.15


def load_api_credentials_from_env() -> ApiCredentials:
    api_key = os.environ.get("NREL_API_KEY") or os.environ.get("NLR_API_KEY")
    email = os.environ.get("NREL_API_EMAIL")
    full_name = os.environ.get("NREL_API_FULL_NAME")
    affiliation = os.environ.get("NREL_API_AFFILIATION", "PlayGround")
    reason = os.environ.get("NREL_API_REASON", "AIDC renewable reliability modeling")
    if not api_key:
        raise RuntimeError("Missing NREL/NLR API key. Set NREL_API_KEY or NLR_API_KEY.")
    if not email:
        raise RuntimeError("Missing NREL API email. Set NREL_API_EMAIL.")
    if not full_name:
        try:
            full_name = subprocess.check_output(["git", "config", "user.name"], text=True).strip()
        except Exception:
            full_name = ""
    if not full_name:
        full_name = "PlayGround User"
    return ApiCredentials(
        api_key=api_key,
        email=email,
        full_name=full_name,
        affiliation=affiliation,
        reason=reason,
        mailing_list=False,
    )


def _site_cache_key(lat: float, lon: float, year: int, dataset: str) -> str:
    raw = f"{dataset}:{year}:{lat:.5f}:{lon:.5f}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def cache_path_for_site(cache_dir: Path, dataset: str, lat: float, lon: float, year: int) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{dataset}_{_site_cache_key(lat, lon, year, dataset)}.csv"


def _build_query(credentials: ApiCredentials, lat: float, lon: float, year: int, attrs: tuple[str, ...]) -> dict[str, str]:
    return {
        "api_key": credentials.api_key,
        "wkt": f"POINT({lon:.3f} {lat:.3f})",
        "attributes": ",".join(attrs),
        "names": str(year),
        "utc": "false",
        "leap_day": "false",
        "interval": "60",
        "email": credentials.email,
        "full_name": credentials.full_name,
        "affiliation": credentials.affiliation,
        "reason": credentials.reason,
        "mailing_list": "false" if not credentials.mailing_list else "true",
    }


def _download_csv(base_url: str, params: dict[str, str], retries: int = 3, retry_delay_s: float = 5.0) -> str:
    url = base_url + "?" + urllib.parse.urlencode(params)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            if exc.code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(retry_delay_s * (attempt + 1))
                last_error = RuntimeError(f"HTTP {exc.code} from {base_url}: {body}")
                continue
            raise RuntimeError(f"HTTP {exc.code} from {base_url}: {body}") from exc
    assert last_error is not None
    raise last_error


def fetch_nsrdb_csv_text(
    lat: float,
    lon: float,
    year: int,
    credentials: ApiCredentials,
    cache_dir: Path,
) -> str:
    cache_path = cache_path_for_site(cache_dir, "nsrdb", lat, lon, year)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    text = _download_csv(NSRDB_BASE_URL, _build_query(credentials, lat, lon, year, SOLAR_ATTRIBUTES))
    cache_path.write_text(text, encoding="utf-8")
    return text


def fetch_wtk_csv_text(
    lat: float,
    lon: float,
    year: int,
    credentials: ApiCredentials,
    cache_dir: Path,
) -> str:
    cache_path = cache_path_for_site(cache_dir, "wtk", lat, lon, year)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    text = _download_csv(WTK_BASE_URL, _build_query(credentials, lat, lon, year, WIND_ATTRIBUTES))
    cache_path.write_text(text, encoding="utf-8")
    return text


def parse_nsrdb_csv(text: str) -> tuple[dict[str, str], list[dict[str, float]]]:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        raise ValueError("NSRDB CSV must contain metadata rows plus data.")
    meta_keys = rows[0]
    meta_values = rows[1]
    metadata = {key.strip(): value.strip() for key, value in zip(meta_keys, meta_values) if key.strip()}
    data_reader = csv.DictReader(io.StringIO("\n".join(",".join(row) for row in rows[2:])))
    out: list[dict[str, float]] = []
    for row in data_reader:
        out.append(
            {
                "year": float(row["Year"]),
                "month": float(row["Month"]),
                "day": float(row["Day"]),
                "hour": float(row["Hour"]),
                "minute": float(row["Minute"]),
                "ghi": float(row["GHI"]),
                "dni": float(row["DNI"]),
                "dhi": float(row["DHI"]),
                "temperature_c": float(row["Temperature"]),
                "wind_speed_ms": float(row["Wind Speed"]),
            }
        )
    return metadata, out


def parse_wtk_csv(text: str) -> tuple[dict[str, str], list[dict[str, float]]]:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        raise ValueError("WTK CSV must contain metadata row plus data.")
    meta_row = rows[0]
    metadata = {}
    for idx in range(0, len(meta_row) - 1, 2):
        metadata[meta_row[idx].strip()] = meta_row[idx + 1].strip()
    data_reader = csv.DictReader(io.StringIO("\n".join(",".join(row) for row in rows[1:])))
    out: list[dict[str, float]] = []
    for row in data_reader:
        out.append(
            {
                "year": float(row["Year"]),
                "month": float(row["Month"]),
                "day": float(row["Day"]),
                "hour": float(row["Hour"]),
                "minute": float(row["Minute"]),
                "wind_speed_ms": float(row["wind speed at 100m (m/s)"]),
                "wind_direction_deg": float(row["wind direction at 100m (deg)"]),
                "temperature_c": float(row["air temperature at 100m (C)"]),
                "pressure_pa": float(row["air pressure at 100m (Pa)"]),
            }
        )
    return metadata, out


def solar_generation_profile_from_nsrdb(rows: list[dict[str, float]], settings: RealResourceSettings) -> np.ndarray:
    profile = np.zeros(len(rows), dtype=np.float64)
    temp_coeff = settings.solar_temperature_coefficient_per_c
    temp_ref = settings.solar_reference_temperature_c
    loss_multiplier = 1.0 - settings.solar_losses_fraction
    for idx, row in enumerate(rows):
        irradiance_fraction = max(0.0, row["ghi"]) / 1000.0
        temp_factor = 1.0 + temp_coeff * (row["temperature_c"] - temp_ref)
        profile[idx] = max(0.0, min(1.2, irradiance_fraction * temp_factor * loss_multiplier))
    return np.clip(profile, 0.0, 1.0)


def _generic_wind_power_fraction(wind_speed_ms: float, settings: RealResourceSettings) -> float:
    if wind_speed_ms < settings.wind_cut_in_ms or wind_speed_ms >= settings.wind_cut_out_ms:
        return 0.0
    if wind_speed_ms >= settings.wind_rated_ms:
        return 1.0
    numerator = wind_speed_ms**3 - settings.wind_cut_in_ms**3
    denominator = max(settings.wind_rated_ms**3 - settings.wind_cut_in_ms**3, 1e-9)
    return max(0.0, min(1.0, numerator / denominator))


def wind_generation_profile_from_wtk(rows: list[dict[str, float]], settings: RealResourceSettings) -> np.ndarray:
    loss_multiplier = 1.0 - settings.wind_losses_fraction
    profile = np.zeros(len(rows), dtype=np.float64)
    for idx, row in enumerate(rows):
        profile[idx] = _generic_wind_power_fraction(row["wind_speed_ms"], settings) * loss_multiplier
    return np.clip(profile, 0.0, 1.0)


def fetch_real_resource_profiles(
    lat: float,
    lon: float,
    credentials: ApiCredentials,
    cache_root: Path,
    settings: RealResourceSettings | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    settings = settings or RealResourceSettings()
    solar_text = fetch_nsrdb_csv_text(lat, lon, settings.solar_year, credentials, cache_root / "nsrdb")
    wind_text = fetch_wtk_csv_text(lat, lon, settings.wind_year, credentials, cache_root / "wtk")
    solar_meta, solar_rows = parse_nsrdb_csv(solar_text)
    wind_meta, wind_rows = parse_wtk_csv(wind_text)
    solar_profile = solar_generation_profile_from_nsrdb(solar_rows, settings)
    wind_profile = wind_generation_profile_from_wtk(wind_rows, settings)
    if len(solar_profile) != len(wind_profile):
        raise ValueError(f"Solar/Wind profile length mismatch: {len(solar_profile)} vs {len(wind_profile)}")
    metadata = {
        "solar": solar_meta,
        "wind": wind_meta,
        "settings": json.loads(json.dumps(settings.__dict__)),
    }
    return solar_profile, wind_profile, metadata
