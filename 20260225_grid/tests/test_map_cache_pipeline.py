from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_interactive_reliability_map import merge_payloads
from scripts.resource_profiles import load_profile_cache, save_profile_cache


class MapCachePipelineTest(unittest.TestCase):
    def test_profile_cache_round_trip(self) -> None:
        solar = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        wind = np.array([0.2, 0.4, 0.6], dtype=np.float64)
        metadata = {"site_id": 7, "name": "Test"}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "site_0007.npz"
            save_profile_cache(path, solar, wind, metadata)
            solar_loaded, wind_loaded, metadata_loaded = load_profile_cache(path)
        np.testing.assert_allclose(solar_loaded, solar)
        np.testing.assert_allclose(wind_loaded, wind)
        self.assertEqual(metadata_loaded, metadata)

    def test_merge_payloads_overrides_primary_sites_only(self) -> None:
        fallback = {
            "meta": {"model": "synthetic", "resource_mode": "synthetic", "site_source": "fallback.json"},
            "axes": {"solar_mw": [0, 100], "wind_mw": [0, 100], "bess_mwh": [0, 1000]},
            "sites": [
                {"site_id": 0, "name": "A", "lat": 1.0, "lon": -1.0},
                {"site_id": 1, "name": "B", "lat": 2.0, "lon": -2.0},
                {"site_id": 2, "name": "C", "lat": 3.0, "lon": -3.0},
            ],
            "values_by_combo": [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
                [13.0, 23.0, 33.0],
                [14.0, 24.0, 34.0],
                [15.0, 25.0, 35.0],
                [16.0, 26.0, 36.0],
                [17.0, 27.0, 37.0],
            ],
        }
        primary = {
            "meta": {"model": "real", "resource_mode": "real", "site_source": "primary.json"},
            "axes": {"solar_mw": [0, 100], "wind_mw": [0, 100], "bess_mwh": [0, 1000]},
            "sites": [
                {"site_id": 1, "name": "B-real", "lat": 20.0, "lon": -20.0},
                {"site_id": 2, "name": "C-real", "lat": 30.0, "lon": -30.0},
            ],
            "values_by_combo": [
                [88.0, 99.0],
                [81.0, 91.0],
                [82.0, 92.0],
                [83.0, 93.0],
                [84.0, 94.0],
                [85.0, 95.0],
                [86.0, 96.0],
                [87.0, 97.0],
            ],
        }
        merged = merge_payloads(primary, fallback)
        self.assertEqual([site["site_id"] for site in merged["sites"]], [0, 1, 2])
        self.assertEqual(merged["sites"][1]["name"], "B-real")
        self.assertEqual(merged["sites"][2]["name"], "C-real")
        self.assertEqual(merged["values_by_combo"][0], [10.0, 88.0, 99.0])
        self.assertEqual(merged["values_by_combo"][7], [17.0, 87.0, 97.0])


if __name__ == "__main__":
    unittest.main()
