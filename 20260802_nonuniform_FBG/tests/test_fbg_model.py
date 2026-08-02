import unittest
from dataclasses import replace

import numpy as np

from fbg_model import (
    FBGParams,
    ThermalParams,
    fbg_shift_nm,
    fin_temperature_rise_k,
    hotspot_temperature_c,
    recovered_hotspot_temperature_c,
    reflection_spectrum,
)


class ModelTests(unittest.TestCase):
    def test_segmented_temperature_sensitivity(self):
        # 23->400 C: 0.77 + 1.18 + 1.33 + 1.44 nm.
        self.assertAlmostEqual(float(fbg_shift_nm(400.0)), 4.72, places=10)
        self.assertAlmostEqual(float(fbg_shift_nm(479.0)), 4.72 + 79 * 0.0151, places=10)

    def test_hotspot_fwhm(self):
        t = hotspot_temperature_c(np.array([5.0e-3, 6.4e-3]), 479.0)
        self.assertAlmostEqual(t[0], 479.0, places=10)
        self.assertAlmostEqual(t[1] - 23.0, 0.5 * (479.0 - 23.0), places=8)

    def test_recovered_hotspot_fwhm(self):
        center = 2.06563e-3
        full_width = 2.74276e-3
        left = center - full_width * 0.15
        right = center + full_width * 0.85
        t = recovered_hotspot_temperature_c(np.array([center, left, right]), 479.0)
        self.assertAlmostEqual(t[0], 479.0, places=5)
        self.assertAlmostEqual(t[1] - 23.0, 228.0, places=4)
        self.assertAlmostEqual(t[2] - 23.0, 228.0, places=4)

    def test_uniform_grating_closed_form(self):
        fbg = replace(FBGParams(), segments=300, apodization="uniform")
        modeled = reflection_spectrum(
            np.array([fbg.lambda0_nm]), np.full(fbg.segments, fbg.lambda0_nm), fbg
        )[0]
        expected = np.tanh(fbg.kappa0_m_inv * fbg.length_m) ** 2
        self.assertAlmostEqual(modeled, expected, places=10)

    def test_default_cold_bandwidth_matches_digitized_trace(self):
        fbg = replace(FBGParams(), segments=400)
        wavelength = np.linspace(1547.0, 1548.0, 2001)
        reflected = reflection_spectrum(
            wavelength, np.full(fbg.segments, fbg.lambda0_nm), fbg
        )
        above = np.flatnonzero(reflected >= reflected.max() / 2.0)
        bandwidth_nm = wavelength[above[-1]] - wavelength[above[0]]
        self.assertGreater(bandwidth_nm, 0.28)
        self.assertLess(bandwidth_nm, 0.32)

    def test_fin_solution_has_insulated_ends(self):
        thermal = ThermalParams()
        dz = 1e-8
        z = np.array([0.0, dz, thermal.length_m - dz, thermal.length_m])
        theta = fin_temperature_rise_k(z, 0.367, thermal)
        left_slope = (theta[1] - theta[0]) / dz
        right_slope = (theta[-1] - theta[-2]) / dz
        peak_scale = theta.max() / thermal.length_m
        self.assertLess(abs(left_slope), 5e-3 * peak_scale)
        self.assertLess(abs(right_slope), 5e-3 * peak_scale)


if __name__ == "__main__":
    unittest.main()
