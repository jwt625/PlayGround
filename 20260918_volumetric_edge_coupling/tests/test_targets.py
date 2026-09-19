"""Unit tests for the paper-target loader and 1-dB crossing analysis."""

from __future__ import annotations

import numpy as np

from src.targets import load_tables, peak_and_edges


def test_tables_round_trip():
    t = load_tables()
    assert t["table1_optimum"]["waveguide_angle_deg"] == 53.50
    assert t["table1_optimum"]["waveguide_thickness_nm"] == 262.0
    assert t["table3_tolerances_1dB"]["thickness"]["tolerance_nm"] == 7.0


def test_relative_crossing_matches_known_gaussian():
    wl = np.linspace(1450, 1650, 4001)
    # intensity FWHM level: exp(-2 (dw/w)^2); 1 dB drop -> level e^-1
    eta = 0.88 * np.exp(-((wl - 1550) / 50.0) ** 2)
    r = peak_and_edges(wl, eta, "relative")
    # level 0.88*10^-0.1 -> (dw/50)^2 = -ln(10^-0.1) -> dw = 24.0 nm
    assert abs(r["lambda_left_nm"] - 1526.0) < 0.5
    assert abs(r["lambda_right_nm"] - 1574.0) < 0.5
    assert abs(r["level"] - 0.6990088465573677) < 1e-9


def test_absolute_definition_is_narrower_than_relative():
    wl = np.linspace(1500, 1600, 2001)
    eta = 0.88 * np.exp(-((wl - 1550) / 70.0) ** 2)
    rel = peak_and_edges(wl, eta, "relative")
    absol = peak_and_edges(wl, eta, "absolute")
    assert absol["bandwidth_nm"] < rel["bandwidth_nm"]
