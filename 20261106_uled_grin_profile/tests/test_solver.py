import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, kv

from grinopt import material
from grinopt.solver import Grid, solve_modes


def test_silica_sodium_d():
    # Malitson fused silica n_D ~ 1.4584
    assert abs(float(material.index(0.5893, 0.0)) - 1.4584) < 2e-4


def test_group_index_derivative():
    lam = 0.43
    h = 1e-5
    n_p = material.index(lam + h, 0.08)
    n_m = material.index(lam - h, 0.08)
    n, ng = material.index_and_group(lam, 0.08)
    assert abs((n - lam * (n_p - n_m) / (2 * h)) - ng) < 1e-8


def _lp_beta(V: float, l: int, a: float, k: float, n1: float, n2: float) -> list[float]:
    def f(u: float) -> float:
        w = np.sqrt(V**2 - u**2)
        return u * jv(l - 1, u) / jv(l, u) + w * kv(l - 1, w) / kv(l, w)

    us = np.linspace(1e-3, V - 1e-6, 20000)
    vals = np.array([f(u) for u in us])
    roots = []
    for i in range(len(us) - 1):
        if np.sign(vals[i]) != np.sign(vals[i + 1]) and abs(vals[i]) < 50 and abs(vals[i + 1]) < 50:
            roots.append(brentq(f, us[i], us[i + 1]))
    return [np.sqrt((k * n1) ** 2 - (u / a) ** 2) for u in roots]


def test_step_index_lp_modes():
    lam, a, n1, n2 = 0.85, 5.0, 1.46, 1.45
    dr = 0.01
    grid = Grid.make(a + 20.0, dr)
    n = np.where(grid.r < a, n1, n2)
    modes = solve_modes(grid, lam, n, n, n2)
    k = 2 * np.pi / lam
    V = k * a * np.sqrt(n1**2 - n2**2)
    for l in (0, 1, 2):
        ref = _lp_beta(V, l, a, k, n1, n2)
        num = np.sort(modes.beta[modes.l == l])[::-1]
        assert len(num) == len(ref)
        # compare normalized propagation constant b
        b_ref = (np.array(ref) ** 2 / k**2 - n2**2) / (n1**2 - n2**2)
        b_num = (num**2 / k**2 - n2**2) / (n1**2 - n2**2)
        assert np.max(np.abs(b_ref - b_num)) < 2e-3


def test_parabolic_harmonic_oscillator():
    lam, a, n1, delta = 0.85, 25.0, 1.46, 0.01
    grid = Grid.make(80.0, 0.02)
    n = n1 * np.sqrt(1 - 2 * delta * (grid.r / a) ** 2)  # infinite parabola, no cutoff inside grid
    k = 2 * np.pi / lam
    floor = n1 * np.sqrt(1 - 2 * delta * 0.6)
    modes = solve_modes(grid, lam, n, n, floor, l_max=3)
    omega = k * n1 * np.sqrt(2 * delta) / a
    for l in range(3):
        num = np.sort(modes.beta[modes.l == l] ** 2)[::-1][:3]
        ref = (k * n1) ** 2 - 2 * omega * (2 * np.arange(1, 4) + l - 1)
        assert np.max(np.abs(num - ref) / (2 * omega)) < 1e-2


def test_hellmann_feynman_group_index():
    a, x0 = 10.0, 0.06
    grid = Grid.make(a + 10.0, 0.02)
    lam = 0.43
    h = 1e-4

    def solve(lam_):
        x = np.where(grid.r < a, x0 * (1 - (grid.r / a) ** 2), 0.0)
        n, ng = material.index_and_group(lam_, x)
        return solve_modes(grid, lam_, n, ng, float(material.index(lam_, 0.0)), l_max=2)

    m0, mp, mm = solve(lam), solve(lam + h), solve(lam - h)
    kp, km = 2 * np.pi / (lam + h), 2 * np.pi / (lam - h)
    for l in range(3):
        b0 = m0.beta[m0.l == l][:4]
        bp = mp.beta[mp.l == l][:4]
        bm = mm.beta[mm.l == l][:4]
        ng_fd = (bp - bm) / (kp - km)
        assert np.max(np.abs(ng_fd - m0.ng[m0.l == l][:4])) < 1e-6
        assert b0.size == 4
