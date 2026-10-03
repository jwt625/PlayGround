"""Procedural SFX track for the film: every effect is synthesised in numpy (no samples, no downloads, deterministic).

Run (repo root): uv run --no-project --with numpy --with soundfile python scripts/audio/make_sfx.py
Reads scripts/audio/sfx_cues.json, writes outputs/audio/sfx_20261002.wav (48 kHz stereo, 62.0 s, peak < -3 dBFS).
Each effect is generated mono (shotgun: stereo), normalised to peak 1.0, scaled by the cue gain (linear peak), panned
(constant power; params.pan_end moves the pan linearly over the effect), faded in/out by a few ms and summed at t - LEAD so
that the main transient of the effect lands at the cue time. The voice track is never read or modified here.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
CUES = os.path.join(HERE, "sfx_cues.json")
OUT = os.path.join(PROJ, "outputs", "audio", "sfx_20261002.wav")
SR = 48000
TOTAL = 62.0
LEAD = 0.005          # silent lead-in so a hard transient is not eaten by the fade-in
FADE_IN, FADE_OUT = 0.003, 0.008
PEAK_LIMIT = 0.70     # -3.1 dBFS


# ---------------------------------------------------------------- DSP helpers
def tarr(n):
    return np.arange(n) / SR


def fft_band(x, lo=None, hi=None, edge=0.3):
    """Zero-phase band filter by FFT masking; lo/hi in Hz, edge = transition width in octaves."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    lf = np.log2(np.maximum(f, 1e-3))
    m = np.ones_like(f)
    if lo:
        m *= 0.5 * (1 + np.tanh((lf - np.log2(lo)) / edge * 2))
    if hi:
        m *= 0.5 * (1 - np.tanh((lf - np.log2(hi)) / edge * 2))
    return np.fft.irfft(X * m, n)


def exp_env(n, tau, att=0.0):
    t = tarr(n)
    e = np.exp(-t / tau)
    if att > 0:
        e *= np.minimum(1.0, t / att)
    return e


def sweep_phase(freq):
    return 2 * np.pi * np.cumsum(freq) / SR


def place(buf, x, start):
    """Add x into buf at sample index start (clipped)."""
    s = int(round(start))
    if s >= len(buf):
        return
    a = max(0, -s)
    e = min(len(x), len(buf) - s)
    if e > a:
        buf[s + a:s + e] += x[a:e]


def norm(x):
    p = np.max(np.abs(x))
    return x / p if p > 0 else x


def swept_noise(n, rng, f0, f1, bw=1.0, curve=1.0, nb=22):
    """Noise whose spectral centre moves from f0 to f1 (Hz, log), band width bw octaves (gaussian), via a filter bank."""
    t = tarr(n)
    u = np.clip(t / max(t[-1], 1e-9), 0, 1) ** curve
    lfc = np.log2(f0) + (np.log2(f1) - np.log2(f0)) * u
    cs = np.logspace(np.log10(80), np.log10(12000), nb)
    w = rng.standard_normal(n)
    out = np.zeros(n)
    for c in cs:
        band = fft_band(w, c / 2 ** 0.25, c * 2 ** 0.25, 0.15)
        g = np.exp(-0.5 * ((np.log2(c) - lfc) / bw) ** 2)
        out += band * g
    return out


def convolve_ir(x, ir, n_out=None):
    n = len(x) + len(ir)
    N = 1 << int(np.ceil(np.log2(n)))
    y = np.fft.irfft(np.fft.rfft(x, N) * np.fft.rfft(ir, N), N)[:n]
    if n_out is not None:
        y = np.pad(y, (0, max(0, n_out - len(y))))[:n_out]
    return y


def asym_env(n, peak, power=1.0):
    u = np.linspace(0, 1, n)
    e = np.where(u < peak, np.sin(0.5 * np.pi * u / max(peak, 1e-6)) ** 2, np.cos(0.5 * np.pi * (u - peak) / max(1 - peak, 1e-6)) ** 2)
    return e ** power


def ring(n, f, tau, att=0.0005):
    t = tarr(n)
    return np.sin(2 * np.pi * f * t) * np.exp(-t / tau) * np.minimum(1.0, t / att)


def click(n, rng, lo, hi, tau):
    return fft_band(rng.standard_normal(n), lo, hi, 0.3) * exp_env(n, tau, 0.0002)


# ---------------------------------------------------------------- effects: each returns array (n,) or (n, 2)
def fx_shotgun(n, rng, p):
    size = p.get("size", 1.0)
    t = tarr(n)
    crack = click(n, rng, 250, 7000, 0.016 * size)
    boom = np.sin(sweep_phase(36 + 120 * np.exp(-t / 0.07))) * exp_env(n, 0.26 * size, 0.0015)
    body = fft_band(rng.standard_normal(n), None, 900, 0.5) * exp_env(n, 0.11, 0.001)
    dry = np.tanh(1.6 * (0.9 * crack + 1.0 * boom + 0.8 * body))
    dry *= np.minimum(1.0, t / 0.0008)
    # echo slap-back and a decaying-noise reverb tail (different noise per channel)
    echo = np.zeros(n)
    place(echo, 0.30 * np.roll(dry, 0)[:int(0.25 * SR)], 0.115 * SR)
    place(echo, 0.14 * fft_band(dry[:int(0.25 * SR)], None, 3000, 0.5), 0.26 * SR)
    ir_n = int(1.3 * SR)
    chans = []
    for c in range(2):
        ir = fft_band(rng.standard_normal(ir_n), 120, 3500, 0.5) * np.exp(-tarr(ir_n) / 0.33)
        ir[0] = 0
        wet = convolve_ir(dry[:int(0.3 * SR)], ir, n) * 0.012
        chans.append(dry + echo + wet)
    out = np.stack(chans, 1)
    if p.get("pump", 0):
        tp = 0.55
        pump = np.zeros(n)
        for k, (dt, f, g) in enumerate(((0.0, 1250, 1.0), (0.14, 1900, 0.85))):
            c = ring(int(0.12 * SR), f, 0.012) + 0.7 * click(int(0.12 * SR), rng, 600, 5000, 0.006) + 0.6 * ring(int(0.12 * SR), 210, 0.03)
            place(pump, g * c, (tp + dt) * SR)
        # ejected shell: bright tink plus two bounces
        for dt, f, g in ((0.32, 4300, 0.5), (0.46, 4100, 0.3), (0.55, 4500, 0.15)):
            place(pump, g * (ring(int(0.25 * SR), f, 0.05) + 0.4 * ring(int(0.25 * SR), f * 1.52, 0.03)), (tp + dt) * SR)
        out = out + 0.22 * np.stack([pump, pump], 1)
    return out


def fx_hole_pop(n, rng, p):
    t = tarr(n)
    f = 280 + 900 * (1 - np.exp(-t / 0.018))
    pop = np.sin(sweep_phase(f)) * exp_env(n, 0.035, 0.001)
    return pop + 0.5 * click(n, rng, 1500, 6000, 0.002)


def fx_chip_click(n, rng, p):
    cnt, step, pitch = int(p.get("n", 1)), p.get("step", 0.05), p.get("pitch", 1.0)
    out = np.zeros(n)
    for i in range(cnt):
        pi_ = pitch * (1 + 0.05 * ((i * 7) % 3))
        m = int(0.12 * SR)
        c = 0.9 * click(m, rng, 1800, 7000, 0.0025) + 0.7 * ring(m, 2300 * pi_, 0.004) + 0.4 * ring(m, 650 * pi_, 0.008)
        place(out, c * (1 - 0.06 * i), i * step * SR)
        if p.get("blip", 1):
            mb = int(0.05 * SR)
            tb = tarr(mb)
            f = 1700 * pi_ * (1 + 0.45 * np.minimum(1, tb / 0.03))
            b = (np.sin(sweep_phase(f)) + 0.3 * np.sin(3 * sweep_phase(f))) * np.minimum(1, tb / 0.002) * np.exp(-tb / 0.02)
            place(out, 0.45 * b, (i * step + 0.012) * SR)
    return out


def fx_rattle(n, rng, p):
    cnt, step = int(p.get("n", 6)), p.get("step", 0.06)
    out = np.zeros(n)
    for i in range(cnt):
        m = int(0.08 * SR)
        f = 1400 + 500 * ((i * 5) % 3)
        c = click(m, rng, 900, 5000, 0.003) + 0.8 * ring(m, f, 0.006) + 0.4 * ring(m, 380, 0.01)
        place(out, c * (0.9 ** i), (i * step + rng.uniform(-0.004, 0.004)) * SR)
    return out


def fx_sizzle(n, rng, p):
    rate, tau = p.get("rate", 500), p.get("tau", 0.4)
    t = tarr(n)
    lam = (rate * np.exp(-t / tau) + 40) / SR
    imp = (rng.random(n) < lam) * rng.exponential(1.0, n) * rng.choice([-1, 1], n)
    crack = fft_band(imp, 1800, 9000, 0.4)
    pops = fft_band((rng.random(n) < lam * 0.12) * rng.exponential(1.0, n), 500, 1400, 0.4)
    hiss = fft_band(rng.standard_normal(n), 3500, 9000, 0.4) * 0.05
    x = (crack + 0.7 * pops + hiss) * (np.exp(-t / (tau * 1.3)) * np.minimum(1, t / 0.03))
    return x


def fx_egg_splat(n, rng, p):
    s = p.get("strength", 1.0)
    m = int(0.2 * SR)
    sq = swept_noise(m, rng, 1300, 260, 0.8, 0.7) * exp_env(m, 0.05, 0.002)
    slap = np.sin(sweep_phase(190 - 110 * np.minimum(1, tarr(m) / 0.08))) * exp_env(m, 0.04, 0.001)
    out = np.zeros(n)
    place(out, s * (1.2 * norm(sq) + 0.9 * slap) + 0.4 * s * click(m, rng, 1500, 6000, 0.003), 0)
    # yolk blip and wobble
    mb = int(0.1 * SR)
    tb = tarr(mb)
    place(out, 0.5 * np.sin(sweep_phase(430 - 170 * np.minimum(1, tb / 0.08)) + 0.4 * np.sin(2 * np.pi * 22 * tb)) * np.exp(-tb / 0.05) * np.minimum(1, tb / 0.003), 0.085 * SR)
    mw = int(0.3 * SR)
    tw = tarr(mw)
    place(out, 0.3 * np.sin(2 * np.pi * 330 * tw) * (1 + 0.6 * np.sin(2 * np.pi * 11 * tw)) * np.exp(-tw / 0.11) * np.minimum(1, tw / 0.01), 0.17 * SR)
    return out


def fx_curve_rise(n, rng, p):
    t = tarr(n)
    u = t / t[-1]
    f = p.get("f0", 1200) * 2 ** (p.get("octaves", 1.5) * u)
    ph = sweep_phase(f)
    x = np.sin(ph + 1.2 * np.sin(2 * ph * 0.5)) * (1 + 0.35 * np.sin(2 * np.pi * 17 * t)) + 0.25 * np.sin(1.5 * ph)
    return x * np.minimum(1, t / 0.08) * np.minimum(1, (t[-1] - t) / 0.15)


def fx_curve_fall(n, rng, p):
    t = tarr(n)
    u = t / t[-1]
    f = p.get("f0", 2400) * (p.get("f1", 450) / p.get("f0", 2400)) ** (u ** 0.8) * (1 + 0.03 * np.sin(2 * np.pi * 6 * t))
    ph = sweep_phase(f)
    x = np.sin(ph) + 0.3 * np.sin(2 * ph)
    return x * np.minimum(1, t / 0.05) * (1 - 0.5 * u) * np.minimum(1, (t[-1] - t) / 0.15)


def fx_marker_squeak(n, rng, p):
    cnt = int(p.get("n", 10))
    out = np.zeros(n)
    t0s = np.sort(rng.uniform(0.0, n / SR - 0.25, cnt))
    for t0 in t0s:
        L = int(rng.uniform(0.06, 0.16) * SR)
        tl = tarr(L)
        f0 = rng.uniform(2600, 3600)
        f = f0 * (1 + 0.04 * np.sin(2 * np.pi * rng.uniform(40, 60) * tl)) * (1 + rng.uniform(-0.08, 0.08) * tl / tl[-1])
        s = np.sin(sweep_phase(f)) * (1 + 0.5 * np.sin(2 * np.pi * rng.uniform(25, 35) * tl))
        s += 0.3 * fft_band(rng.standard_normal(L), 2000, 6000, 0.4)
        s *= np.minimum(1, tl / 0.01) * np.minimum(1, (tl[-1] - tl) / 0.03)
        place(out, s, t0 * SR)
    return out


def fx_electron_zip(n, rng, p):
    t = tarr(n)
    f = p.get("f0", 800) * 4 ** np.clip(t / 0.14, 0, 1)
    ph = sweep_phase(f)
    x = np.sin(ph + 1.5 * np.sin(0.5 * ph)) * asym_env(n, 0.45, 1.5)
    return x + 0.2 * np.sin(2 * ph) * asym_env(n, 0.45, 1.5)


def fx_hum(n, rng, p):
    f, sw, sh = p.get("f", 100), p.get("swell", 0.4), p.get("shimmer", 0.0)
    t = tarr(n)
    wob = 1 + 0.01 * np.sin(2 * np.pi * 6 * t)
    ph = sweep_phase(f * wob)
    x = np.sin(ph) + 0.7 * np.sin(2 * ph) + 0.5 * np.sin(3 * ph) + 0.3 * np.sin(4 * ph)
    env = asym_env(n, sw, 1.3)
    x = x * env
    if sh:
        sp = fft_band(rng.standard_normal(n), 2500, 6500, 0.4) * (0.5 + 0.5 * np.sin(2 * np.pi * 23 * t)) ** 2
        x = x / np.max(np.abs(x)) + 0.25 * sh * sp / max(np.max(np.abs(sp)), 1e-9) * env
    return x


def fx_whoosh(n, rng, p):
    f0, f1 = p.get("f0", 400), p.get("f1", 3000)
    x = swept_noise(n, rng, f0, f1, p.get("bw", 1.0), 1.4)
    x = norm(x) * asym_env(n, p.get("peak", 0.6), 1.2)
    return x


def fx_blip(n, rng, p):
    t = tarr(n)
    f = p.get("f", 900)
    return (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(6 * np.pi * f * t)) * np.exp(-t / 0.03) * np.minimum(1, t / 0.002)


def fx_cable_creak(n, rng, p):
    t = tarr(n)
    u = t / t[-1]
    out = np.zeros(n)
    tt = 0.0
    while tt < t[-1] - 0.02:
        L = int(0.012 * SR)
        f = 450 + 600 * u[min(int(tt * SR), n - 1)] * rng.uniform(0.8, 1.2)
        a = (0.4 + 0.6 * np.sin(np.pi * u[min(int(tt * SR), n - 1)])) * rng.uniform(0.5, 1.0)
        place(out, a * (ring(L, f, 0.004) + 0.5 * ring(L, f * 1.9, 0.002)), tt * SR)
        tt += 1.0 / (30 + 60 * u[min(int(tt * SR), n - 1)]) * rng.uniform(0.7, 1.3)
    groan = np.sin(sweep_phase(80 + 70 * u)) * (1 + 0.5 * np.sin(2 * np.pi * 9 * t)) * asym_env(n, 0.8, 1.0)
    return norm(out) + 0.35 * groan


def fx_strand_pop(n, rng, p):
    t = tarr(n)
    x = np.sin(sweep_phase(1300 - 600 * np.minimum(1, t / 0.03))) * np.exp(-t / 0.012) * np.minimum(1, t / 0.001)
    return x + 0.6 * click(n, rng, 1500, 7000, 0.002)


def fx_steam_hiss(n, rng, p):
    blast = p.get("blast", 0)
    t = tarr(n)
    x = fft_band(rng.standard_normal(n), 2500, 9000 if blast else 7000, 0.4)
    puff = 0.65 + 0.35 * fft_band(rng.standard_normal(n), None, 12, 0.5) / max(np.std(fft_band(rng.standard_normal(n), None, 12, 0.5)), 1e-9) * 0.3
    att = 0.015 if blast else 0.05
    env = np.minimum(1, t / att) * np.exp(-t / (0.35 if blast else 0.45))
    return x * env * np.clip(puff, 0.3, 1.2)


def fx_angry_pop(n, rng, p):
    t = tarr(n)
    m = int(0.16 * SR)
    tm = tarr(m)
    f = 180 * (700 / 180) ** (tm / tm[-1]) * (1 + 0.04 * np.sin(2 * np.pi * 25 * tm))
    rise = np.sin(sweep_phase(f)) * np.minimum(1, tm / 0.01) * (0.4 + 0.6 * tm / tm[-1])
    out = np.zeros(n)
    place(out, rise, 0)
    mp = int(0.2 * SR)
    place(out, fx_hole_pop(mp, rng, {}) * 1.2 + 0.5 * click(mp, rng, 3000, 9000, 0.004), 0.15 * SR)
    return out


def fx_clay_thud(n, rng, p):
    pitch, heavy = p.get("pitch", 1.0), p.get("heavy", 0)
    t = tarr(n)
    th = np.sin(sweep_phase((170 * np.exp(-t / 0.04) + 55) * pitch)) * exp_env(n, 0.10 if heavy else 0.07, 0.001)
    body = fft_band(rng.standard_normal(n), None, 500 / pitch, 0.5) * exp_env(n, 0.035, 0.0008)
    tick = click(n, rng, 800, 2500, 0.004)
    x = 1.0 * th + 0.8 * norm(body) + 0.35 * tick
    if p.get("slap"):
        x += 0.5 * click(n, rng, 600, 3500, 0.012)
    return x


def fx_clay_slap(n, rng, p):
    pitch = p.get("pitch", 1.0)
    t = tarr(n)
    sl = fft_band(rng.standard_normal(n), 600 * pitch, 4200, 0.4) * exp_env(n, 0.022, 0.0005)
    wet = np.sin(sweep_phase((320 - 140 * np.minimum(1, t / 0.05)) * pitch)) * exp_env(n, 0.045, 0.001)
    x = norm(sl) + 0.8 * wet + 0.3 * click(n, rng, 3000, 8000, 0.002)
    if p.get("thud"):
        x += 0.8 * np.sin(sweep_phase((120 * np.exp(-t / 0.05) + 50) * pitch)) * exp_env(n, 0.09, 0.001)
    return x


def fx_cloth_rustle(n, rng, p):
    x = fft_band(rng.standard_normal(n), 1500, 6500, 0.5)
    mod = fft_band(rng.standard_normal(n), 4, 22, 0.5)
    mod = np.abs(mod) / max(np.max(np.abs(mod)), 1e-9)
    t = tarr(n)
    return x * mod ** 1.3 * np.sin(np.pi * t / t[-1]) ** 0.8


def fx_paper_flutter(n, rng, p):
    t = tarr(n)
    out = np.zeros(n)
    tt = 0.0
    while tt < t[-1] - 0.02:
        L = int(rng.uniform(0.006, 0.014) * SR)
        a = rng.uniform(0.4, 1.0) * np.exp(-tt / (t[-1] * 0.6))
        place(out, a * click(L, rng, 1800, 7000, 0.004), tt * SR)
        tt += rng.uniform(0.02, 0.05) * (1 + 1.5 * tt / t[-1])
    sw = fft_band(rng.standard_normal(n), 3000, 8000, 0.4) * asym_env(n, 0.2, 1.0) * 0.15
    return norm(out) + sw


def fx_cap_whoosh(n, rng, p):
    x = swept_noise(n, rng, 2800, 500, 0.9, 1.0)
    t = tarr(n)
    x = norm(x) * (0.6 + 0.4 * np.sin(2 * np.pi * 13 * t))
    return x * asym_env(n, 0.3, 1.2)


def fx_timelapse(n, rng, p):
    cyc = int(p.get("cycles", 3))
    t = tarr(n)
    u = (t / t[-1] * cyc) % 1.0
    tri = 1 - np.abs(2 * u - 1)
    f = 400 * 4.5 ** tri
    x = np.sin(sweep_phase(f)) + 0.3 * np.sin(2 * sweep_phase(f))
    x *= 0.5 + 0.5 * np.sin(np.pi * u) ** 0.5
    shimmer = fft_band(rng.standard_normal(n), 2500, 7000, 0.4) * (0.5 + 0.5 * np.sin(2 * np.pi * 16 * t)) * 0.12
    return (x * 0.5 + shimmer) * np.minimum(1, t / 0.06) * np.minimum(1, (t[-1] - t) / 0.12)


def fx_tick(n, rng, p):
    f = p.get("freq", 2000)
    return click(n, rng, f * 0.6, f * 2.5, 0.002) + ring(n, f, 0.006)


def fx_balloon_inflate(n, rng, p):
    t = tarr(n)
    u = t / t[-1]
    f = (500 + 1000 * u) * (1 + 0.05 * np.sin(2 * np.pi * 31 * t)) * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    squeal = np.sin(sweep_phase(f)) + 0.4 * np.sin(2 * sweep_phase(f))
    pump = (0.5 + 0.5 * np.sin(2 * np.pi * 5.5 * t - 1.5)) ** 1.5
    hiss = fft_band(rng.standard_normal(n), 1200, 5000, 0.5)
    x = (squeal * 0.7 + hiss * 1.2) * pump * (0.5 + 0.5 * u)
    return x * np.minimum(1, t / 0.05) * np.minimum(1, (t[-1] - t) / 0.1)


def fx_sparks(n, rng, p):
    x = fx_sizzle(n, rng, dict(rate=1800, tau=0.07))
    t = tarr(n)
    zap = np.sin(sweep_phase(4500 * np.exp(-t / 0.02) + 800)) * np.exp(-t / 0.012)
    return norm(x) + 0.5 * zap


def fx_laser_zap(n, rng, p):
    t = tarr(n)
    f = 2600 + 3600 * np.minimum(1, t / 0.03)
    x = np.sin(sweep_phase(f)) * np.exp(-t / 0.035) * np.minimum(1, t / 0.001)
    return x + 0.3 * np.sin(2 * sweep_phase(f)) * np.exp(-t / 0.02)


def fx_probe_tick(n, rng, p):
    cnt, step = int(p.get("n", 1)), p.get("step", 0.03)
    out = np.zeros(n)
    for i in range(cnt):
        m = int(0.05 * SR)
        f = 3600 + 400 * ((i * 3) % 4)
        c = ring(m, f, 0.003) + 0.6 * ring(m, f * 1.45, 0.002) + 0.4 * click(m, rng, 2500, 9000, 0.0015)
        place(out, c, i * step * SR)
    return out


def fx_wafer_slide(n, rng, p):
    m = int(0.14 * SR)
    sl = swept_noise(m, rng, 2200, 700, 0.8, 1.0) * asym_env(m, 0.5, 1.0)
    out = np.zeros(n)
    place(out, 0.6 * norm(sl), 0)
    mc = int(0.25 * SR)
    clink = ring(mc, 2250, 0.04) + 0.7 * ring(mc, 3450, 0.025) + 0.3 * ring(mc, 5200, 0.012) + 0.5 * click(mc, rng, 1000, 6000, 0.003)
    place(out, 0.9 * clink, 0.10 * SR)
    # drawer close thunk
    place(out, 0.5 * ring(int(0.12 * SR), 140, 0.04), 0.17 * SR)
    return out


def fx_stage_clunk(n, rng, p):
    t = tarr(n)
    return ring(n, 120, 0.05) + 0.7 * ring(n, 720, 0.02) + 0.5 * click(n, rng, 400, 3000, 0.006)


def fx_trace_ping(n, rng, p):
    f = p.get("f", 1000)
    return ring(n, f, 0.20) + 0.5 * ring(n, f * 2.76, 0.10) + 0.25 * ring(n, f * 5.4, 0.04)


def fx_ding(n, rng, p):
    f = p.get("f", 1568)
    soft = p.get("soft", 0)
    x = ring(n, f, 0.45, 0.002) + 0.35 * ring(n, f * 2.0, 0.28, 0.002) + 0.15 * ring(n, f * 3.01, 0.15, 0.002)
    if not soft:
        ir_n = int(0.6 * SR)
        ir = rng.standard_normal(ir_n) * np.exp(-tarr(ir_n) / 0.15)
        ir[0] = 0
        x = x + 0.08 * convolve_ir(x[:int(0.5 * SR)], ir, n)
    return x


def fx_tray_slide(n, rng, p):
    t = tarr(n)
    u = t / t[-1]
    sc = fft_band(rng.standard_normal(n), 120, 1400, 0.5) * (0.7 + 0.3 * np.sin(2 * np.pi * 21 * t))
    sc = sc * asym_env(n, 0.7, 0.8)
    roll = np.sin(sweep_phase(55 + 30 * u)) * asym_env(n, 0.6, 1.0)
    out = norm(sc) + 0.4 * roll
    # end stop
    m = int(0.18 * SR)
    stop = ring(m, 105, 0.05) + 0.6 * ring(m, 830, 0.035) + 0.4 * click(m, rng, 500, 4000, 0.005)
    place(out, 1.1 * stop, int(0.86 * n))
    return out


def fx_noodle_spill(n, rng, p):
    t = tarr(n)
    lam = (260 * np.exp(-t / 0.55) * np.minimum(1, t / 0.25) + 25) / SR
    imp = (rng.random(n) < lam) * rng.exponential(1.0, n) * rng.choice([-1, 1], n)
    pat = fft_band(imp, 900, 5200, 0.4)
    sl = fft_band(rng.standard_normal(n), 400, 1600, 0.5) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t)) * asym_env(n, 0.3, 1.0)
    return norm(pat) * 0.9 * np.minimum(1, (t[-1] - t) / 0.2) + 0.35 * norm(sl)


def fx_slither(n, rng, p):
    t = tarr(n)
    x = fft_band(rng.standard_normal(n), 500, 2400, 0.5) * (0.5 + 0.5 * np.sin(2 * np.pi * (6 + 2 * rng.random()) * t)) ** 1.5
    return x * asym_env(n, 0.45, 1.0)


def fx_plug_jam(n, rng, p):
    m = int(0.05 * SR)
    out = np.zeros(n)
    place(out, 0.9 * fft_band(rng.standard_normal(m), 400, 3000, 0.4) * exp_env(m, 0.016, 0.0005), 0)
    t = tarr(n)
    out += 0.7 * (ring(n, 1100, 0.05) + 0.6 * ring(n, 2600, 0.03)) + 0.8 * ring(n, 90, 0.07)
    out += 0.3 * click(n, rng, 2000, 8000, 0.002)
    return out


def fx_tie_zip(n, rng, p):
    out = np.zeros(n)
    nt = 14
    tt = 0.0
    for i in range(nt):
        m = int(0.02 * SR)
        place(out, (0.4 + 0.6 * i / nt) * (click(m, rng, 1800, 6500, 0.0018) + 0.5 * ring(m, 2400, 0.003)), tt * SR)
        tt += 0.03 - 0.018 * i / nt
    m = int(0.1 * SR)
    place(out, 1.4 * (click(m, rng, 800, 5000, 0.005) + 0.8 * ring(m, 1500, 0.015) + 0.5 * ring(m, 300, 0.03)), (tt + 0.02) * SR)
    return out


def fx_whip_crack(n, rng, p):
    t = tarr(n)
    crack = fft_band(rng.standard_normal(n), 1500, 15000, 0.3) * np.exp(-t / 0.006) * np.minimum(1, t / 0.0002)
    snap = np.sin(sweep_phase(5200 * np.exp(-t / 0.004) + 1400)) * np.exp(-t / 0.008)
    low = np.sin(sweep_phase(70 + 150 * np.exp(-t / 0.02))) * exp_env(n, 0.04, 0.001) * 0.4
    dry = np.tanh(2.0 * (norm(crack) + 0.6 * snap + low))
    out = dry.copy()
    place(out, 0.35 * fft_band(dry[:int(0.1 * SR)], 800, 9000, 0.4), 0.085 * SR)
    place(out, 0.15 * fft_band(dry[:int(0.1 * SR)], 600, 6000, 0.4), 0.19 * SR)
    ir_n = int(0.5 * SR)
    ir = fft_band(rng.standard_normal(ir_n), 800, 9000, 0.5) * np.exp(-tarr(ir_n) / 0.12)
    ir[0] = 0
    out += 0.02 * convolve_ir(dry[:int(0.05 * SR)], ir, n)
    return out


def fx_flesh_thump(n, rng, p):
    t = tarr(n)
    th = np.sin(sweep_phase(120 * np.exp(-t / 0.05) + 42)) * exp_env(n, 0.11, 0.001)
    wet = fft_band(rng.standard_normal(n), 250, 1800, 0.4) * exp_env(n, 0.03, 0.0006)
    sq = fft_band(rng.standard_normal(n), None, 500, 0.5) * exp_env(n, 0.07, 0.01)
    return th + 0.8 * norm(wet) + 0.3 * norm(sq) + 0.3 * click(n, rng, 1500, 6000, 0.003)


def fx_boing(n, rng, p):
    f0 = p.get("f0", 240)
    t = tarr(n)
    f = f0 * (1 + 0.55 * np.exp(-t / 0.22) * np.sin(2 * np.pi * 8 * t + 0.3)) * (1 - 0.15 * t / t[-1])
    ph = sweep_phase(f)
    x = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sin(3 * ph)
    return x * exp_env(n, 0.28, 0.004)


def fx_squash(n, rng, p):
    t = tarr(n)
    f = (650 * np.exp(-t / 0.07) + 160) * (1 + 0.1 * np.sin(2 * np.pi * 38 * t))
    x = np.sin(sweep_phase(f)) * exp_env(n, 0.09, 0.002)
    sq = swept_noise(n, rng, 1400, 300, 0.8, 0.8) * exp_env(n, 0.06, 0.002)
    return x + 0.8 * norm(sq)


def fx_pkg_flip(n, rng, p):
    pitch = p.get("pitch", 1.0)
    m = int(0.36 * SR)
    sw = swept_noise(m, rng, 600, 2200, 0.9, 1.0) * asym_env(m, 0.6, 1.0)
    out = np.zeros(n)
    place(out, 0.5 * norm(sw), 0)
    mc = int(0.16 * SR)
    c = click(mc, rng, 1500, 7000, 0.003) + 0.8 * ring(mc, 1900 * pitch, 0.012) + 0.6 * ring(mc, 520 * pitch, 0.02)
    place(out, c, 0.35 * SR)
    return out


EFFECTS = {k[3:]: v for k, v in dict(globals()).items() if k.startswith("fx_")}


# ---------------------------------------------------------------- mixing
def pan_gains(pan, n, pan_end=None):
    p = np.full(n, float(pan)) if pan_end is None else np.linspace(pan, pan_end, n)
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.cos(a), np.sin(a)


def render_cue(c, idx):
    typ, dur = c["type"], float(c["dur"])
    p = c.get("params", {})
    n = int(round(dur * SR))
    rng = np.random.default_rng(int(p.get("seed", 0)) * 100003 + idx * 7919 + 17)
    x = EFFECTS[typ](n, rng, p)
    x = x / max(np.max(np.abs(x)), 1e-12) * c["gain"]
    if x.ndim == 1:
        gl, gr = pan_gains(c["pan"], n, p.get("pan_end"))
        x = np.stack([x * gl, x * gr], 1)
    else:
        gl, gr = pan_gains(c["pan"], n)
        x = x * np.stack([gl, gr], 1) * np.sqrt(2)
    lead = int(round(LEAD * SR))
    x = np.vstack([np.zeros((lead, 2)), x])
    fi, fo = int(FADE_IN * SR), int(FADE_OUT * SR)
    ramp_in = 0.5 - 0.5 * np.cos(np.pi * np.arange(fi) / fi)
    x[lead:lead + fi] *= ramp_in[:, None]
    ramp_out = 0.5 + 0.5 * np.cos(np.pi * np.arange(fo) / fo)
    x[-fo:] *= ramp_out[:, None]
    return x, lead


def main():
    cues = json.load(open(CUES))["cues"]
    N = int(round(TOTAL * SR))
    mix = np.zeros((N, 2))
    for i, c in enumerate(cues):
        x, lead = render_cue(c, i)
        s = int(round(c["t"] * SR)) - lead
        a = max(0, -s)
        e = min(len(x), N - s)
        if e > a:
            seg = x[a:e].copy()
            if s + e >= N:   # clipped at the end of the film: fade out
                k = min(len(seg), int(FADE_OUT * SR))
                seg[-k:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k))[:, None]
            mix[s + a:s + e] += seg
    pk = np.max(np.abs(mix))
    if pk > PEAK_LIMIT:
        print("peak %.3f above limit, scaling whole mix by %.3f" % (pk, PEAK_LIMIT / pk))
        mix *= PEAK_LIMIT / pk
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sf.write(OUT, mix.astype(np.float32), SR, subtype="PCM_24")
    pk = np.max(np.abs(mix))
    print("cues %d, length %.3f s, peak %.1f dBFS" % (len(cues), len(mix) / SR, 20 * np.log10(pk)))
    print("wrote", os.path.relpath(OUT, PROJ))


if __name__ == "__main__":
    sys.exit(main())
