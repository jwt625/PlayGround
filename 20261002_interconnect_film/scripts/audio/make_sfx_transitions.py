"""Procedural SFX for the six oscilloscope scene transitions (DevLog/v1/DevLog-005-transitions.md). numpy + soundfile only, deterministic.

Run (repo root):
  uv run --no-project --with numpy --with soundfile python scripts/audio/make_sfx_transitions.py [--write-cues]
Reads scripts/audio/sfx_cues_transitions.json (same schema as sfx_cues.json), writes outputs/audio/sfx_transitions_20261002.wav
(48 kHz stereo, 62.0 s, mostly silent, peak <= 0.7). --write-cues first regenerates the JSON from the design constants below.
The existing sfx_cues.json / make_sfx.py are not touched: mix this track together with sfx_20261002.wav and the VO.
Cue time t = film seconds of the main transient (continuous effects: start). Transition k occupies film time [10k - 0.5, 10k + 0.5);
clip frame i is at t = 10k - 0.5 + i / 30 and u = i / 29 is the normalised clip time used in transitions.py (SPEC).
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
CUES = os.path.join(HERE, "sfx_cues_transitions.json")
OUT = os.path.join(PROJ, "outputs", "audio", "sfx_transitions_20261002.wav")
SR, TOTAL, LEAD, PEAK = 48000, 62.0, 0.004, 0.70


# ------------------------------------------------------------------ cue table (design constants; u -> t = b - 0.5 + u * 29 / 30)
def T(b, u):
    return round(b - 0.5 + u * 29.0 / 30.0, 3)


def design():
    C = []

    def add(t, typ, dur, gain, note, pan=0.0, **params):
        C.append(dict(t=t, type=typ, dur=dur, gain=gain, pan=pan, note=note, params=params))

    for k, b in enumerate((10, 20, 30, 40, 50, 60), 1):
        tag = "T%d (S%d->S%d)" % (k, k, k + 1)
        add(T(b, 0.04), "crt_degauss", 0.25, 0.10 if k != 3 else 0.14, tag + ": CRT effect starts (scanlines, vignette), degauss thump")
    # sweeps (trace sweep line wiping picture N out and picture N+1 in): u windows from transitions.SPEC
    sw = {1: ((0.14, 0.40), (0.56, 0.84)), 2: ((0.14, 0.40), (0.64, 0.88)), 3: ((0.14, 0.38), (0.60, 0.84)),
          4: ((0.14, 0.40), (0.56, 0.84)), 5: ((0.12, 0.36), (0.60, 0.84)), 6: (None, (0.52, 0.78))}
    for k, b in enumerate((10, 20, 30, 40, 50, 60), 1):
        for j, w in enumerate(sw[k]):
            if w is None:
                continue
            a, e = w
            f0, f1 = (500, 2600) if j == 0 else (700, 2800)
            if k == 6:
                f0, f1 = 400, 1500
            add(T(b, a), "sweep_tone", round((e - a) * 29 / 30, 3), 0.045 if k != 6 else 0.035,
                "T%d sweep %d: trace sweep line crosses the screen (rising glide)" % (k, j + 1), pan=-0.3 if j == 0 else 0.3, f0=f0, f1=f1)
    # T1: three retimer steps (eye opens), knob clicks + blips at the eye step times (u = 0.24 + 0.52 * (th + 0.03), th = 0.12, 0.40, 0.68)
    for n, (u, f) in enumerate(((0.318, 880), (0.464, 1175), (0.609, 1568)), 1):
        add(T(10, u), "knob_click", 0.04, 0.09, "T1 retimer step %d: horizontal scale knob click" % n, pan=0.35)
        add(T(10, u) + 0.004, "scope_blip", 0.22, 0.07, "T1 retimer step %d: eye opens one step, BER readout blip (pitch rises per step)" % n, f=f, tau=0.07)
    add(T(10, 0.10), "hsync_roll", 0.10, 0.07, "T1 horizontal-sync roll", f0=140, f1=70)
    for i in (3, 4, 12, 13, 17):
        add(round(9.5 + i / 30.0, 3), "static_burst", 0.06, 0.05, "T1 trigger flicker / tear glitch frame %d" % i)
    # T2: curves collapse, XY button, package outline plotted in XY mode
    add(T(20, 0.28), "sweep_tone", 0.10, 0.05, "T2 curves collapse to a dot (falling glide)", f0=1500, f1=300)
    add(round(20 - 0.5 + 11 / 30.0, 3), "knob_click", 0.04, 0.10, "T2 XY / autoset button press (key frames 11..13)", pan=0.35)
    add(T(20, 0.39), "plot_scribble", 0.23, 0.06, "T2 XY mode plots substrate, ASIC, optical engines, laser diode and beam (sample-and-hold tone)", pan=0.0)
    add(T(20, 0.45), "hsync_roll", 0.10, 0.06, "T2 horizontal-sync roll", f0=120, f1=60)
    for i in (2, 5, 15):
        add(round(19.5 + i / 30.0, 3), "static_burst", 0.06, 0.05, "T2 glitch frame %d" % i)
    # T3: heat spikes (9 spikes, spike j appears at u = 0.297 + 0.0416 j), scope jolt
    add(T(30, 0.28), "heat_sizzle", 0.45, 0.07, "T3 rising sizzle under the heat trace (filtered noise, centre frequency rises)")
    for j in range(9):
        add(T(30, 0.297 + 0.0416 * j), "scope_zap", 0.12, round(0.05 + 0.012 * j, 3), "T3 temperature spike %d (pitch and level rise per spike)" % (j + 1),
            pan=round(-0.5 + 0.125 * j, 3), f0=1800 + 150 * j)
    add(T(30, 0.62), "hsync_roll", 0.10, 0.07, "T3 horizontal-sync roll", f0=160, f1=60)
    # T4: six ring dips (dip k appears at u = 0.24 + 0.52 * (0.1417 k + 0.08)); dip index 2 lands in the pass window
    for kk in range(6):
        u = 0.24 + 0.52 * (0.1417 * kk + 0.08)
        add(T(40, u), "dip_ping", 0.35, 0.07 if kk != 2 else 0.10, "T4 ring resonance dip %d appears%s" % (kk + 1, " (pass window)" if kk == 2 else ""),
            pan=round(-0.6 + 0.24 * kk, 3), f=1760 if kk == 2 else 1100 - 70 * kk)
    add(round(40 - 0.5 + 13 / 30.0, 3), "knob_ratchet", 0.12, 0.07, "T4 horizontal position knob twist (two turns, key frames 13..17)", pan=0.35, n=7)
    add(T(40, 0.08), "hsync_roll", 0.08, 0.06, "T4 horizontal-sync roll", f0=130, f1=70)
    # T5: dips deepen, rotate into fibre lines
    add(T(50, 0.30), "hsync_roll", 0.12, 0.08, "T5 heavy horizontal-sync roll #1 (dips deepen into vertical lines)", f0=170, f1=60)
    add(T(50, 0.48), "hsync_roll", 0.08, 0.06, "T5 horizontal-sync roll #2", f0=140, f1=60)
    add(T(50, 0.44), "fiber_shimmer", 0.45, 0.06, "T5 six lines rotate flat and turn fibre cyan (six detuned partials swell)")
    for i, p in ((12, -0.4), (16, 0.2), (20, 0.5)):
        add(round(49.5 + i / 30.0, 3), "probe_flick", 0.12, 0.05, "T5 probe cable whip (key frames 12..24)", pan=p)
    for i in (6, 7, 8, 14, 20):
        add(round(49.5 + i / 30.0, 3), "static_burst", 0.06, 0.05, "T5 glitch frame %d" % i)
    # T6: CRT power-off, flatline beep, dot, black, card sweeps in; existing ding at 60.0 (sfx_cues.json) stays
    add(T(60, 0.04), "crt_squash", 0.22, 0.10, "T6 picture squashes to a horizontal line (pitch drop)")
    add(round(59.5 + 9 / 30.0, 3), "knob_click", 0.04, 0.10, "T6 power button press (key frames 7..14)", pan=0.35)
    add(T(60, 0.26), "flatline_beep", 0.23, 0.07, "T6 FLATLINE: continuous 1 kHz beep while the line is on screen; cut at 59.98 so the ding at 60.0 lands clean", f=1000)
    add(T(60, 0.40), "dot_pip", 0.12, 0.08, "T6 line collapses to a dot (CRT off pip); LED off at key frame 12")
    return C


# ------------------------------------------------------------------ DSP helpers
def tt(n):
    return np.arange(n) / SR


def bandnoise(rng, n, lo, hi):
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= ((f >= lo) & (f <= hi))
    return np.fft.irfft(X, n)


def expenv(n, tau, att=0.001):
    t = tt(n)
    return np.exp(-t / tau) * np.minimum(1.0, t / att)


def glide(n, f0, f1, harm=(1.0,), amps=(1.0,)):
    t = tt(n)
    f = f0 * (f1 / f0) ** (t / max(t[-1], 1e-9))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return sum(a * np.sin(h * ph) for h, a in zip(harm, amps))


def fx_crt_degauss(n, rng, p):
    t = tt(n)
    thump = np.sin(2 * np.pi * 52 * t) * np.exp(-t / 0.05)
    buzz = (np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)) * np.exp(-t / 0.08) * 0.4
    zip_ = bandnoise(rng, n, 2000, 9000) * np.exp(-t / 0.02) * 0.3
    return thump + buzz + zip_


def fx_sweep_tone(n, rng, p):
    x = glide(n, p["f0"], p["f1"], harm=(1, 2), amps=(1.0, 0.25))
    t = tt(n)
    env = np.sin(np.pi * np.clip(t / t[-1], 0, 1)) ** 1.5
    return x * env + 0.05 * bandnoise(rng, n, 3000, 9000) * env


def fx_scope_blip(n, rng, p):
    t = tt(n)
    return np.sin(2 * np.pi * p["f"] * t) * expenv(n, p.get("tau", 0.07), 0.002) + 0.2 * np.sin(2 * np.pi * 2 * p["f"] * t) * expenv(n, 0.03, 0.002)


def _click(n, rng, at, lo=2000, hi=6000, tau=0.0012):
    x = np.zeros(n)
    m = min(int(0.02 * SR), n - at)
    if m > 0:
        x[at:at + m] = bandnoise(rng, m, lo, hi) * np.exp(-tt(m) / tau)
    return x


def fx_knob_click(n, rng, p):
    x = _click(n, rng, 0) + 0.7 * _click(n, rng, int(0.011 * SR), 1500, 4500)
    return x + 0.4 * np.sin(2 * np.pi * 180 * tt(n)) * expenv(n, 0.01, 0.0005)


def fx_knob_ratchet(n, rng, p):
    x = np.zeros(n)
    k = int(p.get("n", 6))
    for j in range(k):
        at = int(j * n / (k + 0.5))
        x += (0.6 + 0.4 * (j % 2)) * (_click(n, rng, at) + 0.7 * _click(n, rng, at + int(0.008 * SR), 1500, 4500))
    return x


def fx_hsync_roll(n, rng, p):
    t = tt(n)
    f = np.linspace(p["f0"], p["f1"], n)
    saw = 2 * ((np.cumsum(f) / SR) % 1.0) - 1
    am = 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 37 * t))
    return (saw * am * 0.6 + bandnoise(rng, n, 200, 3000) * 0.4) * np.sin(np.pi * np.clip(t / t[-1], 0, 1)) ** 0.5


def fx_static_burst(n, rng, p):
    return bandnoise(rng, n, 800, 12000) * expenv(n, 0.02, 0.0005)


def fx_plot_scribble(n, rng, p):
    steps = max(int(n / (0.012 * SR)), 2)
    r = np.random.default_rng(7)
    seq = np.exp(r.uniform(np.log(300), np.log(1800), steps))
    f = np.repeat(seq, int(np.ceil(n / steps)))[:n]
    ph = 2 * np.pi * np.cumsum(f) / SR
    sq = np.sign(np.sin(ph)) * 0.5 + np.sin(ph) * 0.5
    t = tt(n)
    return sq * np.minimum(1, t / 0.01) * np.minimum(1, (t[-1] - t) / 0.02)


def fx_heat_sizzle(n, rng, p):
    t = tt(n)
    u = t / t[-1]
    x = np.zeros(n)
    w = rng.standard_normal(n)
    X = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1 / SR)
    for lo in np.linspace(1500, 6500, 8):
        pass
    # centre rises over time: blend four fixed bands with time weights
    bands = [(1000, 2500), (2000, 4500), (3500, 7000), (5000, 11000)]
    for j, (lo, hi) in enumerate(bands):
        m = np.fft.irfft(X * ((f >= lo) & (f <= hi)), n)
        wgt = np.exp(-((u - (j + 0.5) / len(bands)) / 0.28) ** 2)
        x += m * wgt
    crackle = (rng.random(n) < 0.0015).astype(float)
    x += np.convolve(crackle, expenv(int(0.004 * SR), 0.0008), mode="same") * 6
    return x * np.sin(np.pi * np.clip(u, 0, 1)) ** 0.7


def fx_scope_zap(n, rng, p):
    t = tt(n)
    z = glide(n, p["f0"], p["f0"] * 0.18, harm=(1, 3), amps=(1.0, 0.3)) * expenv(n, 0.05, 0.0008)
    return z + 0.5 * bandnoise(rng, n, 3000, 10000) * expenv(n, 0.012, 0.0004)


def fx_dip_ping(n, rng, p):
    t = tt(n)
    f = p["f"] * (1 - 0.18 * (1 - np.exp(-t / 0.05)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    return (np.sin(ph) + 0.35 * np.sin(2.76 * ph) * np.exp(-t / 0.05)) * expenv(n, 0.11, 0.002)


def fx_fiber_shimmer(n, rng, p):
    t = tt(n)
    u = t / t[-1]
    base = 1500.0
    x = np.zeros(n)
    for j, r in enumerate((1.0, 1.25, 1.5, 1.875, 2.25, 2.5)):
        d = 1.0 + 0.003 * (j - 2.5)
        x += np.sin(2 * np.pi * base * r * d * t + j) * (0.6 + 0.4 * np.sin(2 * np.pi * (5 + j) * t))
    return x * np.sin(np.pi * np.clip(u, 0, 1)) ** 2


def fx_probe_flick(n, rng, p):
    t = tt(n)
    return bandnoise(rng, n, 1500, 7000) * np.sin(np.pi * np.clip(t / t[-1], 0, 1)) ** 2 * (0.5 + 0.5 * t / t[-1])


def fx_crt_squash(n, rng, p):
    t = tt(n)
    x = glide(n, 900, 120, harm=(1, 2), amps=(1.0, 0.3)) * np.exp(-t / 0.15)
    return x + 0.25 * bandnoise(rng, n, 1000, 6000) * np.exp(-t / 0.06)


def fx_flatline_beep(n, rng, p):
    t = tt(n)
    x = np.sin(2 * np.pi * p["f"] * t) + 0.12 * np.sin(2 * np.pi * 3 * p["f"] * t)
    return x * np.minimum(1, t / 0.004) * np.minimum(1, (t[-1] - t) / 0.004)


def fx_dot_pip(n, rng, p):
    t = tt(n)
    return (np.sin(2 * np.pi * 3500 * t) * expenv(n, 0.05, 0.0005) + 0.5 * np.sin(2 * np.pi * 90 * t) * expenv(n, 0.06, 0.001)
            + 0.3 * bandnoise(rng, n, 2000, 9000) * expenv(n, 0.004, 0.0003))


FX = {k[3:]: v for k, v in globals().items() if k.startswith("fx_")}


def render(cues):
    buf = np.zeros((int(TOTAL * SR), 2))
    for i, c in enumerate(cues):
        n = max(int(c["dur"] * SR), 16)
        rng = np.random.default_rng(1000 + i)
        x = FX[c["type"]](n, rng, c.get("params", {}))
        pk = np.max(np.abs(x))
        x = x / pk * c["gain"] if pk > 0 else x
        fi, fo = int(0.002 * SR), int(0.006 * SR)
        x[:fi] *= np.linspace(0, 1, fi)
        x[-fo:] *= np.linspace(1, 0, fo)
        a = (c["pan"] + 1) * np.pi / 4
        s = int(round((c["t"] - LEAD) * SR))
        e = min(s + n, len(buf))
        if s < 0 or e <= s:
            continue
        buf[s:e, 0] += x[:e - s] * np.cos(a)
        buf[s:e, 1] += x[:e - s] * np.sin(a)
    pk = np.max(np.abs(buf))
    if pk > PEAK:
        buf *= PEAK / pk
    return buf, pk


def main():
    if "--write-cues" in sys.argv:
        d = dict(version=1, sample_rate=SR, length_s=TOTAL,
                 units="t = film seconds of the hit instant (continuous effects: start); gain = linear peak amplitude of the normalised effect; pan -1 left .. 1 right. "
                       "Types are defined in scripts/audio/make_sfx_transitions.py (not in make_sfx.py).",
                 cues=sorted(design(), key=lambda c: c["t"]))
        json.dump(d, open(CUES, "w"), indent=1)
        print("WROTE", CUES, len(d["cues"]), "cues")
    cues = json.load(open(CUES))["cues"]
    buf, pk = render(cues)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sf.write(OUT, buf.astype(np.float32), SR, subtype="PCM_16")
    print("WROTE", OUT, "peak before limit %.3f" % pk, "cues", len(cues))


if __name__ == "__main__":
    main()
