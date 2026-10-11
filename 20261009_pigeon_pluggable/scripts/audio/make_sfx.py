"""Procedural SFX track: every effect is synthesised in numpy (no samples, no downloads, deterministic per event).

Run (project root): uv run --no-project --with numpy --with soundfile python scripts/audio/make_sfx.py [--events FILE] [--overrides FILE] [--supplement] [--list-kinds] [--dry-run]
Input: scripts/audio/scene_events.json (written by the scene build, S) if present, else
scripts/audio/sfx_events_provisional.json (A's cue list from DevLog-006 rev 3 + narration.json). Output:
outputs/audio/sfx_<date>.wav (48 kHz stereo, 62.0 s, PCM 24, peak <= -3 dBFS).

Event schema (list, or {"events": [...]}) : t (film s of the hit / start), kind (KINDS below), name (free text),
optional gain (linear multiplier on the kind default, default 1), pan (-1..1), dur (s, continuous kinds),
pos (accepted, unused), params (kind-specific). Unknown kinds are logged and skipped. Events of one kind closer than
the kind's min gap are thinned (first kept), so dense crowd emitters stay readable.
Several helpers are adapted from the sibling film's make_sfx.py (fft_band, swept_noise, ring, click, pan_gains).
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import HERE, OUT_AUDIO, SR, TODAY, TOTAL, Log, rel  # noqa: E402

LEAD = 0.005
FADE_IN, FADE_OUT = 0.003, 0.010
PEAK_LIMIT = 0.708   # -3 dBFS


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


def swept_noise(n, rng, f0, f1, bw=1.0, curve=1.0, nb=20):
    """Noise whose spectral centre moves from f0 to f1 (Hz, log), band width bw octaves (gaussian), via a filter bank."""
    t = tarr(n)
    u = np.clip(t / max(t[-1], 1e-9), 0, 1) ** curve
    lfc = np.log2(f0) + (np.log2(f1) - np.log2(f0)) * u
    cs = np.logspace(np.log10(60), np.log10(14000), nb)
    w = rng.standard_normal(n)
    out = np.zeros(n)
    for c in cs:
        out += fft_band(w, c / 2 ** 0.25, c * 2 ** 0.25, 0.15) * np.exp(-0.5 * ((np.log2(c) - lfc) / bw) ** 2)
    return out


def asym_env(n, peak, power=1.0):
    u = np.linspace(0, 1, n)
    e = np.where(u < peak, np.sin(0.5 * np.pi * u / max(peak, 1e-6)) ** 2,
                 np.cos(0.5 * np.pi * (u - peak) / max(1 - peak, 1e-6)) ** 2)
    return e ** power


def ring(n, f, tau, att=0.0005):
    t = tarr(n)
    return np.sin(2 * np.pi * f * t) * np.exp(-t / tau) * np.minimum(1.0, t / att)


def click(n, rng, lo, hi, tau):
    return fft_band(rng.standard_normal(n), lo, hi, 0.3) * exp_env(n, tau, 0.0002)


def fade_edges(x, fi, fo):
    n = len(x)
    fi, fo = min(int(fi * SR), n // 2), min(int(fo * SR), n // 2)
    if fi:
        x[:fi] *= np.linspace(0, 1, fi)[:, None] if x.ndim == 2 else np.linspace(0, 1, fi)
    if fo:
        x[-fo:] *= np.linspace(1, 0, fo)[:, None] if x.ndim == 2 else np.linspace(1, 0, fo)
    return x


def tone(n, f, harm=(1.0,), vib=0.0, vib_hz=5.0):
    t = tarr(n)
    ff = np.broadcast_to(np.asarray(f, dtype=float), (n,)) * (1 + vib * np.sin(2 * np.pi * vib_hz * t))
    ph = sweep_phase(ff)
    return sum(a * np.sin((k + 1) * ph) for k, a in enumerate(harm))


def servo(n, rng, f0, f1, buzz=0.5):
    """small DC servo / gear motor: pitched whir with gear-tooth buzz, pitch glides f0 -> f1, soft ends."""
    t = tarr(n)
    f = f0 + (f1 - f0) * np.clip(t / max(t[-1], 1e-9), 0, 1) ** 0.7
    ph = sweep_phase(f)
    x = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.35 * np.sign(np.sin(3 * ph)) * buzz
    x += 0.4 * fft_band(rng.standard_normal(n), 1500, 6000, 0.4) * (0.5 + 0.5 * np.sin(sweep_phase(f / 4)))
    return x * asym_env(n, 0.25, 0.8)


# ---------------------------------------------------------------- effects: (n, rng, params) -> (n,) or (n, 2)
def fx_wing_flap(n, rng, p):
    """one pigeon wingbeat: downstroke air push (low-mid noise thump) + feather flutter ripple."""
    out = np.zeros(n)
    m = min(n, int(0.16 * SR))
    push = fft_band(rng.standard_normal(m), 150, 1800, 0.5) * asym_env(m, 0.3, 1.5)
    flutter = fft_band(rng.standard_normal(m), 1500, 6000, 0.4) * (0.5 + 0.5 * np.sin(2 * np.pi * 70 * tarr(m))) ** 2
    place(out, norm(push) + 0.35 * norm(flutter) * asym_env(m, 0.4), 0)
    return out


def fx_flight(n, rng, p):
    """continuous flapping flight: wingbeats at `rate` Hz (pigeon about 5-8 Hz), slight jitter, gliding gaps."""
    rate = p.get("rate", 7.0)
    out = np.zeros(n)
    t = 0.0
    k = 0
    m = int(0.16 * SR)
    while t * SR < n - 10:
        g = 1.0 if (k % 9) < 7 else 0.35          # every 9 beats: two soft (glide) beats
        place(out, g * fx_wing_flap(m, rng, p), t * SR)
        t += 1.0 / rate * rng.uniform(0.9, 1.1)
        k += 1
    air = fft_band(rng.standard_normal(n), 300, 3000, 0.6) * 0.06
    return fade_edges(out + air, 0.05, 0.15)


def fx_takeoff(n, rng, p):
    """wing claps (pigeons clap the wings over the back at take-off) then a flurry."""
    out = np.zeros(n)
    for i, dt in enumerate((0.0, 0.11, 0.21)):
        m = int(0.06 * SR)
        place(out, (1 - 0.15 * i) * (click(m, rng, 600, 5000, 0.006) + 0.5 * ring(m, 420, 0.01)), dt * SR)
    rest = n - int(0.3 * SR)
    if rest > 0:
        place(out, 0.7 * fx_flight(rest, rng, {"rate": 9.0}), 0.3 * SR)
    return out


def fx_landing_thump(n, rng, p):
    """cyber pigeon lands: soft body thump + talon taps on metal + a feather settle."""
    t = tarr(n)
    thump = np.sin(sweep_phase(70 + 80 * np.exp(-t / 0.03))) * exp_env(n, 0.07, 0.002)
    body = fft_band(rng.standard_normal(n), 80, 700, 0.5) * exp_env(n, 0.05, 0.001)
    out = 1.0 * thump + 0.5 * body
    for dt in (0.012, 0.035):
        m = int(0.08 * SR)
        place(out, 0.35 * (click(m, rng, 2500, 8000, 0.002) + 0.6 * ring(m, 3900, 0.012)), dt * SR)
    place(out, 0.25 * fx_wing_flap(int(0.16 * SR), rng, p), 0.08 * SR)
    return out


def fx_head_servo(n, rng, p):
    f0 = p.get("f0", rng.uniform(700, 1000))
    return servo(n, rng, f0, f0 * p.get("ratio", rng.uniform(1.15, 1.45)), 0.4)


def fx_blink(n, rng, p):
    """eyelid shutter (TX disable): fast close-open, two dry ticks and a tiny whir."""
    out = np.zeros(n)
    for dt, f in ((0.0, 3100), (0.09, 2600)):
        m = int(0.05 * SR)
        place(out, click(m, rng, 2000, 9000, 0.0015) + 0.5 * ring(m, f, 0.006), dt * SR)
    m = int(0.1 * SR)
    place(out, 0.25 * servo(m, rng, 1800, 2200, 0.2), 0.0)
    return out


def fx_eye_glow_on(n, rng, p):
    t = tarr(n)
    f = 180 * 4 ** np.clip(t / (0.7 * t[-1]), 0, 1)
    x = tone(n, f, (1.0, 0.4, 0.2)) * asym_env(n, 0.55, 1.0)
    sh = fft_band(rng.standard_normal(n), 4000, 9000, 0.4) * asym_env(n, 0.7, 2.0)
    return norm(x) + 0.12 * norm(sh)


def fx_eye_glow_off(n, rng, p):
    t = tarr(n)
    f = 700 * 0.25 ** np.clip(t / t[-1], 0, 1)
    return tone(n, f, (1.0, 0.3)) * exp_env(n, 0.12, 0.005)


def fx_pad_charge_hum(n, rng, p):
    """landing pad charging: mains-ish hum + rising charge whine + contact crackle at the start."""
    t = tarr(n)
    hum = tone(n, 100, (1.0, 0.6, 0.35, 0.2), vib=0.003)
    whine = tone(n, 1200 + 1400 * np.clip(t / t[-1], 0, 1) ** 0.5, (1.0, 0.2)) * 0.25
    out = (hum + whine) * asym_env(n, 0.2, 0.7)
    cr = fft_band((rng.random(n) < 0.004) * rng.standard_normal(n), 2000, 9000, 0.4) * exp_env(n, 0.15)
    return norm(out) + 0.4 * norm(cr)


def fx_cell_dock(n, rng, p):
    """back into the pigeonhole: short slide + latch kachunk."""
    out = np.zeros(n)
    m = int(0.18 * SR)
    place(out, 0.4 * norm(swept_noise(m, rng, 800, 2500, 0.8)) * asym_env(m, 0.6), 0)
    m2 = int(0.15 * SR)
    t2 = tarr(m2)
    ka = click(m2, rng, 1200, 7000, 0.004) + 0.7 * ring(m2, 1650, 0.02) + 0.6 * np.sin(sweep_phase(160 + 120 * np.exp(-t2 / 0.01))) * exp_env(m2, 0.04)
    place(out, ka, 0.17 * SR)
    place(out, 0.5 * click(int(0.05 * SR), rng, 2000, 8000, 0.002), 0.24 * SR)
    return out


def fx_cell_undock(n, rng, p):
    """release click then slide out."""
    out = np.zeros(n)
    m = int(0.08 * SR)
    place(out, click(m, rng, 1500, 8000, 0.003) + 0.5 * ring(m, 2100, 0.01), 0)
    m2 = int(0.28 * SR)
    place(out, 0.45 * norm(swept_noise(m2, rng, 2600, 900, 0.8)) * asym_env(m2, 0.3), 0.06 * SR)
    return out


def fx_beak_latch(n, rng, p):
    """beak snaps shut on the cage lip: hard metal snap + short ring + spring."""
    m = n
    x = 1.0 * click(m, rng, 1800, 9000, 0.0025) + 0.6 * ring(m, 3300, 0.03) + 0.35 * ring(m, 5150, 0.018)
    x += 0.4 * ring(m, 900, 0.015)
    return x


def fx_beak_open(n, rng, p):
    """latch release: softer click + small spring twang."""
    t = tarr(n)
    x = 0.7 * click(n, rng, 1200, 6000, 0.003)
    tw = np.sin(sweep_phase(520 + 180 * np.exp(-t / 0.05))) * exp_env(n, 0.09, 0.002) * (1 + 0.3 * np.sin(2 * np.pi * 28 * t))
    return x + 0.5 * tw


def _contact(n, rng, f, zap):
    x = click(n, rng, 1500, 9000, 0.0015) + 0.7 * ring(n, f, 0.012) + 0.25 * ring(n, f * 2.03, 0.006)
    if zap:
        z = fft_band(rng.standard_normal(n), 2500, 9000, 0.4) * exp_env(n, 0.03, 0.002) * (rng.random(n) < 0.35)
        x = x + zap * norm(z)
    return x


def fx_contact_gnd(n, rng, p):
    return _contact(n, rng, 1250, 0.0)


def fx_contact_pwr(n, rng, p):
    return _contact(n, rng, 1750, 0.5)


def fx_contact_sig(n, rng, p):
    out = np.zeros(n)
    for i in range(3):                                # signal pads: a quick ripple of three ticks
        m = int(0.06 * SR)
        place(out, (1 - 0.2 * i) * _contact(m, rng, 2500 + 150 * i, 0.15), i * 0.022 * SR)
    return out


def fx_leash_pull(n, rng, p):
    """tail strap pulled: strap zip + fabric creak."""
    x = swept_noise(n, rng, 600, 2400, 0.7, 0.8) * asym_env(n, 0.5, 1.0)
    cr = fft_band((rng.random(n) < 0.02) * rng.standard_normal(n), 800, 4000, 0.4)
    return norm(x) + 0.3 * norm(cr) * asym_env(n, 0.5)


def fx_whoosh(n, rng, p):
    x = swept_noise(n, rng, p.get("f0", 300), p.get("f1", 2500), p.get("bw", 1.0), 1.3)
    return norm(x) * asym_env(n, p.get("peak", 0.6), 1.2)


def fx_whip(n, rng, p):
    x = swept_noise(n, rng, 500, 4500, 0.7, 0.6)
    return norm(x) * asym_env(n, 0.35, 2.0)


def fx_fill_whoosh(n, rng, p):
    """push into the eye: swelling whoosh + tonal shimmer that lands on the fill colour (red lower, cyan higher)."""
    base = {"red": 220.0, "cyan": 330.0}.get(p.get("color", "red"), 220.0)
    t = tarr(n)
    w = norm(swept_noise(n, rng, 250, 5000, 1.0, 1.6)) * asym_env(n, 0.75, 1.3)
    f = base * 2 ** (1.5 * np.clip(t / t[-1], 0, 1))
    sh = tone(n, f, (1.0, 0.5, 0.33, 0.25), vib=0.004, vib_hz=7) * asym_env(n, 0.8, 1.5)
    return w + 0.35 * norm(sh)


def fx_elec_hum(n, rng, p):
    """close-up electronics bed (PIC / TIA): mains buzz + coil whine + faint data hiss, slow swell."""
    f = p.get("f", 120.0)
    t = tarr(n)
    buzz = tone(n, f, (1.0, 0.8, 0.5, 0.45, 0.3, 0.25), vib=0.002)
    whine = tone(n, p.get("whine", 7400.0), (1.0,), vib=0.001, vib_hz=0.7) * 0.12
    hiss = fft_band(rng.standard_normal(n), 3000, 12000, 0.4) * 0.04 * (1 + 0.5 * np.sin(2 * np.pi * 0.8 * t))
    x = norm(buzz) + whine + hiss
    return fade_edges(x, 0.35, 0.5)


def fx_laser_on(n, rng, p):
    """DFB laser turns on: soft zap + rising whine that settles."""
    t = tarr(n)
    f = 400 + 2600 * (1 - np.exp(-t / 0.12))
    whine = tone(n, f, (1.0, 0.25)) * asym_env(n, 0.2, 0.8)
    zap = fft_band(rng.standard_normal(n), 2000, 9000, 0.4) * exp_env(n, 0.03, 0.001)
    return norm(whine) + 0.6 * norm(zap)


def fx_pam4_chatter(n, rng, p):
    """8 Mach-Zehnder modulators blinking PAM4: 4-level stepped tones at symbol bursts (stylised, far below real rates)."""
    sym = p.get("sym", 0.03)
    levels = np.array([880.0, 1175.0, 1480.0, 1760.0])
    k = int(np.ceil(n / (sym * SR)))
    seq = levels[rng.integers(0, 4, k)]
    f = np.repeat(seq, int(sym * SR))[:n]
    f = np.pad(f, (0, n - len(f)), mode="edge")
    x = np.sign(np.sin(sweep_phase(f))) * 0.5 + np.sin(sweep_phase(f))
    x = fft_band(x, 300, 6000, 0.5)
    gate = np.repeat((rng.random(k) < 0.8).astype(float), int(sym * SR))[:n]
    gate = np.pad(gate, (0, n - len(gate)))
    gate = np.convolve(gate, np.ones(48) / 48, mode="same")
    return fade_edges(norm(x) * gate, 0.08, 0.15)


def fx_pd_flash(n, rng, p):
    t = tarr(n)
    f = p.get("f", rng.uniform(2200, 3200))
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * exp_env(n, 0.018, 0.001)


def fx_beam_launch(n, rng, p):
    """8 collimated beams leave the lenslets: bright 'shing' (8 detuned partials) with a short air swell."""
    t = tarr(n)
    x = sum(np.sin(2 * np.pi * 2100 * (1 + 0.012 * k) * t + k) for k in range(8)) / 8
    x *= exp_env(n, 0.22, 0.01)
    air = fft_band(rng.standard_normal(n), 3000, 11000, 0.4) * asym_env(n, 0.1, 1.5)
    return norm(x) + 0.3 * norm(air)


def fx_i2c_stream(n, rng, p):
    """telemetry over I2C (slow control): soft clocked chirps, bytes of 8 ticks with gaps."""
    out = np.zeros(n)
    bit, t = 0.012, 0.0
    while t * SR < n - 600:
        for b in range(9):
            if rng.random() < 0.6:
                m = int(0.01 * SR)
                place(out, ring(m, 1600 if rng.random() < 0.5 else 2100, 0.003), (t + b * bit) * SR)
        t += 9 * bit + rng.uniform(0.02, 0.06)
    return fade_edges(out, 0.05, 0.1)


def fx_hud_lock(n, rng, p):
    """POV target box locks on the Subject: two beeps then a held lock tone."""
    out = np.zeros(n)
    for dt in (0.0, 0.12):
        m = int(0.07 * SR)
        place(out, tone(m, 1400, (1.0, 0.3)) * exp_env(m, 0.03, 0.002), dt * SR)
    m = n - int(0.26 * SR)
    if m > 0:
        place(out, tone(m, 1870, (1.0, 0.25)) * asym_env(m, 0.1, 0.6), 0.26 * SR)
    return out


def fx_hud_tick(n, rng, p):
    t = tarr(n)
    f = p.get("f", 2400)
    return (np.sin(2 * np.pi * f * t) + 0.4 * click(n, rng, 3000, 9000, 0.001)) * exp_env(n, 0.012, 0.0005)


def fx_explode(n, rng, p):
    """parts float apart: airy whoosh + scattered small clicks (screws, shields) + low glide."""
    w = norm(swept_noise(n, rng, 400, 2000, 1.0, 0.8)) * asym_env(n, 0.35, 1.2)
    out = 0.8 * w
    for _ in range(14):
        m = int(0.05 * SR)
        place(out, rng.uniform(0.2, 0.5) * (click(m, rng, 2000, 8000, 0.002) + 0.5 * ring(m, rng.uniform(1800, 4200), 0.008)),
              rng.uniform(0.05, 0.6) * n)
    t = tarr(n)
    out += 0.3 * np.sin(sweep_phase(120 * 0.6 ** (t / t[-1]))) * asym_env(n, 0.3)
    return out


def fx_snap_back(n, rng, p):
    """parts reassemble: reverse swell into a solid clunk."""
    out = np.zeros(n)
    m = int(0.75 * n)
    place(out, 0.7 * norm(swept_noise(m, rng, 2000, 400, 1.0, 1.0)) * asym_env(m, 0.85, 1.5), 0)
    m2 = n - m
    t2 = tarr(m2)
    cl = click(m2, rng, 800, 6000, 0.004) + 0.8 * np.sin(sweep_phase(140 + 90 * np.exp(-t2 / 0.01))) * exp_env(m2, 0.06)
    place(out, cl, m)
    return out


def fx_robot_servo(n, rng, p):
    return servo(n, rng, p.get("f0", 320), p.get("f1", 520), 0.8)


def fx_conveyor(n, rng, p):
    """assembly line: belt rumble + roller clacks at a steady rhythm."""
    t = tarr(n)
    rumble = fft_band(rng.standard_normal(n), 40, 300, 0.5) * (1 + 0.2 * np.sin(2 * np.pi * 3 * t))
    out = norm(rumble) * 0.6
    period = p.get("period", 0.25)
    k = 0
    while k * period * SR < n - 2000:
        m = int(0.04 * SR)
        place(out, 0.5 * (click(m, rng, 1000, 5000, 0.003) + 0.4 * ring(m, 780, 0.01)), k * period * SR)
        k += 1
    return fade_edges(out, 0.2, 0.3)


def fx_assembly_clunk(n, rng, p):
    """pneumatic press: hiss + heavy clunk."""
    t = tarr(n)
    hiss = fft_band(rng.standard_normal(n), 2500, 9000, 0.4) * asym_env(n, 0.1, 1.0) * np.exp(-t / 0.15)
    m = n - int(0.06 * SR)
    cl = np.zeros(n)
    tt = tarr(m)
    place(cl, click(m, rng, 300, 4000, 0.008) + np.sin(sweep_phase(90 + 100 * np.exp(-tt / 0.012))) * exp_env(m, 0.09), 0.06 * SR)
    return 0.4 * norm(hiss) + cl


def _bed(n, rng, comps):
    chans = []
    for c in range(2):
        r = np.random.default_rng(rng.integers(1 << 31) + c)
        chans.append(sum(g * norm(f(r)) for g, f in comps))
    return np.stack(chans, 1)


def fx_street_amb(n, rng, p):
    """night street bed: distant traffic rumble with slow swells, light rain (sparse droplets + hiss), far city hum."""
    t = tarr(n)
    rain_amt = p.get("rain", 1.0)

    def traffic(r):
        sw = 0.6 + 0.4 * np.sin(2 * np.pi * (0.07 * t + r.uniform())) * np.sin(2 * np.pi * (0.031 * t + r.uniform()))
        return fft_band(r.standard_normal(n), 35, 500, 0.6) * sw

    def rain(r):
        drops = fft_band((r.random(n) < 0.0016) * r.standard_normal(n) * r.exponential(1.0, n), 1800, 9000, 0.4)
        return norm(drops) + 0.6 * norm(fft_band(r.standard_normal(n), 2500, 11000, 0.5))

    def hum(r):
        return tone(n, 58.0 + r.uniform(-0.3, 0.3), (1.0, 0.5, 0.3)) * 0.5 + fft_band(r.standard_normal(n), 150, 900, 0.5) * 0.2

    x = _bed(n, rng, [(1.0, traffic), (0.32 * rain_amt, rain), (0.12, hum)])
    return fade_edges(x, p.get("fade", 0.6), p.get("fade", 0.6))


def fx_hall_amb(n, rng, p):
    """data hall bed: broadband fan roar (low-passed) + blade tones + mains hum, decorrelated L/R."""
    def fans(r):
        return fft_band(r.standard_normal(n), 60, 2500, 0.8)

    def blades(r):
        return tone(n, 233.0 + r.uniform(-2, 2), (1.0, 0.3), vib=0.002, vib_hz=0.3) + tone(n, 311.0 + r.uniform(-2, 2), (0.6,))

    def hum(r):
        return tone(n, 120.0, (1.0, 0.4, 0.2))

    x = _bed(n, rng, [(1.0, fans), (0.05, blades), (0.08, hum)])
    return fade_edges(x, p.get("fade", 0.6), p.get("fade", 0.6))


def fx_hologram(n, rng, p):
    """hologram tree: airy glassy shimmer, slowly rising cluster."""
    t = tarr(n)
    x = sum(np.sin(sweep_phase(f0 * (1 + 0.25 * t / t[-1]) * (1 + 0.003 * np.sin(2 * np.pi * (0.5 + 0.3 * k) * t))))
            for k, f0 in enumerate((523.0, 659.0, 784.0, 1046.0, 1318.0)))
    air = fft_band(rng.standard_normal(n), 4000, 11000, 0.4) * (0.5 + 0.5 * np.sin(2 * np.pi * 9 * t)) ** 2
    return fade_edges(norm(x) + 0.25 * norm(air), 0.5, 0.6)


def fx_footstep(n, rng, p):
    t = tarr(n)
    heel = click(n, rng, 200, 3000, 0.012) + 0.6 * np.sin(sweep_phase(110 + 60 * np.exp(-t / 0.01))) * exp_env(n, 0.03)
    place(heel, 0.4 * click(int(0.05 * SR), rng, 1500, 6000, 0.006), 0.06 * SR)   # toe scuff
    return heel


def fx_door(n, rng, p):
    """push-bar door: latch clank, air swoosh, soft close."""
    out = np.zeros(n)
    m = int(0.12 * SR)
    place(out, click(m, rng, 600, 5000, 0.006) + 0.6 * ring(m, 720, 0.03), 0)
    m2 = int(0.5 * SR)
    place(out, 0.5 * norm(swept_noise(m2, rng, 300, 1200, 1.0)) * asym_env(m2, 0.4), 0.05 * SR)
    m3 = int(0.15 * SR)
    if n > int(0.75 * SR) + m3:
        place(out, 0.4 * (click(m3, rng, 300, 3000, 0.01)), 0.75 * SR)
    return out


def fx_flap_vent(n, rng, p):
    """bird flap in the wall: rubber flap slap, swing creak, second softer slap."""
    out = np.zeros(n)
    m = int(0.1 * SR)
    for dt, g in ((0.0, 1.0), (0.32, 0.5), (0.55, 0.25)):
        place(out, g * (click(m, rng, 250, 2500, 0.012) + 0.4 * ring(m, 300, 0.02)), dt * SR)
    m2 = int(0.3 * SR)
    t2 = tarr(m2)
    cr = np.sin(sweep_phase(700 + 200 * np.sin(2 * np.pi * 3 * t2))) * (rng.random(m2) < 0.3) * asym_env(m2, 0.5)
    place(out, 0.15 * fft_band(cr, 400, 3000, 0.5), 0.05 * SR)
    return out


def fx_sat_ping(n, rng, p):
    """satellite uplink: sonar-like ping with spaced echoes."""
    out = np.zeros(n)
    m = min(n, int(0.9 * SR))
    pg = tone(m, 1320, (1.0, 0.2, 0.1)) * exp_env(m, 0.22, 0.004)
    for dt, g in ((0.0, 1.0), (0.28, 0.35), (0.56, 0.15)):
        place(out, g * pg, dt * SR)
    return out


def fx_cctv_whir(n, rng, p):
    """PTZ camera pan motor + lens focus tick."""
    out = servo(n, rng, 260, 340, 0.9) * 0.9
    place(out, 0.4 * click(int(0.04 * SR), rng, 2000, 8000, 0.002), 0.85 * n)
    return out


def fx_link_blip(n, rng, p):
    """link established: two short ascending blips."""
    out = np.zeros(n)
    f = p.get("f", rng.uniform(1300, 1700))
    for i, dt in enumerate((0.0, 0.055)):
        m = int(0.05 * SR)
        place(out, tone(m, f * (1.5 if i else 1.0), (1.0, 0.25)) * exp_env(m, 0.015, 0.001), dt * SR)
    return out


def fx_link_drop(n, rng, p):
    out = np.zeros(n)
    f = p.get("f", 1500)
    for i, dt in enumerate((0.0, 0.06)):
        m = int(0.05 * SR)
        place(out, tone(m, f / (1.5 if i else 1.0), (1.0,)) * exp_env(m, 0.015, 0.001), dt * SR)
    return out


def fx_disclaimer_whoosh(n, rng, p):
    """hard cut to the card: fast whoosh with a tape-stop drop."""
    t = tarr(n)
    w = norm(swept_noise(n, rng, 3000, 300, 0.8, 0.7)) * asym_env(n, 0.25, 1.3)
    ts = tone(n, 300 * 0.3 ** np.clip(t / t[-1], 0, 1), (1.0, 0.5, 0.3)) * asym_env(n, 0.1, 1.0)
    return w + 0.3 * norm(ts)


def fx_hop(n, rng, p):
    """short hop: one wing flap + light talon landing."""
    out = np.zeros(n)
    place(out, 0.6 * fx_wing_flap(int(0.16 * SR), rng, p), 0)
    m = n - int(0.12 * SR)
    if m > 0:
        place(out, 0.7 * fx_landing_thump(m, rng, p), 0.12 * SR)
    return out


def fx_crystal_tick(n, rng, p):
    """crystal oscillator 'tick' (stylised clock): tiny glassy tick."""
    t = tarr(n)
    return (np.sin(2 * np.pi * 3400 * t) + 0.5 * np.sin(2 * np.pi * 5100 * t)) * exp_env(n, 0.01, 0.0005)


def fx_head_servo_crowd(n, rng, p):
    """many heads turning at once (crowd tracks the Subject): 10-24 small servos spread over dur, mixed pitches."""
    out = np.zeros(n)
    k = int(p.get("count", np.clip(n / SR * 10, 6, 24)))
    for _ in range(k):
        m = int(rng.uniform(0.14, 0.26) * SR)
        f0 = rng.uniform(650, 1250)
        place(out, rng.uniform(0.4, 1.0) * servo(m, rng, f0, f0 * rng.uniform(1.1, 1.5), 0.4), rng.uniform(0, max(1, n - m)))
    return out


def fx_elec_glow(n, rng, p):
    """electrical signal glow (gold waves along contacts, wirebonds, TW electrodes): fast random ticks + crackle + faint
    buzz at `rate` ticks/s (default 90), swelling in and out over dur."""
    t = tarr(n)
    rate = p.get("rate", 90.0)
    imp = (rng.random(n) < rate / SR) * rng.choice([-1.0, 1.0], n) * rng.uniform(0.4, 1.0, n)
    ticks = fft_band(imp, 1800, 9000, 0.4)
    crackle = fft_band((rng.random(n) < 0.01) * rng.standard_normal(n), 3000, 11000, 0.4)
    buzz = tone(n, p.get("f", 240.0), (1.0, 0.5, 0.35, 0.25)) * (0.6 + 0.4 * np.sin(2 * np.pi * 11 * t))
    x = norm(ticks) + 0.35 * norm(crackle) + 0.25 * norm(buzz)
    return fade_edges(x * asym_env(n, 0.3, 0.5), 0.05, 0.15)


def fx_pic_lightup(n, rng, p):
    """PIC light-up front: the light runs through couplers, MMI tree and MZI arms - rising tone sweep with a widening
    cluster of partials (each split adds a partial) and an airy top."""
    t = tarr(n)
    u = np.clip(t / t[-1], 0, 1)
    f0 = p.get("f0", 300.0) * 2 ** (2.5 * u)
    x = np.zeros(n)
    for k, ratio in enumerate((1.0, 1.5, 2.0, 2.5, 3.0)):
        onset = k / 5.0                          # partials enter as the tree splits
        g = np.clip((u - onset) * 6, 0, 1)
        x += g * np.sin(sweep_phase(f0 * ratio)) / (1 + 0.4 * k)
    air = fft_band(rng.standard_normal(n), 4000, 11000, 0.4) * u ** 2
    return (norm(x) + 0.12 * norm(air)) * asym_env(n, 0.85, 0.7)


def fx_wire_bonder(n, rng, p):
    """wire bonder: per bond an ultrasonic chirp (stylised, shifted into hearing: 7-9 kHz, 25 ms) + capillary tap;
    params.count bonds spaced params.step s (default 1 bond)."""
    out = np.zeros(n)
    cnt, step = int(p.get("count", 1)), p.get("step", 0.18)
    for i in range(cnt):
        m = int(0.06 * SR)
        tt = tarr(m)
        chirp = np.sin(sweep_phase(7000 + 2000 * np.clip(tt / 0.025, 0, 1))) * (tt < 0.025) * np.sin(np.pi * np.clip(tt / 0.025, 0, 1))
        tap = click(m, rng, 1500, 7000, 0.0015) + 0.5 * ring(m, 3800, 0.006)
        seg = np.zeros(m)
        place(seg, 0.8 * tap, 0)
        place(seg, 0.5 * chirp[: m - int(0.008 * SR)], 0.008 * SR)
        place(out, seg * (1 - 0.05 * (i % 3)), i * step * SR)
    return out


def fx_align_stage(n, rng, p):
    """active alignment stage: micro stepper/piezo servo - short high whirs in small steps, settling (shorter, quieter)."""
    out = np.zeros(n)
    t, k = 0.0, 0
    while t * SR < n - 0.05 * SR:
        d = max(0.03, 0.12 * 0.8 ** k)
        m = int(d * SR)
        f0 = rng.uniform(1600, 2200)
        place(out, (0.85 ** k) * servo(m, rng, f0, f0 * rng.uniform(1.05, 1.2), 0.3), t * SR)
        t += d + rng.uniform(0.04, 0.12)
        k += 1
    return out


def fx_uv_cure(n, rng, p):
    """UV cure flash: zap onset + electric buzz for the exposure + soft release."""
    t = tarr(n)
    zap = fft_band(rng.standard_normal(n), 2000, 10000, 0.4) * exp_env(n, 0.03, 0.001)
    buzz = tone(n, 120.0, (1.0, 0.7, 0.6, 0.5, 0.4, 0.3)) * asym_env(n, 0.08, 0.4)
    hiss = fft_band(rng.standard_normal(n), 5000, 12000, 0.4) * asym_env(n, 0.1, 0.5) * 0.3
    return norm(zap) + 0.5 * norm(buzz) + 0.4 * norm(hiss) * (t > 0)


def fx_lid_press(n, rng, p):
    """lid / enclosure press: short pneumatic hiss, then a dense metal thunk."""
    out = np.zeros(n)
    m = int(0.12 * SR)
    place(out, 0.25 * fft_band(rng.standard_normal(m), 2500, 9000, 0.4) * asym_env(m, 0.3), 0)
    m2 = n - int(0.1 * SR)
    tt = tarr(m2)
    th = np.sin(sweep_phase(85 + 70 * np.exp(-tt / 0.015))) * exp_env(m2, 0.11, 0.001)
    th += 0.6 * click(m2, rng, 300, 4000, 0.006) + 0.3 * ring(m2, 940, 0.05)
    place(out, th, 0.1 * SR)
    return out


def fx_mimo_reconfig(n, rng, p):
    """satellite MIMO links re-configure: soft chord blips (3-4 note chord, quick upward arpeggio, sine + fifth)."""
    out = np.zeros(n)
    root = p.get("f", rng.choice([660.0, 740.0, 880.0]))
    ratios = (1.0, 1.25, 1.5, 2.0)[: int(p.get("notes", 4))]
    for i, r in enumerate(ratios):
        m = int(0.18 * SR)
        place(out, tone(m, root * r, (1.0, 0.15)) * exp_env(m, 0.06, 0.004), i * 0.035 * SR)
    return out


def fx_beam_sweep(n, rng, p):
    """one sweeping eye beam: band-limited whoosh with a faint tonal edge; layer several events for many beams and use
    pan / params.pan_end to move it across the stereo field."""
    w = norm(swept_noise(n, rng, p.get("f0", 900), p.get("f1", 3500), 0.6, 1.0)) * asym_env(n, p.get("peak", 0.5), 1.0)
    t = tarr(n)
    edge = np.sin(sweep_phase(p.get("tone", 1320.0) * (1 + 0.1 * t / t[-1]))) * asym_env(n, 0.5, 1.5)
    return w + 0.15 * edge


def fx_coupler_in(n, rng, p):
    """fiber/lens coupler block slides into place on the stage: short glide (rising) + soft seat click."""
    out = np.zeros(n)
    m = int(0.7 * n)
    place(out, 0.5 * norm(swept_noise(m, rng, 900, 2600, 0.6, 0.8)) * asym_env(m, 0.8, 1.2), 0)
    m2 = n - m
    place(out, click(m2, rng, 1500, 7000, 0.003) + 0.5 * ring(m2, 2400, 0.012), m)
    return out


def fx_coupler_out(n, rng, p):
    """coupler block released and withdrawn: release click + falling glide."""
    out = np.zeros(n)
    m = int(0.08 * SR)
    place(out, click(m, rng, 1500, 7000, 0.003) + 0.4 * ring(m, 2000, 0.01), 0)
    m2 = n - int(0.05 * SR)
    place(out, 0.5 * norm(swept_noise(m2, rng, 2600, 900, 0.6, 0.8)) * asym_env(m2, 0.3, 1.2), 0.05 * SR)
    return out


def fx_scooter_zoom(n, rng, p):
    """zoom transition into the bird on the scooter: electric-motor whine rising with a Doppler-like glide + rushing
    whoosh, peaking at `peak` (fraction of dur, default 0.8) to land on the cut."""
    t = tarr(n)
    pk = p.get("peak", 0.8)
    u = np.clip(t / t[-1], 0, 1)
    f = 420 * 2 ** (1.6 * np.clip(u / pk, 0, 1)) * np.where(u > pk, 0.85, 1.0)
    whine = tone(n, f, (1.0, 0.45, 0.25, 0.15), vib=0.004, vib_hz=23) * asym_env(n, pk, 1.2)
    rush = norm(swept_noise(n, rng, 400, 6000, 0.9, 1.3)) * asym_env(n, pk, 1.5)
    return 0.6 * norm(whine) + rush


def fx_pass_chime(n, rng, p):
    """test PASS: short bright two-note chime (major third up), bell-like partials."""
    out = np.zeros(n)
    for i, f in enumerate((p.get("f", 1046.5), p.get("f", 1046.5) * 1.26)):
        m = n - int(i * 0.09 * SR)
        x = sum(a * np.sin(2 * np.pi * f * r * tarr(m)) for r, a in ((1, 1.0), (2.0, 0.35), (3.01, 0.15))) * exp_env(m, 0.25, 0.003)
        place(out, x, i * 0.09 * SR)
    return out


def fx_place_click(n, rng, p):
    """precise part placement on the bench (isolator, lens): small tick + tiny glassy ring."""
    return click(n, rng, 2500, 9000, 0.0012) + 0.4 * ring(n, p.get("f", 4200), 0.01)


def fx_pneumatic(n, rng, p):
    """pneumatic press stroke: air hiss with a valve click; params.dir 'down' (rising hiss) or 'up' (falling, softer)."""
    up = p.get("dir", "down") == "up"
    x = norm(swept_noise(n, rng, 5000 if up else 2500, 2500 if up else 6000, 0.7, 1.0)) * asym_env(n, 0.15 if up else 0.7, 1.0)
    m = int(0.04 * SR)
    v = np.zeros(n)
    place(v, click(m, rng, 1500, 6000, 0.002), 0)
    return (0.6 if up else 1.0) * x + 0.6 * v


def fx_lane_ticks(n, rng, p):
    """lanes coming up one by one (link test): params.count short ticks evenly over dur, rising pitch."""
    out = np.zeros(n)
    cnt = int(p.get("count", 8))
    for i in range(cnt):
        m = int(0.03 * SR)
        f = p.get("f0", 1800) * 2 ** (i / max(1, cnt - 1) * p.get("octaves", 0.8))
        place(out, tone(m, f, (1.0, 0.2)) * exp_env(m, 0.008, 0.0005), i * (n - m) / max(1, cnt - 1))
    return out


# kind -> (function, default dur s, default peak gain (linear), min gap s between events of this kind)
KINDS = {
    "wing_flap": (fx_wing_flap, 0.18, 0.30, 0.08),
    "flight": (fx_flight, 1.5, 0.26, 0.0),
    "takeoff": (fx_takeoff, 0.9, 0.40, 0.2),
    "landing_thump": (fx_landing_thump, 0.35, 0.45, 0.1),
    "head_servo": (fx_head_servo, 0.22, 0.07, 0.12),
    "blink": (fx_blink, 0.18, 0.30, 0.1),
    "eye_glow_on": (fx_eye_glow_on, 0.6, 0.20, 0.1),
    "eye_glow_off": (fx_eye_glow_off, 0.45, 0.18, 0.1),
    "pad_charge_hum": (fx_pad_charge_hum, 1.2, 0.16, 0.0),
    "cell_dock": (fx_cell_dock, 0.45, 0.42, 0.1),
    "cell_undock": (fx_cell_undock, 0.4, 0.40, 0.1),
    "beak_latch": (fx_beak_latch, 0.15, 0.45, 0.08),
    "beak_open": (fx_beak_open, 0.25, 0.35, 0.08),
    "contact_gnd": (fx_contact_gnd, 0.12, 0.35, 0.05),
    "contact_pwr": (fx_contact_pwr, 0.12, 0.35, 0.05),
    "contact_sig": (fx_contact_sig, 0.14, 0.32, 0.05),
    "leash_pull": (fx_leash_pull, 0.45, 0.30, 0.1),
    "whoosh": (fx_whoosh, 0.7, 0.28, 0.15),
    "whip": (fx_whip, 0.35, 0.40, 0.15),
    "fill_whoosh": (fx_fill_whoosh, 1.0, 0.40, 0.3),
    "elec_hum": (fx_elec_hum, 2.5, 0.12, 0.0),
    "pic_hum": (fx_elec_hum, 2.5, 0.12, 0.0),
    "tia_hum": (lambda n, r, p: fx_elec_hum(n, r, {"f": 150.0, "whine": 9100.0, **p}), 2.5, 0.12, 0.0),
    "laser_on": (fx_laser_on, 0.5, 0.25, 0.1),
    "pam4_chatter": (fx_pam4_chatter, 1.5, 0.10, 0.0),
    "pd_flash": (fx_pd_flash, 0.08, 0.14, 0.05),
    "beam_launch": (fx_beam_launch, 0.7, 0.26, 0.2),
    "i2c_stream": (fx_i2c_stream, 1.5, 0.10, 0.0),
    "hud_lock": (fx_hud_lock, 0.6, 0.20, 0.2),
    "hud_tick": (fx_hud_tick, 0.06, 0.15, 0.05),
    "explode": (fx_explode, 0.9, 0.35, 0.3),
    "snap_back": (fx_snap_back, 0.6, 0.38, 0.3),
    "robot_servo": (fx_robot_servo, 0.6, 0.18, 0.1),
    "conveyor": (fx_conveyor, 1.5, 0.14, 0.0),
    "assembly_clunk": (fx_assembly_clunk, 0.3, 0.30, 0.1),
    "street_amb": (fx_street_amb, 10.0, 0.16, 0.0),
    "hall_amb": (fx_hall_amb, 10.0, 0.10, 0.0),
    "hologram": (fx_hologram, 4.0, 0.10, 0.0),
    "footstep": (fx_footstep, 0.15, 0.22, 0.2),
    "door": (fx_door, 0.9, 0.30, 0.3),
    "flap_vent": (fx_flap_vent, 0.7, 0.30, 0.2),
    "sat_ping": (fx_sat_ping, 1.3, 0.24, 0.2),
    "cctv_whir": (fx_cctv_whir, 0.8, 0.16, 0.2),
    "link_blip": (fx_link_blip, 0.11, 0.14, 0.09),
    "link_drop": (fx_link_drop, 0.11, 0.10, 0.09),
    "disclaimer_whoosh": (fx_disclaimer_whoosh, 0.5, 0.38, 0.3),
    "hop": (fx_hop, 0.4, 0.35, 0.15),
    "crystal_tick": (fx_crystal_tick, 0.05, 0.10, 0.1),
    "head_servo_crowd": (fx_head_servo_crowd, 1.5, 0.12, 0.3),
    # v3 (DevLog-018): electrical / light-up / assembly / MIMO / sweeps
    "elec_glow": (fx_elec_glow, 1.5, 0.18, 0.0),
    "pic_lightup": (fx_pic_lightup, 2.0, 0.20, 0.3),
    "dfb_on": (fx_laser_on, 0.5, 0.25, 0.1),
    "wire_bonder": (fx_wire_bonder, 0.25, 0.22, 0.06),
    "align_stage": (fx_align_stage, 1.0, 0.14, 0.2),
    "uv_cure": (fx_uv_cure, 0.6, 0.22, 0.3),
    "lid_press": (fx_lid_press, 0.4, 0.38, 0.2),
    "mimo_reconfig": (fx_mimo_reconfig, 0.35, 0.12, 0.15),
    "beam_sweep": (fx_beam_sweep, 1.2, 0.18, 0.0),
    # v4 (DevLog-020): M's assembly events and the S6 -> S7 zoom
    "coupler_in": (fx_coupler_in, 0.45, 0.26, 0.15),
    "coupler_out": (fx_coupler_out, 0.45, 0.24, 0.15),
    "scooter_zoom": (fx_scooter_zoom, 1.2, 0.36, 0.5),
    "pass_chime": (fx_pass_chime, 0.6, 0.22, 0.3),
    "place_click": (fx_place_click, 0.08, 0.20, 0.05),
    "pneumatic": (fx_pneumatic, 0.3, 0.16, 0.1),
    "lane_ticks": (fx_lane_ticks, 0.3, 0.16, 0.2),
}
# accepted alternative names (left: what an emitter may write; right: canonical kind). Includes the kinds S's
# build_film.py emitted before this list was published (DevLog-011 / scene_events.json of 2026-10-10 08:43).
# one-shot kinds whose sound must not stretch with an emitter's scene-length dur (e.g. S's 2 s "card" event)
MAX_DUR = {"pass_chime": 0.8, "place_click": 0.12, "pneumatic": 0.6, "scooter_zoom": 2.0, "coupler_in": 0.8, "coupler_out": 0.8, "lid_press": 0.6, "uv_cure": 1.5, "mimo_reconfig": 0.5, "dfb_on": 0.8, "disclaimer_whoosh": 0.6, "whoosh": 1.2, "whip": 0.6, "fill_whoosh": 1.4, "door": 1.0, "hop": 0.5}
ALIASES = {"flap": "wing_flap", "landing": "landing_thump", "servo": "head_servo", "head_turn": "head_servo",
           "head_turn_crowd": "head_servo_crowd", "dock": "cell_dock", "undock": "cell_undock", "latch": "beak_latch",
           "glow_on": "eye_glow_on", "link": "link_blip", "ping": "sat_ping", "cctv": "cctv_whir",
           "ambience_street": "street_amb", "ambience_hall": "hall_amb", "fill": "fill_whoosh",
           "macro_whoosh": "fill_whoosh", "hologram_hum": "hologram", "flap_swing": "flap_vent",
           "pad_charge": "pad_charge_hum", "contact_mate": "contact_sig", "rx_hum": "tia_hum",
           "data_stream": "i2c_stream", "reassemble": "snap_back", "hatchery_robot": "robot_servo",
           "card": "disclaimer_whoosh", "electrical_glow": "elec_glow", "elec_signal": "elec_glow",
           "pic_light_up": "pic_lightup", "light_up": "pic_lightup", "dfb_switch_on": "dfb_on", "wirebond": "wire_bonder",
           "wire_bond": "wire_bonder", "active_alignment": "align_stage", "uv_flash": "uv_cure", "lid": "lid_press",
           "mimo": "mimo_reconfig", "sat_reconfig": "mimo_reconfig", "sweep": "beam_sweep",
           "bond": "wire_bonder", "lid_close": "lid_press", "eye_on": "eye_glow_on", "eye_off": "eye_glow_off",
           "coupler_insert": "coupler_in", "coupler_remove": "coupler_out", "zoom_transition": "scooter_zoom",
           "scooter_zoom_transition": "scooter_zoom", "pass": "pass_chime", "test_pass": "pass_chime",
           "part_place": "place_click", "press": "pneumatic", "press_down": "pneumatic", "press_up": "pneumatic",
           "eye_diagram_on": "hud_tick", "eye_diagram_off": "hud_tick"}
# (canonical kind, substring of the event name) -> kind override / extra params
NAME_RULES = [("pneumatic", "up", "pneumatic", {"dir": "up"}),
              ("contact_sig", "gnd", "contact_gnd", None), ("contact_sig", "pwr", "contact_pwr", None),
              ("beak_latch", "release", "beak_open", None), ("fill_whoosh", "rx", "fill_whoosh", {"color": "cyan"})]


def canon(e):
    k = ALIASES.get(e.get("kind"), e.get("kind"))
    name = e.get("name", "")
    params = dict(e.get("params", {}))
    for kk, sub, new, extra in NAME_RULES:
        if k == kk and sub in name:
            k = new
            if extra:
                params = {**extra, **params}
    return k, params


def pan_gains(pan, n, pan_end=None):
    p = np.full(n, float(pan)) if pan_end is None else np.linspace(pan, pan_end, n)
    a = (np.clip(p, -1, 1) + 1) * np.pi / 4
    return np.cos(a), np.sin(a)


def render_event(e, idx):
    fn, d_dur, d_gain, _ = KINDS[e["kind"]]
    p = dict(e.get("params", {}))
    dur = min(float(e.get("dur", d_dur)), MAX_DUR.get(e["kind"], 1e9))
    p.setdefault("seed_name", e.get("name", ""))
    n = max(int(round(dur * SR)), 64)
    seed = zlib.crc32(("%s|%s|%.4f|%d" % (e["kind"], e.get("name", ""), e["t"], idx)).encode())
    rng = np.random.default_rng(seed)
    x = fn(n, rng, p)
    x = x / max(np.max(np.abs(x)), 1e-12) * d_gain * float(e.get("gain", 1.0))
    if x.ndim == 1:
        gl, gr = pan_gains(e.get("pan", 0.0), n, p.get("pan_end"))
        x = np.stack([x * gl, x * gr], 1)
    else:
        gl, gr = pan_gains(e.get("pan", 0.0), n)
        x = x * np.stack([gl, gr], 1) * np.sqrt(2)
    lead = int(round(LEAD * SR))
    x = np.vstack([np.zeros((lead, 2)), x])
    fi, fo = int(FADE_IN * SR), int(FADE_OUT * SR)
    x[lead:lead + fi] *= (0.5 - 0.5 * np.cos(np.pi * np.arange(fi) / fi))[:, None]
    x[-fo:] *= (0.5 + 0.5 * np.cos(np.pi * np.arange(fo) / fo))[:, None]
    return x, lead


def apply_overrides(ev, path, log):
    """drop / shift / rekind events by exact name and time (+-tol) and add new ones, before thinning, so thinning
    sees the final list."""
    ov = json.load(open(path))
    n0 = len(ev)
    for d in ov.get("drop", []):
        hit = [e for e in ev if e.get("name") == d["name"] and abs(e["t"] - d["t"]) <= d.get("tol", 0.02)]
        log("override drop %s @ %.2f: %d event(s)" % (d["name"], d["t"], len(hit)))
        ev = [e for e in ev if not any(e is h for h in hit)]
    for d in ov.get("shift", []):
        for e in ev:
            if e.get("name") == d["name"] and abs(e["t"] - d["t"]) <= d.get("tol", 0.02):
                log("override shift %s %.2f -> %.2f" % (d["name"], e["t"], d["t_new"]))
                e["t"] = d["t_new"]
    for d in ov.get("drop_window", []):                # e.g. every S event named "assembly_*" in the assembly slot
        hit = [e for e in ev if e.get("name", "").startswith(d["prefix"]) and d["t0"] <= e["t"] <= d["t1"]]
        log("override drop_window %s* %.2f-%.2f: %d event(s)" % (d["prefix"], d["t0"], d["t1"], len(hit)))
        ev = [e for e in ev if not any(e is h for h in hit)]
    for d in ov.get("rekind", []):
        for e in ev:
            if e.get("name") == d["name"] and abs(e["t"] - d["t"]) <= d.get("tol", 0.02):
                log("override rekind %s @ %.2f: %s -> %s" % (d["name"], e["t"], e["kind"], d["kind"]))
                e["kind"] = d["kind"]
                if "params" in d:
                    e["params"] = dict(e.get("params", {}), **d["params"])
                if "dur" in d:
                    e["dur"] = d["dur"]
    for d in ov.get("add", []):
        e = {k: v for k, v in d.items() if k != "why"}
        k, params = canon(e)
        ev.append(dict(e, kind=k, params=params, src=os.path.basename(path)))
        log("override add %s %s @ %.2f" % (k, e.get("name", ""), e["t"]))
    log("overrides from %s: %d -> %d events" % (rel(path), n0, len(ev)))
    return ev


def load_events(path, log, overrides=None):
    raw = json.load(open(path))
    ev = raw["events"] if isinstance(raw, dict) else raw
    out, skipped = [], {}
    for e in ev:
        k, params = canon(e)
        if k not in KINDS:
            skipped[e.get("kind")] = skipped.get(e.get("kind"), 0) + 1
            continue
        e = dict(e, kind=k, params=params, src=os.path.basename(path))
        if not (0 <= float(e["t"]) < TOTAL):
            skipped["t out of range"] = skipped.get("t out of range", 0) + 1
            continue
        out.append(e)
    for k, c in skipped.items():
        log("skipped %d events: unknown kind %r" % (c, k))
    if overrides:
        out = apply_overrides(out, overrides, log)
    out.sort(key=lambda e: e["t"])
    kept, last = [], {}
    for e in out:                                         # thin dense kinds
        gap = KINDS[e["kind"]][3]
        if gap and e["t"] - last.get(e["kind"], -1e9) < gap:
            continue
        last[e["kind"]] = e["t"]
        kept.append(e)
    if len(kept) < len(out):
        log("thinned %d events (min gap per kind)" % (len(out) - len(kept)))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=None)
    ap.add_argument("--out", default=os.path.join(OUT_AUDIO, "sfx_%s.wav" % TODAY))
    ap.add_argument("--list-kinds", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overrides", default=None, help="JSON with drop/shift lists applied after reading the events")
    ap.add_argument("--supplement", action="store_true", help="also add A's provisional cues of kinds the scene file does not emit "
                    "(off by default since S's film v1 events, 2026-10-10 10:40, cover every kind S wants)")
    args = ap.parse_args()
    if args.list_kinds:
        for k, (fn, d, g, gap) in KINDS.items():
            al = ", ".join("`%s`" % a for a, c in ALIASES.items() if c == k)
            print("| `%s` | %.2f | %.2f | %.2f | %s | %s |" % (k, d, g, gap, al, (fn.__doc__ or "").strip().split("\n")[0]))
        return
    log = Log("sfx")
    src = args.events
    if src is None:
        s = os.path.join(HERE, "scene_events.json")
        src = s if os.path.exists(s) else os.path.join(HERE, "sfx_events_provisional.json")
    log("events from", rel(src))
    ev = load_events(src, log, args.overrides)
    prov = os.path.join(HERE, "sfx_events_provisional.json")
    if args.supplement and os.path.abspath(src) != os.path.abspath(prov) and os.path.exists(prov):
        have = {e["kind"] for e in ev}
        covered = {"flight": "wing_flap", "takeoff": "wing_flap"}   # S emits every wingbeat as wing_flap
        sup = [e for e in load_events(prov, log) if e["kind"] not in have and covered.get(e["kind"]) not in have]
        log("supplement from %s: %d events of kinds the scene file does not emit: %s" % (
            rel(prov), len(sup), ", ".join(sorted({e["kind"] for e in sup}))))
        ev = sorted(ev + sup, key=lambda e: e["t"])
    counts = {}
    for e in ev:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    log("%d events: %s" % (len(ev), ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    if args.dry_run:
        return
    N = int(round(TOTAL * SR))
    mix = np.zeros((N, 2))
    for i, e in enumerate(ev):
        x, lead = render_event(e, i)
        s = int(round(e["t"] * SR)) - lead
        place(mix, x, s)
    pk = np.max(np.abs(mix))
    if pk > PEAK_LIMIT:
        log("peak %.3f above limit, scaling whole track by %.3f" % (pk, PEAK_LIMIT / pk))
        mix *= PEAK_LIMIT / pk
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, mix.astype(np.float32), SR, subtype="PCM_24")
    log("length %.3f s, peak %.1f dBFS -> %s" % (N / SR, 20 * np.log10(np.max(np.abs(mix))), rel(args.out)))
    log("log", rel(log.path))


if __name__ == "__main__":
    main()
