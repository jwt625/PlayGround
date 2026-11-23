# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Synthesize the course's functional sound effects in numpy (no samples, deterministic); mono mp3 96 kbps.

Run from the project root:  uv run course/tools/sound/make_sfx.py
Writes course/media/sound/sfx_<name>.mp3 and course/tools/sound/logs/sfx_<timestamp>.json.
Level: every effect is scaled so its maximum momentary loudness (BS.1770, 400 ms window) is -24 LUFS, with the
sample peak capped at -6 dBFS (the cap wins for very short transients; the log records the final numbers).
Timbres: marimba bars (partials 1 : 3.9 : 9.2), glockenspiel bars (1 : 2.76 : 5.40), additive brass with a brightness
envelope, filtered-noise whoosh and clicks, a small synthetic room reverb. Some helpers follow the procedural SFX
script of an earlier film project (FFT band filter, swept noise, exponential envelopes).
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sound_common import HERE, SOUND_DIR, encode_mp3_target, loudness  # noqa: E402

SR = 48000
TARGET_M = -24.0
PEAK_CAP_DB = -6.0
LEAD = 0.004


def t_(n):
    return np.arange(n) / SR


def N(sec):
    return int(round(sec * SR))


def band(x, lo=None, hi=None, edge=0.3):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    lf = np.log2(np.maximum(f, 1e-3))
    m = np.ones_like(f)
    if lo:
        m *= 0.5 * (1 + np.tanh((lf - np.log2(lo)) / edge * 2))
    if hi:
        m *= 0.5 * (1 - np.tanh((lf - np.log2(hi)) / edge * 2))
    return np.fft.irfft(X * m, len(x))


def place(buf, x, start_s):
    s = N(start_s)
    e = min(len(buf), s + len(x))
    buf[s:e] += x[:e - s]


def bar_tone(f, dur, partials, decay, att=0.0015, seed=0):
    """Struck bar: partials = [(ratio, amp, decay_scale)], plus a short filtered mallet click."""
    n = N(dur)
    t = t_(n)
    y = np.zeros(n)
    for r, a, ds in partials:
        if f * r < SR / 2 * 0.9:
            y += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / (decay * ds))
    y *= np.minimum(1, t / att)
    rng = np.random.default_rng(seed)
    y += 0.05 * band(rng.standard_normal(n), 1500, 7000) * np.exp(-t / 0.004)
    return y


MARIMBA = [(1.0, 1.0, 1.0), (3.9, 0.30, 0.35), (9.2, 0.08, 0.12)]
GLOCK = [(1.0, 1.0, 1.0), (2.76, 0.32, 0.45), (5.40, 0.10, 0.2), (8.93, 0.04, 0.1)]
CHIME = [(1.0, 1.0, 1.0), (2.0, 0.22, 0.6), (3.0, 0.10, 0.35), (4.16, 0.05, 0.2)]


def note(name):
    names = {"C": -9, "C#": -8, "D": -7, "D#": -6, "E": -5, "F": -4, "F#": -3, "G": -2, "G#": -1, "A": 0, "A#": 1, "B": 2}
    p, o = name[:-1], int(name[-1])
    return 440.0 * 2 ** ((names[p] + 12 * (o - 4)) / 12)


def reverb(x, rt=0.35, wet=0.15, pre=0.012, seed=7):
    rng = np.random.default_rng(seed)
    n = N(rt * 1.6)
    ir = band(rng.standard_normal(n), 200, 6000) * np.exp(-t_(n) * 6.9 / rt)
    ir = np.concatenate([np.zeros(N(pre)), ir])
    ir /= np.sqrt((ir ** 2).sum())
    m = len(x) + len(ir)
    K = 1 << int(np.ceil(np.log2(m)))
    w = np.fft.irfft(np.fft.rfft(x, K) * np.fft.rfft(ir, K), K)[:len(x)]
    return (1 - wet) * x + wet * w * (np.abs(x).max() / (np.abs(w).max() + 1e-12))


def brass(f, dur, bright=7.0, att=0.035, rel=0.12, vib=True, seed=0):
    """Additive brass: harmonic amplitudes 1/n * exp(-(n-1)/nb(t)), nb follows a fast attack-brightness envelope."""
    n = N(dur)
    t = t_(n)
    nb = 1.2 + bright * (np.minimum(1, t / 0.04) * (0.75 + 0.25 * np.exp(-t / 0.15)))
    vibr = 1 + (0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.25) / 0.2, 0, 1) if vib else 0)
    ph = 2 * np.pi * np.cumsum(f * vibr) / SR
    y = np.zeros(n)
    for k in range(1, 24):
        if f * k > 12000:
            break
        y += (1 / k) * np.exp(-(k - 1) / nb) * np.sin(k * ph + 0.3 * k)
    env = np.minimum(1, t / att) * (1 + 0.15 * np.exp(-t / 0.06))
    env *= np.clip((dur - t) / rel, 0, 1)
    return y * env


def fade_tail(y, sec=0.03):
    n = min(N(sec), len(y))
    y[-n:] *= np.linspace(1, 0, n)
    return y


# ------------------------------------------------------------------ effects
def sfx_correct():
    """Two-note chime up a perfect fifth (A5 -> E6), marimba body + glockenspiel sparkle."""
    y = np.zeros(N(0.85))
    for i, (nm, t0) in enumerate((("A5", 0.0), ("E6", 0.11))):
        f = note(nm)
        place(y, 0.8 * bar_tone(f, 0.7, MARIMBA, 0.28, seed=i) + 0.45 * bar_tone(f, 0.7, GLOCK, 0.32, seed=10 + i), t0)
    return reverb(y, 0.35, 0.14)


def sfx_wrong():
    """Soft low wooden bonk: a bar tone near C4 with a slight downward pitch drop, low-passed; never harsh."""
    n = N(0.42)
    t = t_(n)
    f = note("C4") * (1 + 0.18 * np.exp(-t / 0.025)) * (1 - 0.06 * t / 0.42)
    ph = 2 * np.pi * np.cumsum(f) / SR
    # strong 2nd and 3rd partials so the bonk still reads on small laptop / phone speakers
    y = (np.sin(ph) * np.exp(-t / 0.16) + 0.6 * np.sin(2 * ph) * np.exp(-t / 0.08)
         + 0.3 * np.sin(3 * ph) * np.exp(-t / 0.05) + 0.12 * np.sin(3.9 * ph) * np.exp(-t / 0.03))
    y *= np.minimum(1, t / 0.003)
    rng = np.random.default_rng(3)
    y += 0.12 * band(rng.standard_normal(n), 120, 900) * np.exp(-t / 0.012)
    y = band(y, 70, 2200)
    return reverb(y, 0.25, 0.10)


def sfx_reveal():
    """Light whoosh rising 500 -> 2600 Hz (filtered noise, asymmetric swell), ending in a tiny soft pop."""
    n = N(0.40)
    t = t_(n)
    rng = np.random.default_rng(11)
    w = rng.standard_normal(n)
    u = t / t[-1]
    out = np.zeros(n)
    centers = np.geomspace(300, 6000, 18)
    lc = np.log2(500) + (np.log2(2600) - np.log2(500)) * u ** 1.3
    for c in centers:
        out += band(w, c / 2 ** 0.25, c * 2 ** 0.25, 0.15) * np.exp(-0.5 * ((np.log2(c) - lc) / 0.8) ** 2)
    env = np.where(u < 0.7, np.sin(0.5 * np.pi * u / 0.7) ** 2, np.cos(0.5 * np.pi * (u - 0.7) / 0.3) ** 2)
    out *= env
    out /= np.abs(out).max()
    pop = np.zeros(n)
    m = N(0.06)
    tp = t_(m)
    pop[n - m - N(0.02):n - N(0.02)] = np.sin(2 * np.pi * np.cumsum(700 + 500 * (1 - np.exp(-tp / 0.01))) / SR) * np.exp(-tp / 0.018)
    return reverb(out + 0.35 * pop, 0.2, 0.08)


def bubble(f0, f1, dur=0.07, tau=0.022):
    n = N(dur)
    t = t_(n)
    f = f0 + (f1 - f0) * (1 - np.exp(-t / 0.012))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / tau) * np.minimum(1, t / 0.001)


def sfx_message():
    """Generic notification pop: two quick rising bubble pops (no platform tone)."""
    y = np.zeros(N(0.32))
    place(y, bubble(520, 1050), 0.0)
    place(y, 0.85 * bubble(760, 1480), 0.085)
    return reverb(y, 0.2, 0.12)


def sfx_popup():
    """Soft ding: one chime tone at D6 with a gentle glockenspiel overtone and a slow decay."""
    f = note("D6")
    y = 0.85 * bar_tone(f, 1.0, CHIME, 0.30, att=0.004, seed=21) + 0.25 * bar_tone(2 * f, 1.0, GLOCK, 0.18, seed=22)
    y = band(y, 300, 7000)
    return reverb(y, 0.4, 0.15)


def sfx_sort_place():
    """Soft click: short wooden tick (filtered noise + 1.8 kHz resonance + small low thump)."""
    n = N(0.10)
    t = t_(n)
    rng = np.random.default_rng(31)
    y = 0.6 * band(rng.standard_normal(n), 1500, 6000) * np.exp(-t / 0.0025)
    y += 0.5 * np.sin(2 * np.pi * 1800 * t) * np.exp(-t / 0.012)
    y += 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.018)
    y *= np.minimum(1, t / 0.0005)
    return reverb(y, 0.12, 0.06)


def sfx_module_done():
    """Short positive flourish: marimba arpeggio C5 E5 G5 C6, then a glockenspiel C major chord ringing out."""
    y = np.zeros(N(1.45))
    for i, nm in enumerate(("C5", "E5", "G5", "C6")):
        place(y, 0.7 * bar_tone(note(nm), 0.5, MARIMBA, 0.22, seed=40 + i), 0.075 * i)
    for j, nm in enumerate(("C6", "E6", "G6")):
        place(y, 0.32 * bar_tone(note(nm), 1.1, GLOCK, 0.45, seed=50 + j), 0.30 + 0.012 * j)
    place(y, 0.5 * bar_tone(note("C4"), 1.1, MARIMBA, 0.4, seed=60), 0.30)
    return reverb(y, 0.45, 0.16)


def timpani(f, dur=1.2, seed=0):
    n = N(dur)
    t = t_(n)
    ff = f * (1 + 0.05 * np.exp(-t / 0.03))
    ph = 2 * np.pi * np.cumsum(ff) / SR
    y = np.sin(ph) * np.exp(-t / 0.55) + 0.4 * np.sin(1.5 * ph) * np.exp(-t / 0.3) + 0.2 * np.sin(1.98 * ph) * np.exp(-t / 0.25)
    rng = np.random.default_rng(seed)
    y += 0.25 * band(rng.standard_normal(n), 60, 700) * np.exp(-t / 0.02)
    return y * np.minimum(1, t / 0.002)


def sfx_certificate():
    """Short fanfare: brass triplet pickup on G4, then a held C major chord (C4 E4 G4 C5) with timpani and glockenspiel."""
    y = np.zeros(N(2.6))
    for k in range(3):
        place(y, 0.55 * (brass(note("G4"), 0.1, vib=False, seed=k) + 0.5 * brass(note("G3"), 0.1, vib=False)), 0.11 * k)
    hold = 1.55
    for nm, a, det in (("C4", 0.45, -3), ("E4", 0.4, 2), ("G4", 0.4, -2), ("C5", 0.6, 3)):
        f = note(nm)
        for d in (det, -det):                                  # two detuned players per part
            place(y, a * brass(f * 2 ** (d / 1200), hold), 0.36)
    place(y, 0.9 * timpani(note("C2") * 1.0), 0.36)
    place(y, 0.5 * timpani(note("G2"), seed=2), 0.0)
    for j, nm in enumerate(("C6", "E6", "G6", "C7")):
        place(y, 0.14 * bar_tone(note(nm), 1.0, GLOCK, 0.4, seed=70 + j), 0.36 + 0.06 * j)
    y = band(y, 50, 10000)
    return reverb(y, 0.9, 0.2)


EFFECTS = {
    "correct": sfx_correct,
    "wrong": sfx_wrong,
    "reveal": sfx_reveal,
    "message": sfx_message,
    "popup": sfx_popup,
    "sortPlace": sfx_sort_place,
    "moduleDone": sfx_module_done,
    "certificate": sfx_certificate,
}
FILES = {"correct": "sfx_correct", "wrong": "sfx_wrong", "reveal": "sfx_reveal", "message": "sfx_message",
         "popup": "sfx_popup", "sortPlace": "sfx_sort_place", "moduleDone": "sfx_module_done",
         "certificate": "sfx_certificate"}


def finish(y):
    y = np.concatenate([np.zeros(N(LEAD)), fade_tail(np.asarray(y, float), 0.03)])
    # trim trailing near-silence (below -70 dB of peak)
    thr = np.abs(y).max() * 10 ** (-70 / 20)
    last = np.nonzero(np.abs(y) > thr)[0][-1]
    y = fade_tail(y[:last + N(0.01)], 0.01)
    L = loudness(y, SR)
    g = 10 ** ((TARGET_M - L["M_max"]) / 20)
    cap = 10 ** (PEAK_CAP_DB / 20) / np.abs(y).max()
    capped = g > cap
    return y * min(g, cap), capped


def main():
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {"date": stamp, "target_M_max_lufs": TARGET_M, "peak_cap_dbfs": PEAK_CAP_DB, "items": {}}
    for key, fn in EFFECTS.items():
        y, capped = finish(fn())
        out = SOUND_DIR / f"{FILES[key]}.mp3"
        g = encode_mp3_target(y, SR, out, "96k", "M_max", TARGET_M, PEAK_CAP_DB)
        L = loudness(y * g, SR)
        log["items"][key] = {"file": out.name, "dur_s": round(len(y) / SR, 3), "M_max": round(L["M_max"], 2),
                             "peak_dbfs": round(L["peak_dbfs"], 2), "peak_capped": bool(capped)}
        print(key, log["items"][key])
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / f"sfx_{stamp}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
