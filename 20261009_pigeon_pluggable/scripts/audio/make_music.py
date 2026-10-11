"""Procedural music bed (numpy additive synthesis; no samples, no third-party music, so no licence beyond this repo).

Run (project root): uv run --no-project --with numpy --with soundfile python scripts/audio/make_music.py
Writes outputs/audio/music_<date>.wav (48 kHz stereo, 62.0 s, peak -6 dBFS; level is set in mix.py).
Dark cyberpunk pad: D minor progression, two chords per 10 s scene (chord changes on the scene grid so beats stay
locked to the cut points), detuned additive-saw pad with a slow filter sweep, a plucked 8th-note arpeggio at 100 BPM,
a sub pulse on beats 1 and 3. Ends with the S6 cut at 60.0 s (short tail), silent under the S7 disclaimer.
"""
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import OUT_AUDIO, SR, TODAY, TOTAL, Log, rel  # noqa: E402

BPM = 100.0
BEAT = 60.0 / BPM
END = 60.0          # S6 -> S7 hard cut


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# (start s, root midi, chord tones as midi) ; D2 = 38
PROG = [
    (0.0, [38, 45, 50, 53, 57]),      # Dm
    (5.0, [38, 45, 52, 53, 57]),      # Dm(add9)
    (10.0, [34, 41, 46, 50, 53]),     # Bb
    (15.0, [36, 43, 48, 52, 55]),     # C
    (20.0, [38, 45, 50, 53, 57]),     # Dm
    (25.0, [41, 48, 53, 57, 60]),     # F
    (30.0, [43, 50, 55, 58, 62]),     # Gm
    (35.0, [45, 52, 57, 61, 64]),     # A
    (40.0, [34, 41, 46, 50, 53]),     # Bb
    (45.0, [36, 43, 48, 52, 55]),     # C
    (50.0, [38, 45, 50, 53, 57]),     # Dm
    (55.0, [45, 52, 57, 61, 64]),     # A
]


def lowpass_sweep(x, t, f_lo, f_hi, rate):
    """block-wise one-pole low-pass with a slowly moving cutoff (cheap, smooth enough for a pad)."""
    out = np.empty_like(x)
    y = 0.0
    blk = 256
    for i in range(0, len(x), blk):
        fc = f_lo + (f_hi - f_lo) * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t[i]))
        a = np.exp(-2 * np.pi * fc / SR)
        seg = x[i:i + blk]
        o = np.empty_like(seg)
        for j in range(len(seg)):
            y = (1 - a) * seg[j] + a * y
            o[j] = y
        out[i:i + blk] = o
    return out


def saw_add(f, n, nh=10, phase=0.0):
    t = np.arange(n) / SR
    return sum(np.sin(2 * np.pi * f * k * t + phase * k) / k for k in range(1, nh + 1) if f * k < 9000)


def main():
    log = Log("music")
    N = int(TOTAL * SR)
    pad = np.zeros((N, 2))
    arp = np.zeros((N, 2))
    sub = np.zeros(N)
    rng = np.random.default_rng(1149)
    xf = int(0.6 * SR)                       # chord crossfade
    for i, (t0, notes) in enumerate(PROG):
        t1 = PROG[i + 1][0] if i + 1 < len(PROG) else END
        s0, s1 = int(t0 * SR), int(t1 * SR)
        a, b = max(0, s0 - xf // 2), min(N, s1 + xf // 2)
        n = b - a
        env = np.ones(n)
        env[:xf] = np.linspace(0, 1, xf) if a > 0 else np.linspace(0, 1, xf) ** 2
        env[-xf:] = np.linspace(1, 0, xf)
        for c, det in enumerate((-0.07, 0.07)):          # detune in semitones per channel
            v = np.zeros(n)
            for m in notes[1:]:
                for d in (det, -det * 0.5):
                    v += saw_add(midi(m + d), n, 8, rng.uniform(0, 6.28))
            v += 0.8 * saw_add(midi(notes[0] + det * 0.3), n, 6)
            pad[a:b, c] += v * env
        # arpeggio: 8th notes, pattern over chord tones (upper octave), plucked
        pat = [1, 2, 3, 4, 3, 2, 4, 3]
        k = 0
        tt = t0
        while tt < t1 - 1e-6:
            m = notes[pat[k % 8]] + 12
            L = int(0.28 * SR)
            ts = np.arange(L) / SR
            f = midi(m)
            pl = (np.sin(2 * np.pi * f * ts) + 0.35 * np.sin(4 * np.pi * f * ts) + 0.12 * np.sin(6 * np.pi * f * ts)) * np.exp(-ts / 0.09)
            pl *= np.minimum(1, ts / 0.003)
            s = int(tt * SR)
            e = min(N, s + L, int(END * SR))
            pan = 0.5 + 0.35 * np.sin(k * 0.9)
            arp[s:e, 0] += pl[:e - s] * np.cos(pan * np.pi / 2)
            arp[s:e, 1] += pl[:e - s] * np.sin(pan * np.pi / 2)
            k += 1
            tt += BEAT / 2
        # sub pulse on beats 1 and 3
        tt = t0
        while tt < t1 - 1e-6:
            L = int(0.5 * SR)
            ts = np.arange(L) / SR
            p = np.sin(2 * np.pi * midi(notes[0] - 12) * ts) * np.exp(-ts / 0.18) * np.minimum(1, ts / 0.01)
            s = int(tt * SR)
            e = min(N, s + L, int(END * SR))
            sub[s:e] += p[:e - s]
            tt += 2 * BEAT
    t = np.arange(N) / SR
    for c in range(2):
        pad[:, c] = lowpass_sweep(pad[:, c], t, 350.0, 1400.0, 1 / 20.0)
    mix = 0.55 * pad / np.max(np.abs(pad)) + 0.22 * arp / np.max(np.abs(arp)) + 0.35 * (sub / np.max(np.abs(sub)))[:, None]
    # intro fade 0-1.5 s, end: 60.0 s cut with a 0.25 s tail, silent under the disclaimer
    mix[: int(1.5 * SR)] *= np.linspace(0, 1, int(1.5 * SR))[:, None] ** 2
    e0, e1 = int(END * SR), int((END + 0.25) * SR)
    mix[e0:e1] *= np.linspace(1, 0, e1 - e0)[:, None]
    mix[e1:] = 0
    mix *= 0.5 / np.max(np.abs(mix))
    out = os.path.join(OUT_AUDIO, "music_%s.wav" % TODAY)
    sf.write(out, mix.astype(np.float32), SR, subtype="PCM_24")
    log("music bed %.1f s, %d chords, %g BPM -> %s" % (N / SR, len(PROG), BPM, rel(out)))
    log("log", rel(log.path))


if __name__ == "__main__":
    main()
