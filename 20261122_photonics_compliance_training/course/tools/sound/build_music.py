# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy"]
# ///
"""Cut the intro loop and the outro from the cached incompetech tracks; normalize; encode mp3 stereo 128 kbps.

Run from the project root (after fetch_music.py):  uv run course/tools/sound/build_music.py
Outputs:
  course/media/sound/music_intro_loop.mp3   seamless loop from "Life of Riley" (Kevin MacLeod), 24 or 32 bars
  course/media/sound/music_outro.mp3        last phrase(s) of "Cheery Monday" (Kevin MacLeod) with its own ending, 20-40 s
  research/sound_samples/alt_intro_loop_carefree.mp3, alt_outro_easy_lemon_30.mp3   alternates for audition only
  course/tools/sound/logs/music_<timestamp>.json   loop points, seam scores, loudness

Loop method: log-band spectrogram; for loop lengths of N bars (tempo from the onset autocorrelation, searched over
+-1.5 %), find the start s where the audio around s best matches the audio around s + L (cosine similarity over
+-2 bars); refine to the sample by waveform cross-correlation; crossfade the last 60 ms of the loop into the audio
just before s, so the end flows into the start. Gain is one constant per file (integrated loudness to -20 LUFS).
"""
import datetime as dt
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from sound_common import CACHE, HERE, SAMPLES_DIR, SOUND_DIR, decode, encode_mp3_target, loudness  # noqa: E402

SR = 44100
FSR = 22050          # analysis rate
HOP = 512
TARGET_LUFS = -20.0
XFADE = 0.060


def spec_features(mono):
    n = 2048
    fr = np.lib.stride_tricks.sliding_window_view(mono, n)[::HOP] * np.hanning(n)
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / FSR)
    edges = np.geomspace(60, 10000, 49)
    B = np.stack([S[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in zip(edges[:-1], edges[1:])], 1)
    B = np.log10(B + 1e-8)
    B -= B.mean(1, keepdims=True)
    return B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)


def beat_period(mono):
    F = spec_features(mono)
    e = np.maximum(0, np.diff(F, axis=0)).sum(1)
    e = e - np.convolve(e, np.ones(32) / 32, "same")
    ac = np.correlate(e, e, "full")[len(e) - 1:]
    fps = FSR / HOP
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 70)
    k = lo + int(np.argmax(ac[lo:hi]))
    y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
    k = k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)            # parabolic refinement
    return k / fps


def _wave_refine(m, a, b0, search=0.02, win=0.25):
    """Best sample offset b near b0 so that the audio after b matches the audio after a (normalized correlation)."""
    w = int(win * SR)
    ref = m[a:a + w]
    best_c, best_b = -2.0, b0
    for b in range(b0 - int(search * SR), b0 + int(search * SR)):
        seg = m[b:b + w]
        c = float(np.dot(ref, seg) / (np.linalg.norm(ref) * np.linalg.norm(seg) + 1e-12))
        if c > best_c:
            best_c, best_b = c, b
    return best_c, best_b


def find_loop(stereo, bar_options, beats_per_bar=4, s_min=2.0, tail=3.0, n_cand=12):
    mono = decode_mono(stereo)
    F = spec_features(mono)
    fps = FSR / HOP
    T = beat_period(mono)
    bar = beats_per_bar * T
    w = int(round(2 * bar * fps))
    sup = int(bar * fps)
    m = stereo.mean(1)
    cands = []
    for bars in bar_options:
        L0 = bars * bar * fps
        for L in range(int(L0 * 0.985), int(L0 * 1.015) + 1):
            sim = (F[:-L] * F[L:]).sum(1)
            score = np.convolve(sim, np.ones(2 * w) / (2 * w), "same")
            lo = int(s_min * fps) + w
            hi = len(F) - L - w - int(tail * fps)
            if hi <= lo:
                continue
            sc = score[lo:hi].copy()
            for _ in range(n_cand):                      # top peaks with 1-bar suppression
                j = int(np.argmax(sc))
                if sc[j] < 0:
                    break
                cands.append((float(sc[j]), lo + j, L, bars))
                sc[max(0, j - sup):j + sup] = -1
    cands.sort(reverse=True)
    best = None
    for sc, i, L, bars in cands[:60]:
        a = int(i * HOP / FSR * SR)
        c, b = _wave_refine(m, a, int((i + L) * HOP / FSR * SR))
        key = c + 0.5 * sc
        if best is None or key > best[0]:
            best = (key, sc, c, a, b, bars)
    _, sc, c, a, b, bars = best
    return {"start_s": a / SR, "end_s": b / SR, "length_s": (b - a) / SR, "beat_s": T, "bpm": 60 / T,
            "bars": bars, "spec_score": sc, "wave_corr": c, "a": a, "b": b}


def decode_mono(stereo):
    m = stereo.mean(1)
    # resample by decimation-free method: re-decode is cheaper to reason about; use simple FFT resample
    from scipy.signal import resample_poly
    return resample_poly(m, FSR, SR)


def make_loop(stereo, a, b):
    y = stereo[a:b].copy()
    n = int(XFADE * SR)
    ramp = np.linspace(0, 1, n)[:, None]
    y[-n:] = y[-n:] * (1 - ramp) + stereo[a - n:a] * ramp   # end flows into the audio that precedes the start
    return y


def seam_check(y):
    """Correlation and level jump across the loop seam (end -> start)."""
    n = int(0.01 * SR)
    joined = np.vstack([y[-n:], y[:n]]).mean(1)
    d = np.abs(np.diff(joined))
    body = np.abs(np.diff(y.mean(1)))
    return {"seam_max_step": float(d.max()), "body_p99_step": float(np.percentile(body, 99))}


def norm_to(y, target=TARGET_LUFS):
    L = loudness(y, SR)
    g = 10 ** ((target - L["I"]) / 20)
    return y * g, L["I"], 20 * np.log10(g)


def outro_tail(stereo, bpm, want=30.0, lo=20.0, hi=40.0, phrase=8):
    """Start on a phrase downbeat (multiple of `phrase` bars from the first note, which opens the track on beat 1;
    checked 2026-11-22 with an onset grid: low-band onsets peak on beats 1 and 3, claps on 2 and 4) and keep the
    track's own ending."""
    m = np.abs(stereo).max(1)
    nz = np.nonzero(m > 10 ** (-60 / 20))[0]
    first, last = nz[0] / SR, nz[-1]
    end = min(len(stereo), last + int(0.6 * SR))
    bar = 4 * 60 / bpm
    opts = [k for k in range(0, 400, phrase) if lo <= end / SR - (first + k * bar) <= hi]
    k = min(opts, key=lambda k: abs(end / SR - (first + k * bar) - want))
    a = int((first + k * bar - 0.02) * SR)
    y = stereo[a:end].copy()
    n = int(0.02 * SR)
    y[:n] *= np.linspace(0, 1, n)[:, None]
    nf = int(0.3 * SR)
    y[-nf:] *= np.linspace(1, 0, nf)[:, None]
    return y, {"start_s": a / SR, "start_bar": k, "end_s": end / SR, "length_s": (end - a) / SR, "bpm": bpm}


def main():
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {"date": stamp, "target_lufs": TARGET_LUFS, "items": {}}
    jobs = [
        ("intro", "Life of Riley.mp3", (24, 32), SOUND_DIR / "music_intro_loop.mp3"),
        ("alt_intro", "Carefree.mp3", (24,), SAMPLES_DIR / "alt_intro_loop_carefree.mp3"),
    ]
    for key, src, bars, out in jobs:
        x = decode(CACHE / src, SR, 2)
        info = find_loop(x, bars)
        y = make_loop(x, info["a"], info["b"])
        y, lin, gdb = norm_to(y)
        info.update(seam_check(y), source=src, input_lufs=lin, gain_db=gdb, out=str(out.relative_to(HERE.parent.parent.parent)))
        encode_mp3_target(y, SR, out, "128k", "I", TARGET_LUFS)
        log["items"][key] = {k: v for k, v in info.items() if k not in ("a", "b")}
        print(key, json.dumps(log["items"][key], indent=None))
    x = decode(CACHE / "Cheery Monday.mp3", SR, 2)
    y, info = outro_tail(x, 102.0)   # catalog tempo; onset estimate 101.3-102.4
    y, lin, gdb = norm_to(y)
    out = SOUND_DIR / "music_outro.mp3"
    encode_mp3_target(y, SR, out, "128k", "I", TARGET_LUFS)
    info = {k: float(v) for k, v in info.items()}
    info.update(source="Cheery Monday.mp3", input_lufs=float(lin), gain_db=float(gdb))
    log["items"]["outro"] = info
    print("outro", info)
    x = decode(CACHE / "Easy Lemon 30 second.mp3", SR, 2)
    m = np.abs(x).max(1)
    nz = np.nonzero(m > 10 ** (-60 / 20))[0]
    y = x[max(0, nz[0] - int(0.01 * SR)):min(len(x), nz[-1] + int(0.3 * SR))]
    y, lin, gdb = norm_to(y)
    encode_mp3_target(y, SR, SAMPLES_DIR / "alt_outro_easy_lemon_30.mp3", "128k", "I", TARGET_LUFS)
    log["items"]["alt_outro"] = {"source": "Easy Lemon 30 second.mp3", "input_lufs": float(lin), "gain_db": float(gdb), "length_s": len(y) / SR}
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / f"music_{stamp}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
