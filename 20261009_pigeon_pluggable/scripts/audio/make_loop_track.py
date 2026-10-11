"""Cut a seamless 15.000 s loop from a music file and normalize it.

Usage: uv run --with numpy --with soundfile --with scipy python scripts/audio/make_loop_track.py <src> <start_s> <len_s> <out.wav>
Steps: refine the loop length so the waveform after the cut best matches the start (onset-envelope cross-correlation, +-60 ms),
time-stretch with ffmpeg atempo (pitch kept) so the loop is exactly 15.000 s, cut with margins, apply an equal-power circular
crossfade (50 ms) so the last sample flows into the first, then apply one linear gain to -16 LUFS integrated (true peak <= -1.5 dBTP).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import correlate

src, start, L0, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
SR, T, XF, M = 48000, 15.0, 0.05, 1.0
tmp = tempfile.mkdtemp()


def ff(*a):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)


raw = os.path.join(tmp, "raw.wav")
ff("-ss", "%.4f" % (start - M), "-t", "%.4f" % (L0 + 2 * M + 1), "-i", src, "-ac", "2", "-ar", str(SR), raw)
y, _ = sf.read(raw)
mono = y.mean(1)


def env(x):
    e = np.abs(np.diff(x, prepend=0.0))
    k = int(0.005 * SR)
    return np.convolve(e, np.ones(k) / k, mode="same")


# refine loop length: match [start, start+1 s] against [start+L, start+L+1 s] for L = L0 +- 60 ms
a = env(mono[int(M * SR):int((M + 1.0) * SR)])
best = None
for d in range(-int(0.06 * SR), int(0.06 * SR) + 1, 24):
    i0 = int((M + L0) * SR) + d
    b = env(mono[i0:i0 + len(a)])
    c = float(np.dot(a - a.mean(), b - b.mean()) / (np.linalg.norm(a - a.mean()) * np.linalg.norm(b - b.mean()) + 1e-12))
    if best is None or c > best[0]:
        best = (c, d)
L = L0 + best[1] / SR
r = L / T
print("refined loop length %.4f s (shift %+.1f ms, envelope corr %.3f); atempo %.5f" % (L, 1000 * best[1] / SR, best[0], r))
st = os.path.join(tmp, "st.wav")
ff("-i", raw, "-filter:a", "atempo=%.6f" % r, "-ar", str(SR), st)
z, _ = sf.read(st)
s0 = int(round(M / r * SR))
N, X = int(T * SR), int(XF * SR)
loop = z[s0:s0 + N].copy()
tail = z[s0 + N:s0 + N + X]
t = np.linspace(0, np.pi / 2, X)[:, None]
loop[:X] = loop[:X] * np.sin(t) + tail * np.cos(t)
# loudness: measure with ffmpeg ebur128, apply one linear gain
pre = os.path.join(tmp, "pre.wav")
sf.write(pre, loop, SR, subtype="FLOAT")
res = subprocess.run(["ffmpeg", "-hide_banner", "-i", pre, "-af", "loudnorm=I=-16:TP=-1.5:print_format=json", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
js = json.loads(res[res.rfind("{"):res.rfind("}") + 1])
gain_db = -16.0 - float(js["input_i"])
gain_db = min(gain_db, -1.5 - float(js["input_tp"]))
loop *= 10 ** (gain_db / 20)
sf.write(out, loop, SR, subtype="PCM_24")
seam = np.abs(loop[0] - loop[-1]).max()
typ = np.median(np.abs(np.diff(loop, axis=0)).max(1))
print("input %.1f LUFS, %.1f dBTP; gain %+.2f dB; samples %d (%.3f s); seam step %.4f vs median step %.4f"
      % (float(js["input_i"]), float(js["input_tp"]), gain_db, len(loop), len(loop) / SR, seam, typ))
