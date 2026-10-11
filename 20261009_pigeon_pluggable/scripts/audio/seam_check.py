"""Compare loop seams with longer windows before and after the cut.
Usage: uv run ... python scripts/audio/seam_check.py <excerpt.wav> <offset_s> <start_abs:len_s> [...]
Score: chroma/mfcc cosine of [end-W, end] vs [start-W, start] (what precedes) and [end, end+W] vs [start, start+W] (what follows).
"""
import sys

import librosa
import numpy as np

wav, off = sys.argv[1], float(sys.argv[2])
y, sr = librosa.load(wav, sr=22050, mono=True)
hop = 512
ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
rms = librosa.feature.rms(y=y, hop_length=hop)[0]


def seg(f, t0, t1):
    return f[..., librosa.time_to_frames(t0 - off, sr=sr, hop_length=hop):librosa.time_to_frames(t1 - off, sr=sr, hop_length=hop)]


def cos(a, b):
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n], b[..., :n]
    return float(np.mean(np.sum(a * b, 0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-9)))


for arg in sys.argv[3:]:
    s, L = map(float, arg.split(":"))
    e = s + L
    out = []
    for W in (1.0, 3.0):
        post = 0.5 * cos(seg(ch, e, e + W), seg(ch, s, s + W)) + 0.5 * cos(seg(mf, e, e + W), seg(mf, s, s + W))
        pre = 0.5 * cos(seg(ch, e - W, e), seg(ch, s - W, s)) + 0.5 * cos(seg(mf, e - W, e), seg(mf, s - W, s))
        out.append("W=%.0fs post %.3f pre %.3f" % (W, post, pre))
    r = np.mean(seg(rms, e - 1, e)) / np.mean(seg(rms, s, s + 1))
    print("start %.3f len %.3f stretch %.3f | %s | rms(end)/rms(start) %.2f" % (s, L, L / 15, " | ".join(out), r))
