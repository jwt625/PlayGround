"""Pick a loopable soundtrack section that fits a video length, with at most a given speedup (pitch kept).

Beat-tracks the track (librosa), forms bars of 4 beats (bar phase chosen by onset strength), and scores every
section that starts on a bar line after --skip seconds, spans a whole number of bars, and has a length L with
D <= L <= D * max_speed (D = video duration). Loop score = cosine similarity of chroma + MFCC + RMS in a window
right after the section start vs right after its end (where a loop would continue), plus loudness match at both
boundaries. Prints the ranking and writes the best (start, length, tempo) as JSON.

Usage: uv run --with librosa python scripts/video/pick_loop.py AUDIO VIDEO_DURATION_S [--skip 10] [--max-speed 1.25]
"""

import argparse
import json

import librosa
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("duration", type=float)
ap.add_argument("--skip", type=float, default=10.0)
ap.add_argument("--max-speed", type=float, default=1.25)
ap.add_argument("--win", type=float, default=3.0)
a = ap.parse_args()

y, sr = librosa.load(a.audio, sr=22050, mono=True)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="frames")
onset = librosa.onset.onset_strength(y=y, sr=sr)
phase = int(np.argmax([onset[beats[p::4]].mean() for p in range(4)]))
bars = librosa.frames_to_time(beats[phase::4], sr=sr)
hop = 512
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
rms = librosa.feature.rms(y=y, hop_length=hop)[0]
ft = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop)


def feat(t0):
    m = (ft >= t0) & (ft < t0 + a.win)
    if not m.any():
        return None
    v = np.concatenate([chroma[:, m].mean(1), (mfcc[:, m].mean(1) - mfcc.mean(1)) / (mfcc.std(1) + 1e-6),
                        [rms[m].mean() / (rms.mean() + 1e-9)]])
    return v / (np.linalg.norm(v) + 1e-9), rms[m].mean()


cands = []
T = len(y) / sr
for i, s in enumerate(bars):
    if s < a.skip:
        continue
    for e in bars[i + 1:]:
        L = e - s
        if L < a.duration or L > a.duration * a.max_speed or e + a.win > T:
            continue
        fs, fe = feat(s), feat(e)
        if fs is None or fe is None:
            continue
        sim = float(fs[0] @ fe[0])
        loud = float(1 - abs(np.log((fs[1] + 1e-6) / (fe[1] + 1e-6))))
        cands.append({"start": round(float(s), 3), "length": round(float(L), 3),
                      "tempo": round(float(L / a.duration), 4), "loop_sim": round(sim, 4),
                      "loud_match": round(float(loud), 3), "score": round(float(sim + 0.3 * loud), 4)})
cands.sort(key=lambda c: -c["score"])
print(json.dumps({"bpm": float(np.atleast_1d(tempo)[0]), "n_bars": len(bars), "top": cands[:6]}, indent=1))
print("BEST", json.dumps(cands[0]))
