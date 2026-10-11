"""Find a beat/bar-aligned loop in a music excerpt.

Usage: uv run --with librosa --with soundfile python scripts/audio/find_loop.py <excerpt.wav> <excerpt_offset_s> <target_start_s>
Prints tempo, beats near the target, and ranked loop candidates (start, length, stretch to 15 s, seam score).
"""
import sys

import librosa
import numpy as np

wav, off, target = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
y, sr = librosa.load(wav, sr=22050, mono=True)
tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="frames")
bt = librosa.frames_to_time(beats, sr=sr)
tempo = float(np.atleast_1d(tempo)[0])
print("tempo %.2f BPM, beat period %.4f s, %d beats" % (tempo, 60 / tempo, len(bt)))
hop = 512
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop)
rms = librosa.feature.rms(y=y, hop_length=hop)[0]
onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)


def win(feat, t, dur=1.0):
    a = librosa.time_to_frames(t, sr=sr, hop_length=hop)
    b = librosa.time_to_frames(t + dur, sr=sr, hop_length=hop)
    return feat[..., a:b]


def sim(t0, t1):
    """How well audio continuing past t1 matches audio at t0 (cosine of chroma + mfcc, rms ratio)."""
    c0, c1 = win(chroma, t0), win(chroma, t1)
    m0, m1 = win(mfcc, t0), win(mfcc, t1)
    n = min(c0.shape[-1], c1.shape[-1])
    if n < 5:
        return -1
    cc = np.mean(np.sum(c0[:, :n] * c1[:, :n], 0) / (np.linalg.norm(c0[:, :n], axis=0) * np.linalg.norm(c1[:, :n], axis=0) + 1e-9))
    mm = np.mean(np.sum(m0[:, :n] * m1[:, :n], 0) / (np.linalg.norm(m0[:, :n], axis=0) * np.linalg.norm(m1[:, :n], axis=0) + 1e-9))
    r0, r1 = np.mean(win(rms, t0)), np.mean(win(rms, t1))
    rr = min(r0, r1) / max(r0, r1)
    return 0.45 * cc + 0.35 * mm + 0.2 * rr


# beat strength per beat to guess downbeats (bar phase with the strongest mean onset at every 4th beat)
bs = onset[beats]
phase = int(np.argmax([bs[p::4].mean() for p in range(4)]))
near = [(k, off + t) for k, t in enumerate(bt) if abs(off + t - target) < 6]
print("beats near target (abs s, * = estimated downbeat):")
print(" ".join("%.2f%s" % (t, "*" if (k - phase) % 4 == 0 else "") for k, t in near))
cands = []
for k, t0 in enumerate(bt):
    if abs(off + t0 - target) > 6:
        continue
    for n in range(8, 64):
        if k + n >= len(bt):
            break
        L = bt[k + n] - t0
        if not (15.0 <= L <= 18.75):
            continue
        bar_ok = (n % 4 == 0) and ((k - phase) % 4 == 0)
        s = sim(t0, bt[k + n])
        cands.append((s + (0.05 if bar_ok else 0.0) + (0.03 if n % 16 == 0 else 0.0) + (0.02 if n % 8 == 0 else 0.0),
                      s, off + t0, n, L, L / 15.0, bar_ok))
cands.sort(reverse=True)
print("rank score  seam  start_abs  beats  len_s  stretch  bar_aligned")
for i, c in enumerate(cands[:12]):
    print("%2d   %.3f  %.3f  %8.3f  %5d  %6.3f  %.4f  %s" % (i, c[0], c[1], c[2], c[3], c[4], c[5], c[6]))
