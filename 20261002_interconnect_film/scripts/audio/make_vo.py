"""Narration track for the film: Kokoro-82M (local ONNX TTS) -> timed segments -> one processed wav.

Run: uv run --no-project --with kokoro-onnx --with soundfile --with numpy python scripts/audio/make_vo.py <model_dir> <out_dir> [voice]
<model_dir> holds kokoro-v1.0.onnx and voices-v1.0.bin (kokoro-onnx release model-files-v1.0, Apache-2.0).
Segments follow DevLog-001 Section 5 and the asm.narr windows of each scene (film time = scene index*10 s + scene time).
Each segment is synthesised, trimmed of silence, fitted into its window (sped up with atempo when too long, never slowed), and mixed.
The S7 disclaimer is read at the hard speed-up of a fast-talking ad disclaimer.
"""
import json
import os
import subprocess
import sys
import wave

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

model_dir, out_dir = sys.argv[1], sys.argv[2]
voice = sys.argv[3] if len(sys.argv) > 3 else "am_liam"
SR = 24000
BASE_SPEED = 1.15  # fast, crisp read (style bible: fast confident young male voice)

# (film start s, film end s, text, max speed-up allowed)
NARR = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "narration.json")))
SEGS = [(d["t0"], d["t1"], d["say"]) for d in NARR["lines"]]
DISC = (NARR["disclaimer"]["t0"], NARR["disclaimer"]["t1"], NARR["disclaimer"]["say"])
TOTAL = 62.0

os.makedirs(out_dir, exist_ok=True)
k = Kokoro(os.path.join(model_dir, "kokoro-v1.0.onnx"), os.path.join(model_dir, "voices-v1.0.bin"))


def trim(a, thr=0.01):
    idx = np.where(np.abs(a) > thr)[0]
    return a[idx[0]:idx[-1] + 1] if len(idx) else a


def atempo(a, factor):
    """Pitch-preserving speed-up via ffmpeg atempo (chained, each stage 0.5..2.0)."""
    if abs(factor - 1) < 1e-3:
        return a
    stages, f = [], factor
    while f > 2.0:
        stages.append(2.0)
        f /= 2.0
    stages.append(f)
    tmp_in, tmp_out = os.path.join(out_dir, "_t_in.wav"), os.path.join(out_dir, "_t_out.wav")
    sf.write(tmp_in, a, SR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp_in, "-af", ",".join("atempo=%.4f" % s for s in stages), tmp_out], check=True)
    b, _ = sf.read(tmp_out)
    os.remove(tmp_in)
    os.remove(tmp_out)
    return b


mix = np.zeros(int(TOTAL * SR))
log = []


def place(t0, t1, text, speed, force_fit=False):
    a, _ = k.create(text, voice=voice, speed=speed, lang="en-us")
    a = trim(np.asarray(a, dtype=np.float64))
    win = t1 - t0
    dur = len(a) / SR
    if not force_fit and dur < win * 0.92:  # keep the voice going: slow the read slightly (floor 1.0) to fill the window
        sp = max(1.0, speed * dur / (win * 0.97))
        a, _ = k.create(text, voice=voice, speed=sp, lang="en-us")
        a = trim(np.asarray(a, dtype=np.float64))
        dur = len(a) / SR
    if dur > win or force_fit:
        a = trim(atempo(a, dur / win * (0.98 if force_fit else 0.97)))
    start = int(t0 * SR)
    mix[start:start + len(a)] += a[:len(mix) - start]
    log.append((t0, text, round(dur, 2), round(len(a) / SR, 2), round(win, 2)))


for t0, t1, text in SEGS:
    place(t0, t1, text, BASE_SPEED)
place(*DISC, speed=float(os.environ.get("DISC_SPEED", "1.5")), force_fit=True)
for row in log:
    print("t=%.2f dur %.2f -> %.2f (win %.2f, x%.2f)  %s" % (row[0], row[2], row[3], row[4], row[2] / row[3], row[1]))
mix /= max(1e-9, np.max(np.abs(mix)))
raw = os.path.join(out_dir, "vo_raw.wav")
sf.write(raw, mix * 0.9, SR)
# broadcast-style polish: high-pass, gentle presence lift, compressor, loudness, 48 kHz
final = os.path.join(out_dir, "vo_%s.wav" % voice)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-af",
                "highpass=f=90,equalizer=f=3000:t=q:w=1:g=2.5,acompressor=threshold=-20dB:ratio=3.5:attack=5:release=60:makeup=4,"
                "loudnorm=I=-16:TP=-1.5:LRA=7,aresample=48000", final], check=True)
os.remove(raw)
print("wrote", final)
