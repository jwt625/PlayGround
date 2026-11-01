"""Word timestamps for short windows of the Marvell CPO narration (faster-whisper).

Usage: transcribe.py <audio> <start_s> <dur_s> [<start_s> <dur_s> ...]
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
src = ROOT / sys.argv[1]
wins = list(zip(map(float, sys.argv[2::2]), map(float, sys.argv[3::2])))
model = WhisperModel("small.en", device="cpu", compute_type="int8")
for t0, dur in wins:
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(t0), "-t", str(dur), "-i", str(src), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    segs, _ = model.transcribe(np.frombuffer(pcm, np.float32), word_timestamps=True, language="en")
    print(f"== window {t0:.1f}+{dur:.1f}s")
    for s in segs:
        print("  " + " ".join(f"{w.word.strip()}[{t0 + w.start:.2f}-{t0 + w.end:.2f}]" for w in s.words))
