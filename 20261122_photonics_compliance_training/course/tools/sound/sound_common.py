"""Shared helpers for the course sound tools: ffmpeg decode/encode, BS.1770 loudness, file probing."""
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent.parent                   # .../course
PROJECT = COURSE.parent
SOUND_DIR = COURSE / "media" / "sound"
SAMPLES_DIR = PROJECT / "research" / "sound_samples"
CACHE = HERE / ".cache"


def decode(path, sr, channels):
    """Decode any audio file to float64 array (n,) or (n, ch) via ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).astype(np.float64)
    return x.reshape(-1, channels) if channels > 1 else x


def encode_mp3(x, sr, out, bitrate):
    """Encode float array (n,) or (n, ch) to mp3 with libmp3lame (LAME header carries encoder delay for gapless decode)."""
    x = np.asarray(x, np.float32)
    ch = 1 if x.ndim == 1 else x.shape[1]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", str(ch), "-i", "-",
         "-c:a", "libmp3lame", "-b:a", bitrate, "-map_metadata", "-1", str(out)],
        input=x.tobytes(), check=True)


def _k_weight(x, sr):
    """ITU-R BS.1770-4 K-weighting (pre-filter shelf + RLB high-pass), coefficients derived for any sample rate."""
    # stage 1: high shelf
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = np.tan(np.pi * f0 / sr)
    Vh, Vb = 10 ** (G / 20), (10 ** (G / 20)) ** 0.4996667741545416
    a0 = 1 + K / Q + K * K
    b1 = [(Vh + Vb * K / Q + K * K) / a0, 2 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0]
    a1 = [1, 2 * (K * K - 1) / a0, (1 - K / Q + K * K) / a0]
    # stage 2: high pass
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = np.tan(np.pi * f0 / sr)
    a2 = [1, 2 * (K * K - 1) / (1 + K / Q + K * K), (1 - K / Q + K * K) / (1 + K / Q + K * K)]
    b2 = [1, -2, 1]
    return lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)


def loudness(x, sr):
    """Return dict: integrated LUFS (gated), momentary max LUFS (400 ms), sample peak dBFS."""
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    pad = np.zeros((int(0.4 * sr), x.shape[1]))
    y = _k_weight(np.vstack([pad, x, pad]), sr)
    ms = (y ** 2).sum(1)                       # channel weights 1.0 for L, R, mono
    win, hop = int(0.4 * sr), int(0.1 * sr)
    c = np.concatenate([[0.0], np.cumsum(ms)])
    starts = np.arange(0, len(ms) - win + 1, hop)
    blk = (c[starts + win] - c[starts]) / win
    lk = -0.691 + 10 * np.log10(blk + 1e-20)
    g = blk[lk > -70]
    integ = None
    if len(g):
        rel = -0.691 + 10 * np.log10(g.mean()) - 10
        g2 = g[(-0.691 + 10 * np.log10(g)) > rel]
        integ = -0.691 + 10 * np.log10(g2.mean())
    return {"I": integ, "M_max": float(lk.max()), "peak_dbfs": float(20 * np.log10(np.abs(x).max() + 1e-12))}


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration,bit_rate:stream=sample_rate,channels,codec_name",
         "-of", "json", str(path)], capture_output=True, check=True, text=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    return {"duration": round(float(d["format"]["duration"]), 3), "sample_rate": int(s["sample_rate"]),
            "channels": int(s["channels"]), "codec": s["codec_name"], "bit_rate": int(d["format"]["bit_rate"])}


def ffmpeg_ebur128(path):
    """Independent check with ffmpeg's ebur128 filter: integrated LUFS and true peak of a file."""
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    summ = err.split("Summary:")[-1]
    vals = {}
    for line in summ.splitlines():
        line = line.strip()
        if line.startswith("I:"):
            vals["I"] = float(line.split()[1])
        elif line.startswith("Peak:"):
            vals["TP"] = float(line.split()[1])
    return vals


def encode_mp3_target(x, sr, out, bitrate, metric, target, peak_cap_db=None):
    """Encode, measure the decoded file, correct the gain once for the codec's level change, re-encode.
    metric: 'I' (integrated) or 'M_max' (max momentary). Returns the final gain factor applied to x."""
    x = np.asarray(x, np.float64)
    ch = 1 if x.ndim == 1 else x.shape[1]
    gain = 1.0
    for _ in range(2):
        encode_mp3(x * gain, sr, out, bitrate)
        L = loudness(decode(out, sr, ch), sr)
        err = target - L[metric]
        if abs(err) < 0.05:
            break
        gain *= 10 ** (err / 20)
        if peak_cap_db is not None:
            gain = min(gain, 10 ** (peak_cap_db / 20) / np.abs(x).max())
    encode_mp3(x * gain, sr, out, bitrate)
    return gain
