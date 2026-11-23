"""Shared Kokoro synthesis + clip processing for the course narration (voice lane, DevLog-003).

Chain per clip: pronounce.say(text) -> Kokoro-82M (kokoro-onnx, CPU, bounded threads) -> trim silence -> light tail fix
(final plosive release) -> gain to TARGET_LUFS (BS.1770 via pyloudnorm) -> ffmpeg: high-pass 70 Hz, limiter at
LIMIT_DBFS -> mono MP3 (libmp3lame CBR) piped from memory (no wav intermediates).
Needs: kokoro-onnx, numpy, pyloudnorm (uv run --no-project --with kokoro-onnx --with pyloudnorm ...).
"""
import os
import re
import subprocess

import numpy as np

KSR = 24000
MODEL_DIR = os.environ.get("KOKORO_DIR", os.path.expanduser("~/.cache/kokoro-onnx"))
TARGET_LUFS = -17.6   # pre-limiter target; measured -18.0 LUFS median on the encoded pilot clips (limiter + high-pass take ~0.4 dB)
LIMIT_DBFS = -1.5
MP3_KBPS = 64
LEAD_PAD = 0.03
TAIL_PAD = 0.08


def load_kokoro(threads=4):
    import onnxruntime as rt
    from kokoro_onnx import Kokoro
    so = rt.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = rt.InferenceSession(os.path.join(MODEL_DIR, "kokoro-v1.0.onnx"), sess_options=so, providers=["CPUExecutionProvider"])
    return Kokoro.from_session(sess, os.path.join(MODEL_DIR, "voices-v1.0.bin"))


def trim(a, thr=0.006, lead=LEAD_PAD, tail=TAIL_PAD):
    idx = np.where(np.abs(a) > thr)[0]
    if not len(idx):
        return a
    return a[max(0, idx[0] - int(lead * KSR)):idx[-1] + 1 + int(tail * KSR)]


def frame_db(a, hop=0.01):
    h = int(hop * KSR)
    n = len(a) // h
    return 20 * np.log10(np.sqrt(np.mean(a[:n * h].reshape(n, h) ** 2, axis=1)) + 1e-9)


def tail_fix(a, keep=0.06, fade=0.025, win=0.35, thr_db=-32.0, min_cut=0.06):
    """Shorten an over-long final plosive release (Kokoro adds 100-150 ms of aspiration or a schwa after a final t/k;
    pigeon DevLog-014). Finds the last closure (10 ms frame under thr_db, relative to the clip peak frame, followed by
    a frame 6 dB louder) in the final `win` s, keeps `keep` s after it, fades out. Only cuts if >= min_cut s would go."""
    db = frame_db(a)
    db = db - db.max()
    n = len(db)
    h = int(0.01 * KSR)
    cand = [i for i in range(max(0, n - int(win / 0.01)), n - 2) if db[i] < thr_db and db[i + 1:].max() > db[i] + 6]
    if not cand:
        return a, 0.0
    end = (cand[-1] + 1) * h + int(keep * KSR)
    if (len(a) - end) / KSR < min_cut:
        return a, 0.0
    b = a[:end].copy()
    f = int(fade * KSR)
    b[-f:] *= np.linspace(1, 0, f)
    return b, (len(a) - end) / KSR


PLOSIVE_END = re.compile(r"(?:[^aeiou\W]|\b)[tkp]\W*$", re.I)


def lufs(a):
    import pyloudnorm as pyln
    m = pyln.Meter(KSR)
    x = a if len(a) >= KSR else np.concatenate([a, np.zeros(KSR - len(a))])   # gating drops the zero blocks
    return m.integrated_loudness(x.astype(np.float64))


def synth_post(raw, say, tailfix=True):
    """Kokoro output -> trimmed, tail-fixed, gain to TARGET_LUFS (float64; limiter is applied at encode)."""
    a = trim(np.asarray(raw, dtype=np.float64))
    raw_dur = len(a) / KSR
    cut = 0.0
    if tailfix and PLOSIVE_END.search(say):
        a, cut = tail_fix(a)
    L = lufs(a)
    g = TARGET_LUFS - L if np.isfinite(L) else 0.0
    a = a * 10 ** (g / 20)
    return a, {"raw_s": round(raw_dur, 3), "tail_cut_s": round(cut, 3), "lufs_in": round(float(L), 2), "gain_db": round(float(g), 2)}


def render(k, tok, text, voice, speed, use_map=True, tailfix=True):
    """Original text -> processed clip. use_map=False synthesizes the raw text (for A/B of the pronunciation map)."""
    import pronounce
    if not use_map:
        raw, _ = k.create(text, voice=voice, speed=speed, lang="en-us")
        return synth_post(raw, text, tailfix) + ({"say": text, "ipa": False},)
    s = pronounce.say_text(text)
    if pronounce.needs_ipa(s):
        raw, _ = k.create(pronounce.to_phonemes(s, tok), voice=voice, speed=speed, lang="en-us", is_phonemes=True)
    else:
        raw, _ = k.create(s, voice=voice, speed=speed, lang="en-us")
    return synth_post(raw, s, tailfix) + ({"say": s, "ipa": pronounce.needs_ipa(s)},)


def encode_mp3(a, path, kbps=MP3_KBPS):
    """float audio at KSR -> mono MP3 at KSR via ffmpeg stdin (no intermediate file)."""
    tmp = path + ".part"
    af = "highpass=f=70,alimiter=limit=%.4f:attack=2:release=60:level=false" % (10 ** (LIMIT_DBFS / 20))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(KSR), "-ac", "1", "-i", "pipe:0", "-af", af,
                    "-ac", "1", "-ar", str(KSR), "-c:a", "libmp3lame", "-b:a", "%dk" % kbps, "-f", "mp3", tmp],
                   input=a.astype("<f4").tobytes(), check=True)
    os.replace(tmp, path)


def measure_file(path):
    """ebur128 integrated loudness and true peak of an encoded file (ffmpeg)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    I = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    tp = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", r.stderr)
    return (float(I[-1]) if I else None), (float(tp[-1]) if tp else None)
