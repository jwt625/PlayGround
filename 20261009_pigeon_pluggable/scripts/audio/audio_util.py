"""Shared helpers for the audio scripts: paths, timestamped logs, ffmpeg loudness (EBU R128) measure and linear normalise.

Imported by make_vo.py, make_sfx.py, make_music.py, mix.py, check_vo.py (all run with `uv run --no-project ...` from the
project root). No third-party imports here except numpy/soundfile where noted.
"""
import datetime
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
OUT_AUDIO = os.path.join(PROJ, "outputs", "audio")
OUT_LOGS = os.path.join(PROJ, "outputs", "logs")
SCENE_LEN = 10.0
try:   # film length and scene windows follow the narration source (rev 5: 63.0 s, S5 11 s); fallback: 62 s, 10 s scenes
    _N = json.load(open(os.path.join(HERE, "narration.json")))
except (OSError, ValueError):
    _N = {}
TOTAL = float(_N.get("total_s", 62.0))
SCENE_BOUNDS = [tuple(b) for b in _N.get("scene_bounds", [[10.0 * i, 10.0 * (i + 1)] for i in range(6)] + [[60.0, 62.0]])]
SR = 48000
TODAY = datetime.date.today().strftime("%Y%m%d")


class Log:
    """print + append to outputs/logs/audio_<name>_<YYYYmmdd_HHMMSS>.log"""

    def __init__(self, name):
        os.makedirs(OUT_LOGS, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(OUT_LOGS, "audio_%s_%s.log" % (name, stamp))
        self.f = open(self.path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()


def rel(p):
    return os.path.relpath(p, PROJ)


def run(cmd, log=None):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        msg = "command failed: %s\n%s" % (" ".join(cmd), r.stderr[-3000:])
        if log:
            log(msg)
        raise RuntimeError(msg)
    return r


def ebur128(path):
    """Integrated loudness (LUFS), loudness range (LU), true peak (dBTP) of a file via ffmpeg ebur128."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-af", "ebur128=peak=true:framelog=quiet",
                        "-f", "null", "-"], capture_output=True, text=True)
    s = r.stderr
    tail = s[s.rfind("Summary:"):]
    I = float(re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", tail).group(1))
    LRA = float(re.search(r"LRA:\s+(-?[\d.]+) LU", tail).group(1))
    tp = re.search(r"True peak:\s+Peak:\s+(-?[\d.]+|-inf) dBFS", tail)
    TP = float(tp.group(1)) if tp else float("nan")
    return {"I": I, "LRA": LRA, "TP": TP}


def loudnorm_linear(src, dst, I=-16.0, TP=-1.5, LRA=11.0, extra_af="", log=None):
    """Two-pass ffmpeg loudnorm in linear mode (one gain, no dynamic pumping), 48 kHz PCM 24-bit output.

    If linear mode is impossible (gain would push the true peak over TP), loudnorm falls back to dynamic mode by itself;
    we detect that from the second-pass JSON and report it.
    """
    pre = (extra_af + ",") if extra_af else ""
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", src, "-af",
             pre + "loudnorm=I=%g:TP=%g:LRA=%g:print_format=json" % (I, TP, LRA), "-f", "null", "-"], log)
    m = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}") + 1])
    af = (pre + "loudnorm=I=%g:TP=%g:LRA=%g:linear=true:measured_I=%s:measured_TP=%s:measured_LRA=%s:measured_thresh=%s:"
          "offset=%s:print_format=json,aresample=%d" % (I, TP, LRA, m["input_i"], m["input_tp"], m["input_lra"],
                                                        m["input_thresh"], m["target_offset"], SR))
    r2 = run(["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", src, "-af", af, "-c:a", "pcm_s24le", dst], log)
    m2 = json.loads(r2.stderr[r2.stderr.rfind("{"):r2.stderr.rfind("}") + 1])
    if log:
        log("loudnorm %s: in I=%s TP=%s LRA=%s -> type=%s out I=%s TP=%s" % (
            os.path.basename(dst), m["input_i"], m["input_tp"], m["input_lra"], m2.get("normalization_type"),
            m2["output_i"], m2["output_tp"]))
    return m2


def measure_I(src, chain=""):
    r = run(["ffmpeg", "-hide_banner", "-nostats", "-i", src, "-af",
             (chain + "," if chain else "") + "loudnorm=print_format=json", "-f", "null", "-"])
    m = json.loads(r.stderr[r.stderr.rfind("{"):r.stderr.rfind("}") + 1])
    return float(m["input_i"])


def master(src, dst, I=-16.0, TP=-1.5, chain="", log=None):
    """chain -> gain to target I -> oversampled (4x) peak limiter 0.6 dB under TP -> two-pass linear loudnorm trim.

    The 4x-oversampled limiter approximates a true-peak limiter, so the final loudnorm can stay in linear mode.
    """
    I0 = measure_I(src, chain)
    lim = 10 ** ((TP - 0.6) / 20)
    c2 = (chain + "," if chain else "") + ("volume=%.2fdB,aresample=192000,alimiter=limit=%.4f:attack=1:release=60:level=false,"
                                           "aresample=%d" % (I - I0, lim, SR))
    if log:
        log("master %s: chain I=%.2f LUFS, gain %+.2f dB, limiter %.2f dBFS (4x oversampled)" % (
            os.path.basename(dst), I0, I - I0, TP - 0.6))
    return loudnorm_linear(src, dst, I=I, TP=TP, LRA=20, extra_af=c2, log=log)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)
