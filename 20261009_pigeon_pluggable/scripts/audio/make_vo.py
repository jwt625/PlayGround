"""Voice-over track: Kokoro-82M (kokoro-onnx, local, Apache-2.0) -> per-line takes -> laid out on the film clock -> one wav.

Run (project root):
  KOKORO_DIR=<dir with kokoro-v1.0.onnx + voices-v1.0.bin> \
  uv run --no-project --with kokoro-onnx --with soundfile --with numpy python scripts/audio/make_vo.py [--voice am_liam] [--dry-run]

Reads scripts/audio/narration.json (say/show/anchor per line), writes back t0/t1/dur (film seconds; the master clock for
scene beats, DevLog-010 P6), writes outputs/audio/vo_<date>.wav (48 kHz mono, -16 LUFS, TP <= -1.5 dBTP) and
outputs/audio/vo_lines/<id>.wav (one trimmed take per line, for the Whisper check).

Layout: every line is read at the same speed (narration.json "speed"); a line starts at max(scene start + anchor,
previous line end + breath); if a scene overflows (last line must end 0.12 s before the scene end) the lines are pulled
earlier (backward pass, keeping the breath). Lines are never time-stretched, except the S7 disclaimer ("fast": true),
which is read faster and squeezed into its window like an ad disclaimer.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import HERE, OUT_AUDIO, SCENE_BOUNDS, TODAY, TOTAL, Log, master, rel  # noqa: E402

KSR = 24000
NARR = os.path.join(HERE, "narration.json")
TAIL_GUARD = 0.12     # last line of a scene ends at least this long before the scene end
HEAD_GUARD = 0.15     # first line of a scene starts at least this long after the scene start
DISC_SPEED = 1.45     # disclaimer read speed before the squeeze


def trim(a, thr=0.008, pad=0.02):
    idx = np.where(np.abs(a) > thr)[0]
    if not len(idx):
        return a
    p = int(pad * KSR)
    return a[max(0, idx[0] - p):idx[-1] + 1 + p]


def tail_fix(a, keep=0.04, fade=0.025, win=0.4, thr_db=-32.0):
    """Shorten an over-long final plosive release (Kokoro adds ~100-150 ms of aspiration or a schwa after a final
    t/k, which Whisper hears as 'lights' / 'heat sinker'): find the last closure (10 ms frame under thr_db followed by
    a frame 6 dB louder) in the final `win` s, keep `keep` s after it, fade out. Only for lines flagged "tail_fix"."""
    h = int(0.01 * KSR)
    n = len(a) // h
    db = 20 * np.log10(np.array([np.sqrt(np.mean(a[i * h:(i + 1) * h] ** 2)) for i in range(n)]) + 1e-9)
    cand = [i for i in range(max(0, n - int(win / 0.01)), n - 2) if db[i] < thr_db and db[i + 1:].max() > db[i] + 6]
    if not cand:
        return a, 0.0
    end = (cand[-1] + 1) * h + int(keep * KSR)
    b = a[:end].copy()
    f = int(fade * KSR)
    b[-f:] *= np.linspace(1, 0, f)
    return b, (len(a) - end) / KSR


def atempo(a, factor, tmpdir):
    if abs(factor - 1) < 1e-3:
        return a
    stages, f = [], factor
    while f > 2.0:
        stages.append(2.0)
        f /= 2.0
    stages.append(f)
    ti, to = os.path.join(tmpdir, "_atempo_in.wav"), os.path.join(tmpdir, "_atempo_out.wav")
    sf.write(ti, a, KSR)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", ti, "-af", ",".join("atempo=%.4f" % s for s in stages), to], check=True)
    b, _ = sf.read(to)
    os.remove(ti)
    os.remove(to)
    return b


def write_narration(doc):
    """Keep narration.json hand-editable: header keys, then one line per narration line."""
    head = {k: v for k, v in doc.items() if k != "lines"}
    s = "{\n" + ",\n".join(" %s: %s" % (json.dumps(k), json.dumps(v, ensure_ascii=False)) for k, v in head.items())
    s += ',\n "lines": [\n' + ",\n".join("  " + json.dumps(L, ensure_ascii=False) for L in doc["lines"]) + "\n ]\n}\n"
    with open(NARR, "w") as f:
        f.write(s)


def layout(lines, breath):
    """Scene-wise placement; returns nothing, fills t0/t1 in place. Raises if a scene cannot fit."""
    scenes = sorted({L["scene"] for L in lines if not L.get("fast")})
    for sc in scenes:
        Ls = [L for L in lines if L["scene"] == sc and not L.get("fast")]
        s0, s1 = SCENE_BOUNDS[sc - 1]
        prev = s0 + HEAD_GUARD - breath
        for L in Ls:                                   # forward pass: anchors, breath
            L["t0"] = max(s0 + L["anchor"], prev + breath)
            L["t1"] = L["t0"] + L["dur"]
            prev = L["t1"]
        limit = s1 - TAIL_GUARD                         # backward pass: pull into the scene
        for L in reversed(Ls):
            if L["t1"] > limit:
                L["t1"] = limit
                L["t0"] = limit - L["dur"]
            limit = L["t0"] - breath
        if Ls[0]["t0"] < s0 + HEAD_GUARD - 1e-6:
            need = sum(L["dur"] for L in Ls) + breath * (len(Ls) - 1)
            raise SystemExit("scene %d does not fit: speech %.2f s + breaths in %.2f s" % (sc, need, s1 - s0 - HEAD_GUARD - TAIL_GUARD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=os.environ.get("KOKORO_DIR", os.path.expanduser("~/.cache/kokoro-onnx")))
    ap.add_argument("--voice", default=None)
    ap.add_argument("--dry-run", action="store_true", help="synthesise and print the layout; do not write json or wavs")
    ap.add_argument("--tag", default="", help="output name suffix: vo_<date>_<tag>.wav (e.g. rev3)")
    args = ap.parse_args()
    log = Log("vo")
    doc = json.load(open(NARR))
    voice = args.voice or doc["voice"]
    speed, breath = float(doc["speed"]), float(doc["breath_s"])
    k = Kokoro(os.path.join(args.model_dir, "kokoro-v1.0.onnx"), os.path.join(args.model_dir, "voices-v1.0.bin"))
    lines_dir = os.path.join(OUT_AUDIO, "vo_lines")
    os.makedirs(lines_dir, exist_ok=True)
    log("voice %s speed %.2f breath %.2f model_dir $KOKORO_DIR" % (voice, speed, breath))

    takes = {}
    for L in doc["lines"]:
        sp = float(L.get("speed", DISC_SPEED)) if L.get("fast") else float(L.get("speed", speed))   # per-line speed: max ~1.2x of base (owner rule)
        if L.get("phonemes"):
            a, _ = k.create(L["phonemes"], voice=voice, speed=sp, lang="en-us", is_phonemes=True)
        else:
            a, _ = k.create(L["say"], voice=voice, speed=sp, lang="en-us")
        a = trim(np.asarray(a, dtype=np.float64))
        if L.get("tail_fix"):        # true -> keep 40 ms after the last closure; a number -> keep that many seconds
            tf = L["tail_fix"]
            a, cut = tail_fix(a, keep=0.04 if tf is True else float(tf))
            log("%s tail_fix: cut %.0f ms of final release" % (L["id"], cut * 1000))
        if L.get("fast"):
            w0, w1 = L["window"]
            d = len(a) / KSR
            if d > (w1 - w0):
                log("%s fast: natural %.2f s squeezed x%.2f into %.2f s (atempo, pitch kept)" % (L["id"], d, d / (w1 - w0) * 1.01, w1 - w0))
                a = trim(atempo(a, d / (w1 - w0) * 1.01, lines_dir))
            L["t0"], L["t1"] = w0, w0 + len(a) / KSR
        L["dur"] = round(len(a) / KSR, 3)
        takes[L["id"]] = a
    layout(doc["lines"], breath)
    for L in doc["lines"]:
        L["t0"], L["t1"] = round(L["t0"], 3), round(L["t0"] + L["dur"], 3)

    # report: per line, and per scene speech density / gaps
    nwords = 0
    for L in doc["lines"]:
        sc0 = SCENE_BOUNDS[L["scene"] - 1][0]
        w = len(L["say"].replace("-", " ").split())
        nwords += 0 if L.get("fast") else w
        log("%-4s S%d %6.2f-%6.2f (scene %5.2f-%5.2f) dur %4.2f  %4.1f w/s  anchor %s%s  | %s" % (
            L["id"], L["scene"], L["t0"], L["t1"], L["t0"] - sc0, L["t1"] - sc0, L["dur"], w / L["dur"],
            L["anchor"], " FAST" if L.get("fast") else "", L["say"]))
    speech = sum(L["dur"] for L in doc["lines"] if not L.get("fast"))
    log("speech %.2f s over 60 s (%.0f %%), %.2f words/s overall (non-fast)" % (speech, speech / 60 * 100, nwords / speech))
    if args.dry_run:
        log("dry run: nothing written")
        return

    mix = np.zeros(int(TOTAL * KSR))
    for L in doc["lines"]:
        a = takes[L["id"]]
        s = int(round(L["t0"] * KSR))
        mix[s:s + len(a)] += a[:len(mix) - s]
        sf.write(os.path.join(lines_dir, "%s.wav" % L["id"]), a.astype(np.float32), KSR)
    pk = np.max(np.abs(mix))
    raw = os.path.join(OUT_AUDIO, "_vo_raw.wav")
    sf.write(raw, (mix / pk * 0.9).astype(np.float32), KSR)
    final = os.path.join(OUT_AUDIO, "vo_%s%s.wav" % (TODAY, "_" + args.tag if args.tag else ""))
    # broadcast-style polish (sibling film): high-pass, presence lift, gentle compression; then one linear gain to -16 LUFS
    master(raw, final, I=-16.0, TP=-1.5,
           chain="highpass=f=90,equalizer=f=3000:t=q:w=1:g=2.5,"
                 "acompressor=threshold=-20dB:ratio=3.5:attack=5:release=60:makeup=4", log=log)
    os.remove(raw)
    doc["voice"] = voice
    write_narration(doc)
    log("wrote", rel(final), "and", rel(NARR), "; takes in", rel(lines_dir))
    log("log", rel(log.path))


if __name__ == "__main__":
    main()
