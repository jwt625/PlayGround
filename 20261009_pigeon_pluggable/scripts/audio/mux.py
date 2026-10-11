"""Mux the final mix onto the scene build's video: H.264 stream copied, AAC 192 kb/s, +faststart.

Run (project root): uv run --no-project python scripts/audio/mux.py [--video outputs/film_v1_<date>.mp4] [--audio outputs/audio/mix_v1_<date>.wav]
Defaults: newest outputs/film_v1_*.mp4 that is not already an _av file, and outputs/audio/mix_v1_<today>.wav.
Output: outputs/film_v1_<date>_av.mp4 (<date> taken from the video name when it has one, else today).
Checks: video and audio durations within 1 frame of each other (warns otherwise), prints ffprobe stream summary and
the loudness of the muxed AAC (ebur128).
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import OUT_AUDIO, PROJ, TODAY, Log, ebur128, rel, run  # noqa: E402


def probe(path):
    r = run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,duration,"
             "sample_rate,channels,bit_rate,pix_fmt:format=duration", "-of", "json", path])
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    log = Log("mux")
    vid = args.video
    if vid is None:
        c = [p for p in glob.glob(os.path.join(PROJ, "outputs", "film_v1_*.mp4")) if not p.endswith("_av.mp4")]
        if not c:
            sys.exit("no outputs/film_v1_*.mp4 found")
        vid = max(c, key=os.path.getmtime)
    aud = args.audio or os.path.join(OUT_AUDIO, "mix_v1_%s.wav" % TODAY)
    m = re.search(r"(\d{8})", os.path.basename(vid))
    out = args.out or os.path.join(PROJ, "outputs", "film_v1_%s_av.mp4" % (m.group(1) if m else TODAY))
    pv = probe(vid)
    pa = probe(aud)
    vs = [s for s in pv["streams"] if s["codec_type"] == "video"][0]
    dv, da = float(pv["format"]["duration"]), float(pa["format"]["duration"])
    fps = eval(vs["r_frame_rate"])
    log("video %s: %s %sx%s %s fps %s, %.3f s" % (rel(vid), vs["codec_name"], vs["width"], vs["height"], vs["r_frame_rate"], vs.get("pix_fmt"), dv))
    log("audio %s: %.3f s" % (rel(aud), da))
    if abs(dv - da) > 1.0 / fps:
        log("WARNING: duration mismatch %.3f s (video %.3f, audio %.3f); output follows the video" % (da - dv, dv, da))
    run(["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", vid, "-i", aud, "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-t", "%.3f" % dv,
         "-movflags", "+faststart", out], log)
    po = probe(out)
    for s in po["streams"]:
        log("out stream: %s" % {k: s.get(k) for k in ("codec_type", "codec_name", "width", "height", "r_frame_rate", "sample_rate", "channels", "bit_rate", "duration")})
    lo = ebur128(out)
    log("out %s: %.3f s, audio I %.1f LUFS, TP %.1f dBTP" % (rel(out), float(po["format"]["duration"]), lo["I"], lo["TP"]))
    with open(out, "rb") as f:
        head = f.read(64 * 1024)
    log("faststart (moov before mdat): %s" % (head.find(b"moov") != -1 and (head.find(b"mdat") == -1 or head.find(b"moov") < head.find(b"mdat"))))
    log("log", rel(log.path))


if __name__ == "__main__":
    main()
