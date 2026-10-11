"""Music bed from an existing track (v4: Plague Inc OST "Plague Blossom (Main Theme, Evolved)"; DevLog-020 W6).

NOT CLEARED FOR PUBLIC POSTING: third-party music, personal / preview use only (owner decision W6). The source file
stays outside the repo (read-only); only the derived wav is written, to git-ignored outputs/audio/.

Run (project root):
  uv run --no-project python scripts/audio/make_music_bed.py --src "<path to the mp3>" [--start 0.0] [--fade-start 60.8]
Writes outputs/audio/music_plague_<date>.wav (48 kHz stereo PCM 24, film length from narration.json total_s).

Choice of start (DevLog-014 section 14): the track is 390 s at about 161.5 BPM; its opening (0-50 s) is a sparse pulse
with hits about every 3 s and builds into the full theme at about 50-52 s. Starting at 0.0 s (first beat 0.116 s) puts
the intro under the film intro and lands the build near the S6 street finale (51.0 s), with no loop or edit needed.
Processing: high-pass 40 Hz, voice pocket (-5 dB at 1.6 kHz and -4 dB at 3.2 kHz, about 1 octave wide), fade-in
0.3 s, fade-out from --fade-start to the end (under the S7 disclaimer). Level and ducking are set in mix.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_util import OUT_AUDIO, SR, TODAY, TOTAL, Log, ebur128, rel, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--fade-start", type=float, default=TOTAL - 2.2)
    ap.add_argument("--out", default=os.path.join(OUT_AUDIO, "music_plague_%s.wav" % TODAY))
    args = ap.parse_args()
    log = Log("music_bed")
    af = ("aresample=%d,aformat=channel_layouts=stereo,highpass=f=40,"
          "equalizer=f=1600:t=o:w=1.0:g=-5,equalizer=f=3200:t=o:w=1.0:g=-4,"
          "afade=t=in:st=0:d=0.3,afade=t=out:st=%.3f:d=%.3f" % (SR, args.fade_start, TOTAL - args.fade_start))
    run(["ffmpeg", "-hide_banner", "-nostats", "-y", "-ss", "%.3f" % args.start, "-t", "%.3f" % TOTAL, "-i", args.src,
         "-af", af, "-c:a", "pcm_s24le", args.out], log)
    m = ebur128(args.out)
    log("music bed: source start %.3f s, %.1f s, fade-out %.2f-%.2f s; I %.1f LUFS, TP %.1f dBTP -> %s (NOT CLEARED FOR PUBLIC POSTING)"
        % (args.start, TOTAL, args.fade_start, TOTAL, m["I"], m["TP"], rel(args.out)))
    log("log", rel(log.path))


if __name__ == "__main__":
    main()
