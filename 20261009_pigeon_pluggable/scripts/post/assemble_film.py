"""Silent picture from frames: intro pass cross-dissolved into the main pass, labelled frames preferred over clean ones.

Run (project root):
  uv run --no-project python scripts/post/assemble_film.py --out outputs/film_v4_20261010.mp4 \
      [--intro outputs/film_v4_intro_frames] [--main outputs/film_v4_frames] [--lab outputs/film_v4_frames_lab] \
      [--start 136] [--end 1920] [--offset 4.5] [--fade 0.8] [--dur 64.0]

Intro fNNNN.png from frame 1; main pass fNNNN.png from --start (film time (start-1)/30 = --offset); for each main frame
the labelled copy in --lab is used when it exists (callouts / headers / beam note), else the clean frame. A temporary
symlink list feeds ffmpeg and is removed afterwards. xfade (fade) at --offset for --fade s, as in v3 (DevLog-011).
Output: H.264 yuv420p, CRF 18, preset medium, 30 fps, --dur s, +faststart; no audio (burn_captions.py adds it).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--intro", default="outputs/film_v4_intro_frames")
    ap.add_argument("--main", default="outputs/film_v4_frames")
    ap.add_argument("--lab", default="outputs/film_v4_frames_lab")
    ap.add_argument("--start", type=int, default=136)
    ap.add_argument("--end", type=int, default=1920)
    ap.add_argument("--offset", type=float, default=4.5)
    ap.add_argument("--fade", type=float, default=0.8)
    ap.add_argument("--dur", type=float, default=64.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="mainseq_")
    n_lab = 0
    for f in range(args.start, args.end + 1):
        nm = "f%04d.png" % f
        lab = os.path.join(args.lab, nm)
        src = lab if os.path.exists(lab) else os.path.join(args.main, nm)
        if not os.path.exists(src):
            shutil.rmtree(tmp)
            sys.exit("missing frame " + src)
        n_lab += src == lab
        os.symlink(os.path.abspath(src), os.path.join(tmp, nm))
    fc = ("[0:v]format=yuv444p,settb=1/%d[i];[1:v]format=yuv444p,settb=1/%d[m];"
          "[i][m]xfade=transition=fade:duration=%g:offset=%g,format=yuv420p[v]" % (args.fps, args.fps, args.fade, args.offset))
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-y", "-framerate", str(args.fps), "-start_number", "1",
           "-i", os.path.join(args.intro, "f%04d.png"), "-framerate", str(args.fps), "-start_number", str(args.start),
           "-i", os.path.join(tmp, "f%04d.png"), "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "18",
           "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(args.fps), "-t", "%.3f" % args.dur, "-movflags", "+faststart", args.out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    shutil.rmtree(tmp)
    if r.returncode:
        sys.exit(r.stderr[-3000:])
    print("wrote %s: main frames %d-%d (%d labelled), intro xfade %.1f s at %.1f s" % (args.out, args.start, args.end, n_lab, args.fade, args.offset))


if __name__ == "__main__":
    main()
