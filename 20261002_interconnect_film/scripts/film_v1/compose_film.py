#!/usr/bin/env python3
"""Assemble the film from the scene clips and the six oscilloscope transitions, optionally mux the audio mix.

Pipeline (DevLog/v1/DevLog-005-transitions.md):
  1. tex     scene frames (last/first 0.5 s around each boundary) -> RGBA oscilloscope screen textures   (numpy, via uv)
  2. render  Blender (scenes/v1/transitions.blend) renders 30 frames per boundary at the scene resolution
  3. encode  each boundary clip -> H.264 (CRF 10) in --clips-dir
  4. concat  one ffmpeg filter graph: S1[0:285] T1 S2[15:285] T2 ... S6[15:285] T6 S7[15:] = exactly 1860 frames (62.0 s, 30 fps);
             the transition clip REPLACES film frames b*30-15 .. b*30+14; scenes are not time-shifted (VO/SFX cue times stay valid)
  5. mux     optional audio (48 kHz AAC)

Examples (repo root):
  draft p50:  python3 scripts/film_v1/compose_film.py --scene-glob 'outputs/v1/s{n:02d}_*_v1_1_draft_p50.mp4' \
                --s7-glob 'outputs/v1/s07_*_v1_0_draft_p50.mp4' --out outputs/film_v1_2_transitions_test_draft_p50_20261002.mp4
  final p100: python3 scripts/film_v1/compose_film.py --scene-glob 'outputs/v1/s{n:02d}_*_v1_2_p100.mp4' \
                --s7-glob 'outputs/v1/s07_*_p100.mp4' --out outputs/film_v1_2_p100_20261002.mp4 --audio outputs/audio/mix_vo_sfx_20261002.wav
  re-concat only (clips already encoded): add --reuse-clips
{n:02d} in a glob is replaced by the scene number; if several files match, the last in sorted order is used.
Only the standard library is needed here; the numpy texture step is run through `uv run --no-project --with numpy`.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(HERE))
TRANS = os.path.join(HERE, "transitions.py")
BLEND = os.path.join(PROJ, "scenes", "v1", "transitions.blend")
BLENDER = os.environ.get("BLENDER", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bslot.sh"))  # 3-process cap
NF, HALF = 30, 15
SCENE_FRAMES = [300] * 6 + [60]


def run(cmd, **kw):
    txt = " ".join(str(c) for c in cmd)
    print("+", txt if len(txt) < 400 else txt[:400] + " ...", flush=True)
    subprocess.run(cmd, check=True, **kw)


def pick(pattern, n):
    pat = pattern.format(n=n)
    pat = pat if os.path.isabs(pat) else os.path.join(PROJ, pat)
    fs = sorted(glob.glob(pat))
    if not fs:
        sys.exit("no scene file for S%d: %s" % (n, pat))
    return fs[-1]


def probe(path):
    o = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames",
                                 "-of", "csv=p=0", path]).decode().strip().split(",")
    return int(o[0]), int(o[1]), int(o[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-glob", required=True, help="pattern with {n:02d} for scenes 1..7 (S7 too unless --s7-glob)")
    ap.add_argument("--s7-glob", help="separate pattern for the disclaimer card (S7)")
    ap.add_argument("--out", required=True, help="output mp4")
    ap.add_argument("--audio", help="audio file (wav/m4a) muxed as AAC 48 kHz")
    ap.add_argument("--clips-dir", help="where the encoded transition clips go (default outputs/v1/transitions/<out stem>)")
    ap.add_argument("--work", help="scratch dir for textures and PNG frames (deleted afterwards)")
    ap.add_argument("--reuse-clips", action="store_true", help="skip tex/render/encode, use existing clips in --clips-dir")
    ap.add_argument("--only", help="render only these boundaries (comma list); others must already exist in --clips-dir")
    ap.add_argument("--xfade", type=int, default=0, help="cross-fade N frames between scene and clip at both clip ends (default 0)")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--crf", default="14", help="CRF of the final film (transition clips use 10)")
    a = ap.parse_args()

    out = a.out if os.path.isabs(a.out) else os.path.join(PROJ, a.out)
    stem = os.path.splitext(os.path.basename(out))[0]
    clips = a.clips_dir or os.path.join(PROJ, "outputs", "v1", "transitions", stem)
    os.makedirs(clips, exist_ok=True)
    work = a.work or os.path.join(os.environ.get("TMPDIR", "/tmp"), "transitions_" + stem)
    scenes = [pick(a.s7_glob if (n == 7 and a.s7_glob) else a.scene_glob, n) for n in range(1, 8)]
    dims = [probe(p) for p in scenes]
    W, H, _ = dims[0]
    for n, (w, h, nb) in enumerate(dims, 1):
        if (w, h) != (W, H):
            sys.exit("scene %d has size %dx%d, expected %dx%d" % (n, w, h, W, H))
        if nb != SCENE_FRAMES[n - 1]:
            sys.exit("scene %d has %d frames, expected %d (30 fps; S1..S6 300, S7 60)" % (n, nb, SCENE_FRAMES[n - 1]))
    ks = [int(x) for x in a.only.split(",")] if a.only else list(range(1, 7))
    clip = lambda k: os.path.join(clips, "t%d.mp4" % k)

    if not a.reuse_clips:
        tex, png = os.path.join(work, "tex"), os.path.join(work, "png")
        for d in (tex, png):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
        t0 = time.time()
        run(["uv", "run", "--no-project", "--with", "numpy", "python", TRANS, "tex", "--out", tex, "--scene-glob", a.scene_glob,
             "--only", ",".join(map(str, ks))] + (["--s7-glob", a.s7_glob] if a.s7_glob else []), cwd=PROJ)
        t1 = time.time()
        run([BLENDER, "-b", BLEND, "--python", TRANS, "--", "render", tex, png, str(W), str(H), ",".join(map(str, ks))], stdout=subprocess.DEVNULL)
        t2 = time.time()
        for k in ks:
            run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", os.path.join(png, "t%d_%%04d.png" % k), "-c:v", "libx264",
                 "-crf", "10", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "30", clip(k)])
        print("timing: textures %.1f s, Blender render %.1f s (%d frames), total %.1f s" % (t1 - t0, t2 - t1, NF * len(ks), time.time() - t0))
        if not a.keep_work:
            shutil.rmtree(work, ignore_errors=True)
    for k in range(1, 7):
        if not os.path.exists(clip(k)):
            sys.exit("missing clip %s" % clip(k))

    # ---- filter graph
    inputs = []                          # every segment opens its own input (a filter-graph input label can be used once)
    fg, lab, cnt = [], [], [0]

    def seg(path, s, e):
        inputs.append(path)
        inp = len(inputs) - 1
        cnt[0] += 1
        name = "v%d" % cnt[0]
        fg.append("[%d:v]trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS,setsar=1,format=yuv420p[%s]" % (inp, s, e, name))
        return name

    def blend(x, y, n):
        """x: n frames, y: n frames; weight of y rises (i+1)/(n+1)."""
        cnt[0] += 1
        name = "v%d" % cnt[0]
        fg.append("[%s][%s]blend=all_expr='A*(1-(N+1)/%d)+B*((N+1)/%d)':shortest=1[%s]" % (x, y, n + 1, n + 1, name))
        return name

    xf = a.xfade
    for n in range(1, 8):
        s0 = HALF if n > 1 else 0
        s1 = SCENE_FRAMES[n - 1] - HALF if n < 7 else SCENE_FRAMES[n - 1]
        lab.append(seg(scenes[n - 1], s0, s1))
        if n < 7:
            k = n
            if xf <= 0:
                lab.append(seg(clip(k), 0, NF))
            else:
                # scene N tail frames (same film time as the first xf clip frames) fade into the clip, and the clip fades into scene N+1
                tail = seg(scenes[n - 1], SCENE_FRAMES[n - 1] - HALF, SCENE_FRAMES[n - 1] - HALF + xf)
                lab.append(blend(tail, seg(clip(k), 0, xf), xf))
                lab.append(seg(clip(k), xf, NF - xf))
                head = seg(scenes[n], HALF - xf, HALF)
                lab.append(blend(seg(clip(k), NF - xf, NF), head, xf))
    fg.append("%sconcat=n=%d:v=1:a=0[outv]" % ("".join("[%s]" % l for l in lab), len(lab)))
    cmd = ["ffmpeg", "-v", "error", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    if a.audio:
        cmd += ["-i", a.audio]
    cmd += ["-filter_complex", ";".join(fg), "-map", "[outv]"]
    if a.audio:
        cmd += ["-map", "%d:a" % len(inputs), "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += ["-c:v", "libx264", "-crf", a.crf, "-preset", "medium", "-pix_fmt", "yuv420p", "-r", "30", "-t", "62", "-movflags", "+faststart", out]
    run(cmd)
    w, h, nb = probe(out)
    print("OUT", out, "%dx%d" % (w, h), "frames", nb, "(expected 1860)")


if __name__ == "__main__":
    main()
