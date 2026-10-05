"""Render all v1.0 scene blends, encode each to mp4, delete the PNG frames, then concatenate the film.

Usage (plain python3): python3 scripts/film_v1/render_all.py <scratch_dir> [res_pct] [scenes e.g. 1,2,3,4,5,6,7] [version]
Writes outputs/v1/sNN.mp4 per scene and outputs/film_v1_0_<date>.mp4 (full film) if all seven scenes are present.
Logs go to <scratch_dir>/render_<timestamp>.log. Frames are deleted only after a successful encode.
"""
import datetime
import glob
import os
import shutil
import subprocess
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bslot.sh")  # Blender behind the 3-process cap
SCENES = {1: "s01_copper", 2: "s02_retimers", 3: "s03_npo", 4: "s04_cpo", 5: "s05_wafer_test", 6: "s06_fiber", 7: "s07_disclaimer"}

scratch = sys.argv[1]
pct = sys.argv[2] if len(sys.argv) > 2 else "100"
which = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(SCENES)
ver = sys.argv[4] if len(sys.argv) > 4 else "v1_0"
stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log = open(os.path.join(scratch, "render_%s.log" % stamp), "a")


def say(msg):
    line = "%s %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


os.makedirs(os.path.join(PROJ, "outputs", "v1"), exist_ok=True)
for n in which:
    name = SCENES[n]
    blend = os.path.join(PROJ, "scenes", "v1", name + ".blend")
    frames = os.path.join(scratch, "frames_v1", name + "_" + ver)
    os.makedirs(frames, exist_ok=True)
    out_mp4 = os.path.join(PROJ, "outputs", "v1", "%s_%s_p%s.mp4" % (name, ver, pct))
    t0 = time.time()
    say("render start %s (res %s%%)" % (name, pct))
    env = dict(os.environ)
    if os.environ.get("RENDER_PRESET"):
        env["RENDER_PRESET"] = os.environ["RENDER_PRESET"]
    r = subprocess.run([BLENDER, "-b", blend, "--python", os.path.join(PROJ, "scripts", "film_v1", "render_scene.py"), "--", frames, pct, "cartoon"],
                       stdout=log, stderr=subprocess.STDOUT, env=env)
    nfr = len(glob.glob(os.path.join(frames, "f_*.png")))
    say("render done %s rc=%s frames=%d in %.0f s" % (name, r.returncode, nfr, time.time() - t0))
    if r.returncode != 0 or nfr == 0:
        say("SKIP encode for %s" % name)
        continue
    e = subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "30", "-i", os.path.join(frames, "f_%04d.png"), "-c:v", "libx264",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", "-r", "30", out_mp4], stdout=log, stderr=subprocess.STDOUT)
    say("encode %s rc=%s -> %s" % (name, e.returncode, out_mp4))
    if e.returncode == 0 and "frames_v1" in frames and os.path.isdir(frames):
        shutil.rmtree(frames)
        say("frames deleted for %s" % name)
if all(os.path.exists(os.path.join(PROJ, "outputs", "v1", "%s_%s_p%s.mp4" % (SCENES[k], ver, pct))) for k in SCENES):
    lst = os.path.join(scratch, "concat_%s.txt" % stamp)
    with open(lst, "w") as f:
        for k in sorted(SCENES):
            f.write("file '%s'\n" % os.path.join(PROJ, "outputs", "v1", "%s_%s_p%s.mp4" % (SCENES[k], ver, pct)))
    final = os.path.join(PROJ, "outputs", "film_%s_p%s_%s.mp4" % (ver, pct, datetime.date.today().isoformat().replace("-", "")))
    c = subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart", final],
                       stdout=log, stderr=subprocess.STDOUT)
    say("concat rc=%s -> %s" % (c.returncode, final))
say("ALL DONE")
