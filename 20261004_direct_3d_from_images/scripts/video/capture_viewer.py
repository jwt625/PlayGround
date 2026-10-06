"""Capture the two-viewer page (glb on top, 3DGS ply below) frame by frame with headless Chrome and pipe it to ffmpeg.

Usage: uv run --with playwright python scripts/video/capture_viewer.py --port PORT [--frames 0,450] [--out file.mp4]
Needs: a static server on PORT serving outputs/video/viewer (index.html from scripts/video/viewer/, cfg.json,
model.glb, scene.ply symlinks). Uses the installed Google Chrome (channel "chrome") with a fresh temporary profile.
With --frames it writes PNG stills to outputs/video/viewer_test/ instead of a video.
"""

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--amb", default="3.0")
    ap.add_argument("--out", default=str(OUTPUTS / "video" / f"crt_model_vs_3dgs_{date.today():%Y%m%d}.mp4"))
    a = ap.parse_args()
    cfg = json.loads((OUTPUTS / "video" / "viewer" / "cfg.json").read_text())
    S, fps = cfg["size"], 30
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True,
                              args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        page = b.new_page(viewport={"width": cfg.get("w", S), "height": 2 * cfg.get("h", S)}, device_scale_factor=1)
        page.on("console", lambda m: print("console:", m.text) if m.type in ("error", "warning") else None)
        page.goto(f"http://127.0.0.1:{a.port}/index.html?amb={a.amb}")
        page.wait_for_function("window.ready === true", timeout=300000)
        if a.frames:
            out = OUTPUTS / "video" / "viewer_test"
            out.mkdir(parents=True, exist_ok=True)
            for f in [int(x) for x in a.frames.split(",")]:
                page.evaluate(f"window.setFrame({f})")
                page.screenshot(path=str(out / f"v_{f:05d}.png"))
            print("STILLS", out)
            return
        # real-time in-page recording (MediaRecorder on a composite canvas), then transcode to H.264
        webm = Path(a.out).with_suffix(".webm")
        with page.expect_download(timeout=600000) as dl:
            info = page.evaluate(f"window.record({fps})")
        dl.value.save_as(str(webm))
        print("recorded", info, flush=True)
        b.close()
        # MediaRecorder output is full range: convert to tv-range yuv420p for players
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-vf",
                        "scale=in_range=pc:out_range=tv,format=yuv420p", "-c:v", "libx264", "-preset", "medium",
                        "-crf", "18", "-r", str(fps), "-color_range", "tv", "-movflags", "+faststart", a.out],
                       check=True)
        webm.unlink()
    print("VIDEO", a.out)


if __name__ == "__main__":
    main()
