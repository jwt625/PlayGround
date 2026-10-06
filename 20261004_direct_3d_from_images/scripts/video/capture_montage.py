"""Capture the build montage page (outputs/video/viewer/montage.html) frame by frame (150 frames at 5 images/s)
with headless Chrome and encode a 30 fps H.264 mp4 (each image held 6 frames).
Usage: uv run --with playwright python scripts/video/capture_montage.py --port PORT [--frames 0,40,90]"""

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUTPUTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--frames", default=None)
a = ap.parse_args()
out = OUTPUTS / "video" / f"crt_build_montage_{date.today():%Y%m%d}.mp4"
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True, args=["--use-angle=metal", "--ignore-gpu-blocklist"])
    page = b.new_page(viewport={"width": 1080, "height": 1620})
    page.on("console", lambda m: print("console:", m.text) if m.type == "error" else None)
    page.goto(f"http://127.0.0.1:{a.port}/montage.html")
    page.wait_for_function("window.ready === true", timeout=300000)
    canvas = page.locator("#comp")
    if a.frames:
        for f in [int(x) for x in a.frames.split(",")]:
            page.evaluate(f"window.setFrame({f})")
            canvas.screenshot(path=str(OUTPUTS / "video" / "viewer_test" / f"m_{f:03d}.png"))
        sys.exit(0)
    n = page.evaluate("window.nFrames")
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", "5", "-c:v", "png",
                             "-i", "-", "-vf", "format=yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                             "-r", "30", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    for f in range(n):
        page.evaluate(f"window.setFrame({f})")
        proc.stdin.write(canvas.screenshot(type="png"))
    proc.stdin.close()
    proc.wait()
    b.close()
print("VIDEO", out)
