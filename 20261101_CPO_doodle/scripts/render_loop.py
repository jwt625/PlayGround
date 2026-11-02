"""Capture the looping line scene to numbered PNGs and encode an MP4 (or a GIF, by --out extension), same capture approach as render_stills.py.

Usage: uv run --no-project --with playwright python scripts/render_loop.py [--w 1280] [--h 720] [--fps 30] [--still T ...]
Frames go to cache/line_loop/ (not kept); the video goes to outputs/cpo_line_loop.mp4. A local static server binds
127.0.0.1 on an OS-assigned port.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import functools
import http.server
import shutil
import subprocess
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from render_stills import ARGS, ROOT, Quiet, find_chromium


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=844)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--colors", type=int, default=64)
    ap.add_argument("--still", type=float, nargs="*", help="render these times to PNG only, no GIF")
    ap.add_argument("--out", default="outputs/cpo_line_loop.mp4")
    ap.add_argument("--query", default="", help="extra scene URL params, e.g. pace=B&bars=8")
    ap.add_argument("--tag", default="line_loop", help="frame cache subfolder")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--shard", default="0/1", help="i/n: render every n-th frame only, into the shared cache; no GIF unless n = 1")
    ap.add_argument("--speed", type=int, default=2, help="integer speed-up: keep every n-th frame, same frame rate")
    a = ap.parse_args()
    cache = ROOT / "cache" / a.tag
    si, sn = (int(v) for v in a.shard.split("/"))
    if a.still is None and sn == 1 and cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    log = (logs / f"render_loop_{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}.log").open("w")
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=find_chromium(), args=ARGS, headless=True)
        page = b.new_page(viewport={"width": a.w, "height": a.h})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{srv.server_address[1]}/scenes/line_loop.html?w={a.w}&h={a.h}&{a.query}", wait_until="load")
        page.wait_for_function("window.DD_READY === true", timeout=120_000)
        period = page.evaluate("LOOP.T")
        times = a.still if a.still is not None else [a.speed * i / a.fps for i in range(round(period * a.fps / a.speed))]
        t0 = time.perf_counter()
        for i, t in enumerate(times):
            if i % sn != si:
                continue
            page.evaluate("t => DD.frame(t)", t)
            name = f"still_{t:07.3f}.png" if a.still is not None else f"f{i:05d}.png"
            (cache / name).write_bytes(base64.b64decode(page.evaluate("document.querySelector('canvas').toDataURL('image/png').split(',')[1]")))
            if errors:
                print("ERRORS", errors[:3]); return 1
            if i % 60 == 0:
                msg = f"{i}/{len(times)} {time.perf_counter() - t0:.1f}s"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if a.still is None:
            same = page.evaluate("T => { const c = document.querySelector('canvas'); DD.frame(0); const u = c.toDataURL(); DD.frame(T); return u === c.toDataURL(); }", period)
            print("loop closes:", same, "period", period, flush=True)
        b.close()
    srv.shutdown()
    if a.still is not None or sn != 1:
        return 0
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gif":
        vf = (f"split[s0][s1];[s0]palettegen=max_colors={a.colors}:stats_mode=full[p];"
              "[s1][p]paletteuse=dither=none:diff_mode=rectangle")
        cmd = ["-vf", vf, "-loop", "0"]
    else:
        cmd = ["-c:v", "libx264", "-preset", "slow", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-r", str(a.fps), "-movflags", "+faststart"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(a.fps), "-i", str(cache / "f%05d.png"), *cmd, str(out)], check=True)
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
