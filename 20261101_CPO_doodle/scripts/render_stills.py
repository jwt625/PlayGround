"""Capture scene stills from headless Chromium, same approach as the hybrid bonding short renderer.

Usage: uv run --no-project --with playwright python scripts/render_stills.py oe:A oe:B line:A ... [--t 0.0]
Writes outputs/style_drafts/<scene>_<style>.png. A local static server binds 127.0.0.1 on an OS-assigned port.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import re
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "style_drafts"
PW_CACHE = Path.home() / "Library" / "Caches" / "ms-playwright"
# No swiftshader flags: Canvas2D does not need them, and they cost 3000 ms versus 115 ms per frame (scripts/bench_render.py).
ARGS = ["--font-render-hinting=none"]
W, H = 1920, 1080


def find_chromium() -> str | None:
    def rev(p: Path) -> int:
        m = re.search(r"-(\d+)$", p.name)
        return int(m.group(1)) if m else -1

    for d in sorted(PW_CACHE.glob("chromium_headless_shell-*"), key=rev, reverse=True):
        exe = sorted(d.glob("chrome-headless-shell-*/chrome-headless-shell"))
        if exe:
            return str(exe[0])
    return None


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="+", help="scene:style")
    ap.add_argument("--t", type=float, default=0.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=find_chromium(), args=ARGS, headless=True)
        page = b.new_page(viewport={"width": W, "height": H})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        for job in a.jobs:
            scene, style = job.split(":")
            errors.clear()
            page.goto(f"{base}/scenes/{scene}.html?style={style}", wait_until="load")
            page.wait_for_function("window.DD_READY === true", timeout=60_000)
            ms = page.evaluate("t => { const t0 = performance.now(); DD.frame(t); return performance.now() - t0; }", a.t)
            dest = OUT / f"{scene}_{style}.png"
            page.screenshot(path=str(dest), clip={"x": 0, "y": 0, "width": W, "height": H}, animations="disabled")
            print(f"{dest.name} {ms:.0f} ms {'ERRORS ' + str(errors) if errors else ''}", flush=True)
        b.close()
    srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
