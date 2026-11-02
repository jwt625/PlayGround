"""Time one line_loop frame, split by stage, for several Chromium flag sets. Runs one browser at a time.

Usage: uv run --no-project --with playwright python scripts/bench_render.py
"""
from __future__ import annotations

import functools
import http.server
import threading
import time

from playwright.sync_api import sync_playwright

from render_stills import ROOT, Quiet, find_chromium

W, H = 1280, 720
CONFIGS = {
    "swiftshader (current)": (["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist", "--font-render-hinting=none"], ""),
    "no gpu flags": (["--font-render-hinting=none"], ""),
    "cpu canvas": (["--font-render-hinting=none", "--disable-gpu", "--disable-accelerated-2d-canvas"], ""),
    "cpu canvas, 1 raster thread": (["--font-render-hinting=none", "--disable-gpu", "--disable-accelerated-2d-canvas", "--num-raster-threads=1"], ""),
}
TIMES = [3.0, 11.0, 19.0, 27.0]

STAGES = """async (t) => {
  const canvas = document.querySelector('canvas');
  const t0 = performance.now();
  DD.frame(t);
  const t1 = performance.now();
  const d = canvas.getContext('2d').getImageData(0, 0, 4, 4).data[0];  // forces the raster flush
  const t2 = performance.now();
  const blob = await new Promise((r) => canvas.toBlob(r, 'image/png'));
  const t3 = performance.now();
  return { js: t1 - t0, flush: t2 - t1, png: t3 - t2, bytes: blob.size, solids: DD.liveCount ? DD.liveCount() : -1, d };
}"""


def main() -> int:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/scenes/line_loop.html?w={W}&h={H}"
    with sync_playwright() as pw:
        for name, (args, _) in CONFIGS.items():
            b = pw.chromium.launch(executable_path=find_chromium(), args=args, headless=True)
            page = b.new_page(viewport={"width": W, "height": H})
            page.goto(url, wait_until="load")
            page.wait_for_function("window.DD_READY === true", timeout=120_000)
            rows = [page.evaluate(STAGES, t) for t in TIMES]
            s0 = time.perf_counter()
            page.screenshot(path=str(ROOT / "cache" / "bench.png"), clip={"x": 0, "y": 0, "width": W, "height": H})
            shot = (time.perf_counter() - s0) * 1000
            n = len(rows)
            print(f"{name:32s} js {sum(r['js'] for r in rows) / n:7.0f} ms  flush {sum(r['flush'] for r in rows) / n:7.0f} ms  "
                  f"png {sum(r['png'] for r in rows) / n:6.0f} ms  screenshot {shot:6.0f} ms  solids {rows[0]['solids']}", flush=True)
            b.close()
    srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
