"""List solids whose AABBs intersect, over a sample of frames. Zero pairs means painter order is exact.

Usage: uv run --no-project --with playwright python scripts/check_layers.py [--step 0.25]
"""
from __future__ import annotations

import argparse
import collections
import functools
import http.server
import re
import threading

from playwright.sync_api import sync_playwright

from render_stills import ARGS, ROOT, Quiet, find_chromium


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.25)
    a = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=find_chromium(), args=ARGS, headless=True)
        page = b.new_page(viewport={"width": 1280, "height": 720})
        page.goto(f"http://127.0.0.1:{srv.server_address[1]}/scenes/line_loop.html?w=1280&h=720", wait_until="load")
        page.wait_for_function("window.DD_READY === true", timeout=120_000)
        T = page.evaluate("LOOP.T")
        kinds: collections.Counter[str] = collections.Counter()
        sample: dict[str, str] = {}
        n = 0
        t = 0.0
        while t < T:
            for pair in page.evaluate("t => DD.overlaps(t)", t):
                k = re.sub(r"\[[^\]]*\]", "", pair)
                kinds[k] += 1
                sample.setdefault(k, f"t={t:.2f} {pair}")
            t += a.step
            n += 1
        b.close()
    srv.shutdown()
    print(f"{n} frames, {sum(kinds.values())} overlapping pairs, {len(kinds)} kinds")
    for k, c in kinds.most_common(40):
        print(f"{c:6d} {k:32s} {sample[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
