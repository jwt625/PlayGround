"""Count solids drawn in the wrong order, over a sample of frames. Usage: check_order.py [--step 0.5]"""
from __future__ import annotations

import argparse
import functools
import http.server
import threading

from playwright.sync_api import sync_playwright

from render_stills import ARGS, ROOT, Quiet, find_chromium


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--query", default="")
    a = ap.parse_args()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=find_chromium(), args=ARGS, headless=True)
        page = b.new_page(viewport={"width": 1280, "height": 844})
        page.goto(f"http://127.0.0.1:{srv.server_address[1]}/scenes/line_loop.html?w=1280&h=844&{a.query}", wait_until="load")
        page.wait_for_function("window.DD_READY === true", timeout=120_000)
        T = page.evaluate("LOOP.T")
        total, frames, worst, t, samples = 0, 0, 0, 0.0, []
        while t < T:
            r = page.evaluate("t => DD.violations(t)", t)
            total += r["n"]; frames += 1; worst = max(worst, r["n"])
            if r["out"] and len(samples) < 12:
                samples += [f"t={t:.2f} {x}" for x in r["out"][:3]]
            t += a.step
        print(f"{frames} frames, {total} wrong-order pairs, worst frame {worst}, solids {r['solids']}, overlapping pairs {r['pairs']}")
        print("\n".join(samples))
        b.close()
    srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
