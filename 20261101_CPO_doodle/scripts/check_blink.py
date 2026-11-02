"""Frame-by-frame FAIL runs of the Known Good Die screen, per line, over one video loop."""
from __future__ import annotations

import functools
import http.server
import threading

from playwright.sync_api import sync_playwright

from render_stills import ARGS, ROOT, Quiet, find_chromium

FPS, SPEED = 30, 2


def main() -> int:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=find_chromium(), args=ARGS, headless=True)
        page = b.new_page(viewport={"width": 1280, "height": 844})
        page.goto(f"http://127.0.0.1:{srv.server_address[1]}/scenes/line_loop.html?w=1280&h=844", wait_until="load")
        page.wait_for_function("window.DD_READY === true", timeout=120_000)
        T = page.evaluate("LOOP.T")
        n = round(T * FPS / SPEED)
        rows = page.evaluate("""([n, step]) => { const out = []; for (let i = 0; i < n; i++) { DD.frame(i * step); out.push(LOOP.LINES.map((L) => !!L.gateBad)); } return out; }""", [n, SPEED / FPS])
        b.close()
    srv.shutdown()
    for li in range(len(rows[0])):
        runs, cur = [], 0
        for r in rows:
            if r[li]:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        if cur:
            runs.append(cur)
        print(f"line {li}: {len(runs)} FAIL blinks in {n} frames, shortest {min(runs) if runs else 0}, longest {max(runs) if runs else 0}, on {sum(runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
