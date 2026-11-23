"""Headless dev walk: open player.html?c=<course>&dev=1 (file://), walk one module with the real speech path (clips play
muted at 4x), click every tab / option / hotspot / card on each step, and read the dev panel: "Recorded clips: X of Y"
and the "Spoken without a recorded clip" list (must be empty).

Run (from course/): uv run --no-project --with playwright python tools/tts/dev_walk.py --course ferroelectric
Uses the installed Google Chrome (channel="chrome"); no server, no network.
"""
import argparse
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COURSE = os.path.abspath(os.path.join(HERE, "..", ".."))

INIT = """
(() => {
  window.__aud = { playing: 0, started: 0, errors: 0, last: Date.now() };
  const orig = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function () {
    this.muted = true; this.playbackRate = 4; const A = window.__aud;
    if (!this.__hooked) {
      this.__hooked = true;
      const end = () => { A.playing = Math.max(0, A.playing - 1); A.last = Date.now(); };
      this.addEventListener('ended', end); this.addEventListener('pause', end);
      this.addEventListener('error', () => { A.errors++; end(); });
    }
    A.playing++; A.started++; A.last = Date.now();
    return orig.call(this);
  };
})();
"""
CLICKABLE = ".stage-body .reveal-tabs button, .stage-body .kc-opts button, .stage-body button.hs-mk, .stage-body button.card, .stage-body button.flip"


def wait_idle(page, quiet=1.5, cap=120):
    t0 = time.time()
    last_cap, since = None, time.time()
    while time.time() - t0 < cap:
        st = page.evaluate("() => ({p: window.__aud.playing, cap: (document.querySelector('.cc-text') || {}).textContent || ''})")
        if st["p"] > 0 or st["cap"] != last_cap:
            since = time.time()
            last_cap = st["cap"]
        elif time.time() - since > quiet:
            return
        time.sleep(0.2)


def dismiss_modal(page):
    for _ in range(20):
        b = page.query_selector(".mf .btn")
        if not b or not b.is_visible():
            return
        if b.is_enabled():
            b.click()
            return
        time.sleep(0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", required=True)
    ap.add_argument("--max-steps", type=int, default=80)
    a = ap.parse_args()
    from playwright.sync_api import sync_playwright
    url = "file://%s/player.html?c=%s&dev=1" % (COURSE, a.course)
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
        page = br.new_page(viewport={"width": 1280, "height": 900})
        page.add_init_script(INIT)
        errs = []
        page.on("pageerror", lambda e: errs.append(str(e)))
        page.goto(url)
        page.wait_for_timeout(800)
        rec = page.evaluate("() => (([...document.querySelectorAll('.dp-row.dp-dim')].map(e => e.textContent)).join(' | '))")
        print("dev panel:", rec)
        page.click("text=Begin course")   # module 0 (Welcome) starts from the splash
        steps = 0
        while steps < a.max_steps:
            dismiss_modal(page)
            wait_idle(page)
            n = len(page.query_selector_all(CLICKABLE))
            for i in range(n):
                els = page.query_selector_all(CLICKABLE)
                if i < len(els) and els[i].is_visible() and els[i].is_enabled():
                    try:
                        els[i].click(timeout=2000)
                    except Exception:
                        pass
                    wait_idle(page)
                    dismiss_modal(page)
            steps += 1
            nb = page.query_selector("button.next")
            label = nb.text_content() if nb else ""
            if not nb or "module" in label.lower() or not nb.is_enabled():
                break
            nb.click()
        missing = page.evaluate("() => [...document.querySelectorAll('.dp-missing li')].map(e => e.textContent)")
        aud = page.evaluate("() => window.__aud")
        print("course %s module 0: steps walked %d, clips played %d, clip load errors %d, page errors %d" % (
            a.course, steps, aud["started"], aud["errors"], len(errs)))
        print("spoken without a recorded clip: %d" % len(missing))
        for m in missing:
            print("  MISSING", m)
        for e in errs[:5]:
            print("  PAGE ERROR", e)
        br.close()


if __name__ == "__main__":
    main()
