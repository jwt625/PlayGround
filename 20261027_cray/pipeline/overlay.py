"""Overlay check: render a trace with the shipping renderer on top of its scan crop.

Usage: python3 -I pipeline/overlay.py <pn> <fig-id> [--region x0,y0,x1,y1] [--scale S]
Writes work/<pn>/check/fig-<id>[-region].overlay.png (scan grey, trace red) and
work/<pn>/check/fig-<id>[-region].trace.png (trace alone, black on white).
Read both: grey not covered by red = missing geometry; red off grey = misplaced geometry.
--region crops to a sub-rectangle (crop px) for close inspection; --scale (default: fit ~1600 px).
"""
import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;background:#fff}}
svg{{display:block}}
.tr-s{{stroke:{ink};fill:none;stroke-linecap:round;stroke-linejoin:round}}
.tr-fill-solid{{fill:{ink}}} .tr-fill-light{{fill:{ink};fill-opacity:.12}}
.tr-arrow{{fill:{ink}}} .tr-hatchline{{stroke:{ink};stroke-width:2}}
.tr-t{{fill:{ink};font-family:Helvetica,Arial,sans-serif}} .tr-fill-paper{{fill:#fff}} .tr-ink-paper{{stroke:#fff}} .tr-t.tr-ink-paper{{fill:#fff}} .tr-it{{font-style:italic}} .tr-b{{font-weight:700}}
</style></head><body>
<svg id="s" width="{W}" height="{H}" viewBox="{vb}" preserveAspectRatio="none">{bg}<g id="t"></g></svg>
<script>{js}</script>
<script>
const tr = {trace};
const g = document.getElementById('t');
const tmp = document.createElementNS('http://www.w3.org/2000/svg','svg');
TraceRender.renderTrace(tmp, tr, {{uid:'k'}});
[...tmp.childNodes].forEach(n => g.appendChild(n));
</script></body></html>"""


def shot(html: str, w: int, h: int, out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
        src = f.name
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                    f"--window-size={w},{h}", f"--screenshot={out}", "--default-background-color=ffffffff",
                    "--virtual-time-budget=2000", f"file://{src}"], check=True, capture_output=True)
    Path(src).unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pn")
    ap.add_argument("fig")
    ap.add_argument("--region")
    ap.add_argument("--scale", type=float)
    a = ap.parse_args()
    trace = json.loads((ROOT / "data" / a.pn / "traces" / f"fig-{a.fig}.json").read_text())
    png = ROOT / "work" / a.pn / "figs" / f"fig-{a.fig}.png"
    W, H = trace["w"], trace["h"]
    x0, y0, x1, y1 = [int(v) for v in a.region.split(",")] if a.region else (0, 0, W, H)
    rw, rh = x1 - x0, y1 - y0
    k = a.scale or min(1.0, 1600 / max(rw, rh)) if not a.region else (a.scale or min(3.0, 1600 / max(rw, rh)))
    ow, oh = max(1, round(rw * k)), max(1, round(rh * k))
    js = (ROOT / "viewer" / "trace-render.js").read_text()
    img = base64.b64encode(png.read_bytes()).decode()
    bg = f'<image href="data:image/png;base64,{img}" x="0" y="0" width="{W}" height="{H}" opacity="0.38"/>'
    out = ROOT / "work" / a.pn / "check"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"fig-{a.fig}" + (f"-{x0}_{y0}_{x1}_{y1}" if a.region else "")
    common = dict(W=ow, H=oh, vb=f"{x0} {y0} {rw} {rh}", js=js, trace=json.dumps(trace))
    shot(PAGE.format(ink="#e00000", bg=bg, **common), ow, oh, out / f"{tag}.overlay.png")
    shot(PAGE.format(ink="#000000", bg="", **common), ow, oh, out / f"{tag}.trace.png")
    print(out / f"{tag}.overlay.png")
    print(out / f"{tag}.trace.png")


if __name__ == "__main__":
    main()
