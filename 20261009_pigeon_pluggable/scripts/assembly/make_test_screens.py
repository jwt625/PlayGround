"""Instrument screen textures for the S5 final-test stand (workstream M, DevLog-019 section 9). Plain styling: dark grey
background, light grey text, one accent (green) for PASS / OK; the live PAM4 eyes are separate quads in Blender (amber
atlas). All numbers are illustrative display values chosen inside the referenced ranges (research/assembly_refs.md
section 6): TDECQ 1.6 dB (typical 1-2 dB), pre-FEC BER values below the KP4 RS(544,514) limit 2.4E-4.
Writes <dir>/scr_tx.png, scr_rx_base.png, scr_rx_curve.png (RGBA, revealed left->right), scr_link_base.png,
scr_link_rows.png (RGBA, revealed top->bottom). Usage: uv run --with numpy --with pillow python scripts/assembly/make_test_screens.py <dir>"""
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 900, 600
BG = (16, 18, 20)
FG = (214, 218, 222)
DIM = (120, 126, 132)
GRID = (52, 57, 62)
OK = (90, 205, 120)
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
MONO = "/System/Library/Fonts/Supplemental/Andale Mono.ttf"


def f(size, mono=False):
    return ImageFont.truetype(MONO if mono else FONT, size)


def frame(d, title, right):
    d.rectangle((0, 0, W - 1, H - 1), outline=GRID, width=4)
    d.rectangle((0, 0, W - 1, 62), fill=(30, 33, 37))
    d.text((22, 12), title, font=f(34), fill=FG)
    tw = d.textlength(right, font=f(28))
    d.text((W - 22 - tw, 17), right, font=f(28), fill=DIM)


def tx(out):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    frame(d, "TX  OPTICAL EYE", "DCA  /  REF RX")
    d.rectangle((30, 78, W - 30, 470), outline=GRID, width=2)        # live eye quad sits inside (Blender)
    d.line((30, 490, W - 30, 490), fill=GRID, width=2)          # per-lane label + readout are text objects (Blender, v4.2)
    im.save(os.path.join(out, "scr_tx.png"))


def rx(out):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    frame(d, "RX  SENSITIVITY", "BERT ED  /  VOA SWEEP")
    x0, x1, y0, y1 = 130, 860, 90, 500
    d.rectangle((x0, y0, x1, y1), outline=GRID, width=2)
    decades = [3, 5, 7, 9, 11]
    ylog = lambda e: y0 + (e - 2) / (12 - 2) * (y1 - y0)       # log10(BER) = -e
    for e in decades:
        yy = ylog(e)
        d.line((x0, yy, x1, yy), fill=GRID, width=1)
        d.text((22, yy - 14), "1E-%02d" % e, font=f(24, True), fill=DIM)
    lim = ylog(-math.log10(2.4e-4))
    for xx in range(x0, x1, 24):
        d.line((xx, lim, xx + 12, lim), fill=FG, width=3)
    d.text((x1 - 330, lim - 36), "KP4 FEC LIMIT 2.4E-4", font=f(24), fill=FG)
    d.text((x0 + 8, y1 + 12), "LOG BER  vs  RX OMA, dBm (VOA)", font=f(26), fill=DIM)
    im.save(os.path.join(out, "scr_rx_base.png"))
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    c = ImageDraw.Draw(cv)
    pts = []
    for i in range(13):
        u = i / 12
        e = 2.6 + 8.6 * (math.exp(2.4 * u) - 1) / (math.exp(2.4) - 1)     # waterfall: steeper as power rises (illustrative)
        pts.append((x0 + 20 + u * (x1 - x0 - 40), ylog(e)))
    c.line(pts, fill=FG + (255,), width=4)
    for p in pts:
        c.ellipse((p[0] - 8, p[1] - 8, p[0] + 8, p[1] + 8), fill=FG + (255,))
    cv.save(os.path.join(out, "scr_rx_curve.png"))


ROWS = [("L1", "3.1E-08"), ("L2", "1.2E-08"), ("L3", "5.6E-09"), ("L4", "2.4E-08"), ("L5", "8.9E-09"), ("L6", "1.7E-08"),
        ("L7", "4.3E-09"), ("L8", "2.0E-08")]
ROW_Y0, ROW_DY = 120, 44


def link(out):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    frame(d, "FULL LINK  LOOPBACK", "8 x 100G PAM4")
    d.text((34, 76), "LANE   PRE-FEC BER", font=f(26), fill=DIM)
    d.line((30, 112, 450, 112), fill=GRID, width=2)
    d.text((34, ROW_Y0 + 8 * ROW_DY + 14), "LIMIT  2.4E-4  (KP4 FEC)", font=f(24), fill=DIM)
    d.rectangle((480, 78, W - 30, 400), outline=GRID, width=2)     # live eye quad (Blender)
    d.text((492, 84), "HOST RX EYE", font=f(24), fill=DIM)
    im.save(os.path.join(out, "scr_link_base.png"))
    rw = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    r = ImageDraw.Draw(rw)
    for k, (ln, ber) in enumerate(ROWS):
        y = ROW_Y0 + k * ROW_DY
        r.text((34, y), ln, font=f(32, True), fill=FG + (255,))
        r.text((130, y), ber, font=f(32, True), fill=FG + (255,))
        r.text((360, y), "OK", font=f(32), fill=OK + (255,))
    rw.save(os.path.join(out, "scr_link_rows.png"))


if __name__ == "__main__":
    o = sys.argv[1]
    os.makedirs(o, exist_ok=True)
    tx(o)
    rx(o)
    link(o)
    print("wrote screens to", o, "rows y0", ROW_Y0, "dy", ROW_DY)
