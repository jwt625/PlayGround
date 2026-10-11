"""Tile the FX test frames into strips with frame labels (workstream T, DevLog-008 Sec. 8).

Run from the project folder:
  uv run --with pillow python scripts/assets/optics_tx/compose_tx_fx_strip.py outputs/assets/optics_tx/fx_test outputs/assets/optics_tx
"""
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFont

SRC, OUT = sys.argv[1], sys.argv[2]
FONT = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 18)
NOTE = "* light invisible (1311 nm); rendered for viewing. Electrical = amber, optical = red."
for tag, cols, title in (("fxA", 5, "TX light-up (30 fps): DFB on f10, front f10-80, MZM outputs modulated from f51, RF glow from f40"),
                         ("fxB", 3, "TX RF signal glow (30 fps): flex contacts -> carrier pairs -> GSSG bonds -> TW electrodes, from f40")):
    fs = sorted(glob.glob(os.path.join(SRC, tag + "_*.png")))
    w, h = Image.open(fs[0]).size
    rows = (len(fs) + cols - 1) // cols
    im = Image.new("RGB", (w * cols, h * rows + 64), (6, 8, 12))
    dr = ImageDraw.Draw(im)
    for i, f in enumerate(fs):
        x, y = (i % cols) * w, (i // cols) * h + 32
        im.paste(Image.open(f), (x, y))
        dr.rectangle([x + 8, y + 8, x + 92, y + 34], fill=(8, 12, 18))
        dr.text((x + 14, y + 11), "f %d" % int(os.path.basename(f)[4:8]), fill=(235, 240, 245), font=FONT)
    dr.text((10, 6), title, fill=(235, 240, 245), font=FONT)
    dr.text((10, h * rows + 38), NOTE, fill=(235, 240, 245), font=FONT)
    p = os.path.join(OUT, "optics_tx_fx_strip_%s.png" % ("lightup" if tag == "fxA" else "signal"))
    im.save(p)
    print("STRIP_OK", p, im.size)
