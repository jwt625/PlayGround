"""Tile the RX FX test frames into strips with frame numbers.

  uv run --with pillow python scripts/assets/optics_rx/compose_rx_fx_strip.py outputs/assets/optics_rx/fx_test outputs/assets/optics_rx
-> optics_rx_fx_strip_arrival.png (fxA_*) and optics_rx_fx_strip_signal.png (fxB_*).
"""
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFont

SRC, DST = sys.argv[1:3]
FONT = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 22)
NOTE = "* beams invisible (1311 nm); rendered for viewing"
for tag, name in (("A", "arrival"), ("B", "signal"), ("C", "bits_tia")):
    files = sorted(glob.glob(os.path.join(SRC, "fx%s_*.png" % tag)))
    if not files:
        continue
    ims = [Image.open(f).convert("RGB") for f in files]
    w, h = ims[0].size
    cols = min(5, len(ims))
    rows = (len(ims) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, rows * h + 40), (10, 12, 16))
    d = ImageDraw.Draw(out)
    for i, (f, im) in enumerate(zip(files, ims)):
        x, y = (i % cols) * w, (i // cols) * h
        out.paste(im, (x, y))
        d.rectangle([x + 6, y + 6, x + 90, y + 36], fill=(0, 0, 0))
        d.text((x + 12, y + 8), "f%d" % int(os.path.basename(f)[4:8]), font=FONT, fill=(235, 240, 245))
    d.text((10, rows * h + 8), NOTE + ("" if tag == "A" else "; amber = electrical signal"), font=FONT, fill=(235, 240, 245))
    p = os.path.join(DST, "optics_rx_fx_strip_%s.png" % name)
    out.save(p)
    print("wrote", p, out.size)
