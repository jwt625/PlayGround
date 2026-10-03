"""Palette-quantize preview PNGs in place (keeps 900x675) to stay under the size budget. usage: python shrink_previews.py dir [colors]"""
import os, sys
from PIL import Image
d = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 160
tot0 = tot1 = 0
for f in sorted(os.listdir(d)):
    if not f.endswith(".png"):
        continue
    p = os.path.join(d, f)
    s0 = os.path.getsize(p)
    im = Image.open(p).convert("RGB")
    q = im.quantize(colors=n, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG)
    q.save(p, optimize=True)
    tot0 += s0
    tot1 += os.path.getsize(p)
print("before", tot0 // 1024, "KB after", tot1 // 1024, "KB")
