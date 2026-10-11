"""Compose two 800x1200 panels side by side with small captions (flex routing close-up).

Run: uv run --with pillow python scripts/assets/compose_panels.py left.png right.png out.png "caption left" "caption right"
"""
import sys

from PIL import Image, ImageDraw, ImageFont

a, b, out, ca, cb = sys.argv[1:6]
A, B = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
img = Image.new("RGB", (A.width + B.width, max(A.height, B.height)), (0, 0, 0))
img.paste(A, (0, 0))
img.paste(B, (A.width, 0))
dr = ImageDraw.Draw(img)
dr.line([(A.width, 0), (A.width, img.height)], fill=(60, 70, 80), width=2)
font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 22)
for x0, t in ((16, ca), (A.width + 16, cb)):
    w = font.getlength(t)
    dr.rectangle([x0 - 6, 12, x0 + w + 6, 44], fill=(6, 8, 12))
    dr.text((x0, 14), t, font=font, fill=(225, 230, 235))
img.save(out)
print("COMPOSED", out)
