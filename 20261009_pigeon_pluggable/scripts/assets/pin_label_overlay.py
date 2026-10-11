"""Draw OSFP pin numbers next to every foot / belly pad on the bottom-view render.

Run from the project folder:
  uv run --with pillow python scripts/assets/pin_label_overlay.py outputs/hero/work/pigeon_osfp_dr8_feet_pins_raw.png \
      outputs/hero/work/pigeon_osfp_dr8_feet_pins_labels.json outputs/hero/pigeon_osfp_dr8_feet_pins.png
Colors by MSA class (Fig 15-1): GND gray, TX red, RX cyan, VCC orange, low-speed violet. Merged VCC pads get "45/46".
"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont

RAW, JS, OUT = sys.argv[1:4]
COL = {"GND": (190, 195, 200), "TX": (255, 90, 80), "RX": (60, 220, 255), "VCC": (255, 170, 40), "LS": (200, 140, 255)}
d = json.load(open(JS))
img = Image.open(RAW).convert("RGBA")
font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 15)
small = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 17)
ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
dr = ImageDraw.Draw(ov)
groups = {}
for r in d["pins"]:
    key = (round(r["anchor"][0]), round(r["anchor"][1]))
    groups.setdefault(key, []).append(r)
for key, rs in groups.items():
    rs.sort(key=lambda r: r["pin"])
    r = rs[0]
    text = "/".join(str(x["pin"]) for x in rs)
    ax, ay = r["anchor"]
    lx, ly = r["label"]
    dr.line([(ax, ay), (lx, ly)], fill=COL[r["cls"]] + (150,), width=1)
    tw = font.getlength(text)
    bx0, by0 = lx - tw / 2 - 3, ly - 9
    dr.rectangle([bx0, by0, bx0 + tw + 6, by0 + 18], fill=(6, 8, 12, 210))
    dr.text((lx - tw / 2, ly - 8), text, font=font, fill=COL[r["cls"]] + (255,))
x, y = 24, img.size[1] - 34
dr.rectangle([12, y - 8, 820, y + 26], fill=(6, 8, 12, 235))
for cls, nm in (("GND", "GND (long, mates 1st)"), ("VCC", "VCC (medium)"), ("TX", "TX signal"), ("RX", "RX signal"),
                ("LS", "low-speed")):
    dr.rectangle([x, y + 3, x + 14, y + 17], fill=COL[cls] + (255,))
    dr.text((x + 20, y), nm, font=small, fill=(225, 230, 235, 255))
    x += 30 + small.getlength(nm)
Image.alpha_composite(img, ov).convert("RGB").save(OUT)
print("PIN_LABELS_DONE", OUT, sum(len(v) for v in groups.values()), "pins in", len(groups), "pads")
