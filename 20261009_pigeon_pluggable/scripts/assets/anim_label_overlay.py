"""Draw the fading callout labels onto the explode-orbit frames (camera-facing 2D callouts, readable during the orbit).

Run: uv run --with pillow python scripts/assets/anim_label_overlay.py outputs/hero/frames_explode outputs/hero/work/anim_label_anchors_0001_0360.json
Loop v2 timing: labels fade in from frame 151 (0.4-frame stagger, 9-frame fade), hold, fade out over frames
259-277; no labels outside 151-277. Column assignment and order are fixed at the
first label frame so labels do not jump; their heights follow the anchors with a minimum gap.
"""
import glob
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FR, JS = sys.argv[1:3]
d = {int(k): v for k, v in json.load(open(JS)).items()}
font = ImageFont.truetype("/System/Library/Fonts/HelveticaNeue.ttc", 21)
LINE, TEXT, BOX = (150, 210, 230), (235, 240, 245), (8, 12, 18)
F_LAB, STAG, FADE = 151, 0.4, 9
F_OUT0, F_OUT1 = 259, 277
first = d[min(f for f in d if f >= F_LAB)]
xs = sorted(r[1] for r in first)
split = xs[len(xs) // 2]
side = {r[0]: ("L" if r[1] < split else "R") for r in first}
order = {s: [r[0] for r in sorted(first, key=lambda r: r[2]) if side[r[0]] == s] for s in ("L", "R")}
idx = {r[0]: i for i, r in enumerate(first)}
H = W = 1080
n = 0
cut = []                 # (frame, label) whose box or anchor falls outside the frame
for path in sorted(glob.glob(os.path.join(FR, "frame_*.png"))):
    f = int(os.path.basename(path)[6:10])
    if f < F_LAB or f >= F_OUT1 or f not in d:
        continue
    rows = {r[0]: (r[1], r[2]) for r in d[f]}
    img = Image.open(path).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    for s in ("L", "R"):
        names = order[s]
        hgt = 30
        ys = [rows[t][1] - hgt / 2 for t in names]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + hgt + 6)
        over = ys[-1] + hgt + 10 - H if ys else 0
        if over > 0:
            ys = [y - over for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - hgt - 6)
        for t, y in zip(names, ys):
            a = max(0.0, min(1.0, (f - (F_LAB + idx[t] * STAG)) / FADE, (F_OUT1 - f) / (F_OUT1 - F_OUT0)))
            if a <= 0:
                continue
            y = max(8, y)
            tw = font.getlength(t)
            w = tw + 14
            x = 14 if s == "L" else W - 14 - w
            ax, ay = rows[t]
            ex = x + w if s == "L" else x
            knee = ex + (16 if s == "L" else -16)
            al = int(255 * a)
            dr.line([(ax, ay), (knee, y + hgt / 2), (ex, y + hgt / 2)], fill=LINE + (al,), width=2)
            dr.ellipse([ax - 4, ay - 4, ax + 4, ay + 4], fill=LINE + (al,))
            dr.rectangle([x, y, x + w, y + hgt], fill=BOX + (int(215 * a),), outline=LINE + (al,), width=1)
            dr.text((x + 7, y + 4), t, font=font, fill=TEXT + (al,))
            if x < 0 or x + w > W or y < 0 or y + hgt > H or not (0 <= ax <= W and 0 <= ay <= H):
                cut.append((f, t))
    Image.alpha_composite(img, ov).convert("RGB").save(path)
    n += 1
print("ANIM_LABELS_DONE frames", n, "labels cut/off-frame:", len(cut), cut[:5])
