"""caps.py: detect discrete components (caps/resistors) on the light-blue field of the package top texture and
write their bounding rectangles (package mm) to scripts/model/package/caps.json (relief test, Wave 2).
Field = ring opening minus OE footprints and lid (0.4 mm margin); a component = connected blob brighter than
the local field (gray > field median + --delta) of 0.15-3 mm^2. Debug overlay: outputs/scratch/package/caps.png"""
import json, sys, tomllib
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, "scripts/model/package")
from bake_top import part_map
delta = float(sys.argv[1]) if len(sys.argv) > 1 else 25
P = tomllib.loads(Path("config/model/package.toml").read_text())
tex = cv2.imread(P["texture"]["top"])
h, w = tex.shape[:2]; ppm = 10.0
hx, hy = P["substrate"]["size_x"] / 2, P["substrate"]["size_y"] / 2
X, Y = np.meshgrid(-hx + (np.arange(w) + 0.5) / ppm, hy - (np.arange(h) + 0.5) / ppm)
names, idx, _ = part_map(P, X, Y)
field = (idx == names.index("package.field")).astype(np.uint8)
field = cv2.erode(field, np.ones((9, 9), np.uint8)) > 0
g = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY).astype(float)
bg = cv2.medianBlur(tex, 31)
gb = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(float)
m = ((g - gb > delta) & field).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
rects = []
dbg = tex.copy()
for i in range(1, n):
    x, y, ww, hh, a = st[i]
    if not (15 <= a <= 300) or max(ww, hh) > 30:
        continue
    x0, x1 = -hx + x / ppm, -hx + (x + ww) / ppm
    y1, y0 = hy - y / ppm, hy - (y + hh) / ppm
    rects.append([round(x0, 2), round(x1, 2), round(y0, 2), round(y1, 2)])
    cv2.rectangle(dbg, (x, y), (x + ww - 1, y + hh - 1), (0, 0, 255), 1)
json.dump({"delta": delta, "rects_mm": rects}, open("scripts/model/package/caps.json", "w"))
cv2.imwrite("outputs/scratch/package/caps.png", dbg[int((53.9 - 45) * 10):int((53.9 - 5) * 10), int((54.15 + 5) * 10):int((54.15 + 45) * 10)])
print(len(rects), "components")
