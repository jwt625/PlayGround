"""Orthophoto on plane z = Z (world mm) from one train view, with a labeled mm grid.
usage: uv run python scripts/model/lower/ortho.py VIEW Z x0 x1 y0 y1 [px_per_mm] [out]"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from tools.geom import load_views  # noqa: E402

a = sys.argv[1:]
view, Z = a[0], float(a[1])
x0, x1, y0, y1 = map(float, a[2:6])
s = float(a[6]) if len(a) > 6 else 3.0
out = a[7] if len(a) > 7 else str(ROOT / f"outputs/scratch/lower/ortho_{view}_{int(Z)}.jpg")
assert view not in ("IMG_5715", "IMG_5728", "IMG_5737", "IMG_5747", "IMG_5750")
v = load_views(2)[view]
img = v.image()
xs = np.arange(x0, x1, 1 / s)
ys = np.arange(y0, y1, 1 / s)
X, Y = np.meshgrid(xs, ys)
P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, Z)], 1) / 1e3
uv, d = v.project(P)
mx = uv[:, 0].reshape(X.shape).astype(np.float32)
my = uv[:, 1].reshape(X.shape).astype(np.float32)
o = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=(40, 40, 40))
# rows = Y increasing downward, cols = X increasing to the right
step = 10 if (x1 - x0) < 250 else 25
for x in np.arange(np.ceil(x0 / step) * step, x1, step):
    c = int((x - x0) * s)
    cv2.line(o, (c, 0), (c, o.shape[0]), (0, 255, 255) if x % 50 == 0 else (0, 160, 160), 1)
    cv2.putText(o, f"{int(x)}", (c + 2, 12), 0, 0.4, (0, 255, 255), 1)
for y in np.arange(np.ceil(y0 / step) * step, y1, step):
    r = int((y - y0) * s)
    cv2.line(o, (0, r), (o.shape[1], r), (0, 255, 255) if y % 50 == 0 else (0, 160, 160), 1)
    cv2.putText(o, f"{int(y)}", (2, r - 2), 0, 0.4, (0, 255, 255), 1)
cv2.imwrite(out, o, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(out, o.shape)
