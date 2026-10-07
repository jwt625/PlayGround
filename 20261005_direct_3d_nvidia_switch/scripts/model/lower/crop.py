"""crop.py VIEW u0 u1 v0 v1 [step] [maxw] -> labeled grid crop in s2 pixel coords"""
import sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[3]
v, u0, u1, v0, v1 = sys.argv[1], *map(int, sys.argv[2:6])
step = int(sys.argv[6]) if len(sys.argv) > 6 else 50
maxw = int(sys.argv[7]) if len(sys.argv) > 7 else 1400
assert v not in ("IMG_5715", "IMG_5728", "IMG_5737", "IMG_5747", "IMG_5750")
im = cv2.imread(str(ROOT / f"data/undistorted_s2/{v}.jpg"), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
c = im[v0:v1, u0:u1]
s = min(maxw / c.shape[1], 1400 / c.shape[0], 3.0)
c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
for u in range((u0 // step + 1) * step, u1, step):
    x = int((u - u0) * s); cv2.line(c, (x, 0), (x, c.shape[0]), (0, 200, 255), 1)
    cv2.putText(c, str(u), (x + 2, 12), 0, 0.4, (0, 255, 255), 1)
for w in range((v0 // step + 1) * step, v1, step):
    y = int((w - v0) * s); cv2.line(c, (0, y), (c.shape[1], y), (0, 200, 255), 1)
    cv2.putText(c, str(w), (2, y - 2), 0, 0.4, (0, 255, 255), 1)
out = ROOT / f"outputs/scratch/lower/crop_{v}_{u0}_{v0}.jpg"
cv2.imwrite(str(out), c, [cv2.IMWRITE_JPEG_QUALITY, 85]); print(out, s)
