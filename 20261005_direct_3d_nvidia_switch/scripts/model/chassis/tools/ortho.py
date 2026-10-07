"""Orthophoto: resample a view onto an axis plane (world mm) with an mm grid.
usage: ortho.py VIEW AXIS VALUE a0 a1 b0 b1 [ppmm=1] [step=50] [flip]
AXIS z: image cols = X (a), rows = Y (b). AXIS x: cols = Y (a), rows = Z (b, top=max). AXIS y: cols = X (a), rows = Z."""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.geom import load_views
a = sys.argv[1:]
view, ax, val = a[0], a[1], float(a[2])
a0, a1, b0, b1 = map(float, a[3:7])
ppm = float(a[7]) if len(a) > 7 else 1.0
step = float(a[8]) if len(a) > 8 else 50
v = load_views(2)[view]
img = v.image()
A = np.arange(a0, a1, 1 / ppm); B = np.arange(b0, b1, 1 / ppm)
if ax != "z":
    B = B[::-1]
AA, BB = np.meshgrid(A, B)
V = np.full(AA.shape, val)
X = {"z": (AA, BB, V), "x": (V, AA, BB), "y": (AA, V, BB)}[ax]
P = np.stack([x.ravel() for x in X], 1) / 1e3
uv, d = v.project(P)
mx = uv[:, 0].reshape(AA.shape).astype(np.float32); my = uv[:, 1].reshape(AA.shape).astype(np.float32)
out = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=(40, 0, 40))
out[(d.reshape(AA.shape) <= 0)] = 0
for i, x in enumerate(A):
    if abs(x - step * round(x / step)) < 0.5 / ppm:
        out[:, i] = (0, 255, 255) if abs(x) < 1e-6 else (0, 200, 0)
        for j in range(0, len(B), int(200 * ppm)):
            cv2.putText(out, f"{x:.0f}", (i + 2, j + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
for j, y in enumerate(B):
    if abs(y - step * round(y / step)) < 0.5 / ppm:
        out[j, :] = (0, 255, 255) if abs(y) < 1e-6 else (0, 200, 0)
        for i in range(0, len(A), int(200 * ppm)):
            cv2.putText(out, f"{y:.0f}", (i + 2, j - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
Path("outputs/scratch/chassis").mkdir(parents=True, exist_ok=True)
o = Path("outputs/scratch/chassis") / f"ortho_{view}_{ax}{val:g}_{a0:g}_{a1:g}_{b0:g}_{b1:g}.jpg"
cv2.imwrite(str(o), out); print(o, out.shape)
