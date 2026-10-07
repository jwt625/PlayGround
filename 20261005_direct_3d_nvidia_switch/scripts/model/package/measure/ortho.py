"""Orthophoto of the package plane z=h (package frame) from one view. usage: ortho.py VIEW h [half_mm] [mm_per_px] [extra_R_deg tx ty tz]"""
import json, sys
import numpy as np, cv2
sys.path.insert(0, "scripts")
from tools.geom import load_views
V = load_views(2)
view = sys.argv[1]; h = float(sys.argv[2]); half = float(sys.argv[3]) if len(sys.argv) > 3 else 75; res = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
fr = json.load(open("config/frames.json"))["frames"]["package"]
R = np.array(fr["R"]); t = np.array(fr["t_mm"])
v = V[view]
n = int(2 * half / res)
xs = -half + (np.arange(n) + 0.5) * res
X, Y = np.meshgrid(xs, -xs)  # image row 0 = +y
Lp = np.stack([X.ravel(), Y.ravel(), np.full(X.size, h)], 1)
W = (Lp @ R.T + t) / 1e3
uv, z = v.project(W)
img = v.image()
mx = uv[:, 0].reshape(n, n).astype(np.float32); my = uv[:, 1].reshape(n, n).astype(np.float32)
out = cv2.remap(img, mx, my, cv2.INTER_LINEAR)
# grid every 10 mm, labels every 20
for k in range(-int(half) // 10 * 10, int(half) + 1, 10):
    p = int((k + half) / res)
    col = (0, 0, 255) if k == 0 else (0, 200, 255)
    cv2.line(out, (p, 0), (p, n - 1), col, 1); cv2.line(out, (0, n - 1 - p), (n - 1, n - 1 - p), col, 1)
    if k % 20 == 0:
        cv2.putText(out, str(k), (p + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(out, str(k), (2, n - 1 - p - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
o = f"outputs/scratch/package/ortho_{view}_z{h:g}.jpg"
cv2.imwrite(o, out); print(o)
