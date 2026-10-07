"""anaglyph of two orthos at plane z=h: ana.py V1 V2 h [half res]"""
import json, sys
import numpy as np, cv2
sys.path.insert(0, "scripts")
from tools.geom import load_views
V = load_views(2)
fr = json.load(open("config/frames.json"))["frames"]["package"]
R = np.array(fr["R"]); t = np.array(fr["t_mm"])
def ortho(view, h, half, res, cx=0, cy=0):
    n = int(2 * half / res); xs = -half + (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xs + cx, -xs + cy)
    Lp = np.stack([X.ravel(), Y.ravel(), np.full(X.size, h)], 1)
    uv, _ = V[view].project((Lp @ R.T + t) / 1e3)
    g = V[view].image(gray=True)
    return cv2.remap(g, uv[:, 0].reshape(n, n).astype(np.float32), uv[:, 1].reshape(n, n).astype(np.float32), cv2.INTER_LINEAR), n
v1, v2, h = sys.argv[1], sys.argv[2], float(sys.argv[3])
half = float(sys.argv[4]) if len(sys.argv) > 4 else 75; res = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
cx = float(sys.argv[6]) if len(sys.argv) > 6 else 0; cy = float(sys.argv[7]) if len(sys.argv) > 7 else 0
a, n = ortho(v1, h, half, res, cx, cy); b, _ = ortho(v2, h, half, res, cx, cy)
out = np.dstack([b, b, a])
step = 10 if half > 30 else 2
for k in np.arange(-half // step * step, half + 0.01, step):
    p = int((k + half) / res)
    if 0 <= p < n:
        cv2.line(out, (p, 0), (p, n - 1), (0, 160, 0), 1); cv2.line(out, (0, n - 1 - p), (n - 1, n - 1 - p), (0, 160, 0), 1)
        cv2.putText(out, f"{k+cx:g}", (p + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        cv2.putText(out, f"{k+cy:g}", (2, n - 1 - p - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
o = f"outputs/scratch/package/ana_{v1[-4:]}_{v2[-4:]}_z{h:g}_{cx:g}_{cy:g}.jpg"
cv2.imwrite(o, out); print(o)
