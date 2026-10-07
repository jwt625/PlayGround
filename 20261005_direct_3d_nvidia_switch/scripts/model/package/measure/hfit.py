"""Height of a textured patch by multi-view ortho NCC. hfit.py x y [half_mm] views..."""
import json, sys
import numpy as np, cv2
sys.path.insert(0, "scripts")
from tools.geom import load_views
V = load_views(2)
fr = json.load(open("config/frames.json"))["frames"]["package"]
R = np.array(fr["R"]); t = np.array(fr["t_mm"])
def patch(view, x, y, h, half, res=0.1):
    n = int(2 * half / res); xs = -half + (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xs + x, -xs + y)
    Lp = np.stack([X.ravel(), Y.ravel(), np.full(X.size, h)], 1)
    uv, _ = V[view].project((Lp @ R.T + t) / 1e3)
    g = V[view].image(gray=True).astype(np.float32)
    p = cv2.remap(g, uv[:, 0].reshape(n, n).astype(np.float32), uv[:, 1].reshape(n, n).astype(np.float32), cv2.INTER_LINEAR)
    p = p - p.mean(); return p / (np.linalg.norm(p) + 1e-6)
x, y, half = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
views = sys.argv[4].split(",")
hs = np.arange(-8, 10.01, 0.25); sc = []
for h in hs:
    P = [patch(v, x, y, h, half) for v in views]
    s = np.mean([(P[i] * P[j]).sum() for i in range(len(P)) for j in range(i + 1, len(P))])
    sc.append(s)
sc = np.array(sc); i = sc.argmax()
print(f"({x:g},{y:g}) best h {hs[i]:.2f}  ncc {sc[i]:.3f}  (ncc at h=0: {sc[np.argmin(abs(hs))]:.3f})")
