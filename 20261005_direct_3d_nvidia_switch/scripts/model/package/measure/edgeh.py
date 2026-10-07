"""edgeh.py axis c a0 a1 sign [views]: edge position per view vs plane height h; best h = min spread.
axis x: profile along x at y=c; axis y: along y at x=c. sign +1 dark->bright with increasing coordinate."""
import json, sys
import numpy as np, cv2
sys.path.insert(0, "scripts")
from tools.geom import load_views
V = load_views(2)
fr = json.load(open("config/frames.json"))["frames"]["package"]; R = np.array(fr["R"]); t = np.array(fr["t_mm"])
axis, c, a0, a1, sign = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
views = (sys.argv[6] if len(sys.argv) > 6 else "IMG_5711,IMG_5712,IMG_5717,IMG_5740").split(",")
def edge(view, h):
    s = np.arange(a0, a1, 0.05); w = np.full_like(s, c)
    pts = np.stack([s, w, np.full_like(s, h)], 1) if axis == "x" else np.stack([w, s, np.full_like(s, h)], 1)
    off = np.array([0, 1, 0]) if axis == "x" else np.array([1, 0, 0])
    I = 0
    for d in (-1.0, -0.5, 0, 0.5, 1.0):   # average 5 parallel lines (2 mm band)
        uv, _ = V[view].project(((pts + d * off) @ R.T + t) / 1e3)
        g = V[view].image(gray=True).astype(np.float32)
        I = I + cv2.remap(g, uv[:, 0:1].astype(np.float32), uv[:, 1:2].astype(np.float32), cv2.INTER_LINEAR).ravel()
    I = np.convolve(I, np.ones(5) / 5, "same"); gr = np.gradient(I) * sign; gr[:6] = gr[-6:] = 0
    return s[np.argmax(gr)]
best = None
for h in np.arange(-1, 5.01, 0.25):
    e = [edge(v, h) for v in views]; sp = np.std(e)
    if best is None or sp < best[1]: best = (h, sp, e)
print(f"{axis}@{c:g} [{a0:g},{a1:g}] best h {best[0]:.2f} spread {best[1]:.2f} edges {np.round(best[2], 2).tolist()}")
