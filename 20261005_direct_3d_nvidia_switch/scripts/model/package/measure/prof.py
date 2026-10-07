import json, sys
import numpy as np, cv2
sys.path.insert(0, "scripts")
from tools.geom import load_views
V = load_views(2)
fr = json.load(open("config/frames.json"))["frames"]["package"]
R = np.array(fr["R"]); t = np.array(fr["t_mm"])
def samp(view, pts):
    uv, _ = V[view].project((pts @ R.T + t) / 1e3)
    g = V[view].image(gray=True).astype(np.float32)
    return cv2.remap(g, uv[:, 0:1].astype(np.float32), uv[:, 1:2].astype(np.float32), cv2.INTER_LINEAR).ravel()
view = sys.argv[1]; axis = sys.argv[2]; c = float(sys.argv[3]); a0, a1 = float(sys.argv[4]), float(sys.argv[5]); h = float(sys.argv[6]) if len(sys.argv) > 6 else 0
s = np.arange(a0, a1, 0.25)
pts = np.stack([s, np.full_like(s, c), np.full_like(s, h)], 1) if axis == "x" else np.stack([np.full_like(s, c), s, np.full_like(s, h)], 1)
I = samp(view, pts)
# smooth and report strong gradient positions
Is = np.convolve(I, np.ones(3) / 3, "same")
g = np.gradient(Is)
idx = np.argsort(-np.abs(g))
picked = []
for i in idx:
    if all(abs(i - j) > 6 for j in picked) and 2 < i < len(s) - 3:
        picked.append(i)
    if len(picked) >= 8: break
print(view, axis, "@", c, "edges:", ", ".join(f"{s[i]:.2f}({g[i]:+.0f})" for i in sorted(picked)))
