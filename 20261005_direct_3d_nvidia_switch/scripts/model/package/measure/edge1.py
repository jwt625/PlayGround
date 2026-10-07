import json, sys
import numpy as np, cv2
sys.path.insert(0, "scripts")
from tools.geom import load_views
V = load_views(2)
fr = json.load(open("config/frames.json"))["frames"]["package"]; R = np.array(fr["R"]); t = np.array(fr["t_mm"])
def edge(view, axis, c, a0, a1, h, sign):
    s = np.arange(a0, a1, 0.1)
    pts = np.stack([s, np.full_like(s, c), np.full_like(s, h)], 1) if axis == "x" else np.stack([np.full_like(s, c), s, np.full_like(s, h)], 1)
    uv, _ = V[view].project((pts @ R.T + t) / 1e3)
    g = V[view].image(gray=True).astype(np.float32)
    I = cv2.remap(g, uv[:, 0:1].astype(np.float32), uv[:, 1:2].astype(np.float32), cv2.INTER_LINEAR).ravel()
    I = np.convolve(I, np.ones(5) / 5, "same"); gr = np.gradient(I) * sign
    gr[:5] = gr[-5:] = 0
    return s[np.argmax(gr)]
h = 2.0
rows = []
for view in ["IMG_5711", "IMG_5712", "IMG_5717"]:
    r = {}
    for c in [-40, -20, 0, 20, 30]:
        r[f"L@y{c}"] = edge(view, "x", c, -64, -52, h, +1)   # bg -> ring (dark->bright)
        r[f"R@y{c}"] = edge(view, "x", c, 45, 57, h, -1)
        r[f"B@x{c}"] = edge(view, "y", c, -64, -52, h, +1)
        r[f"T@x{c}"] = edge(view, "y", c, 45, 57, h, -1)
    rows.append((view, r))
keys = rows[0][1].keys()
for k in keys: print(k, " ".join(f"{r[k]:7.2f}" for _, r in rows), f" med {np.median([r[k] for _, r in rows]):.2f}")
