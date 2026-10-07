"""Fit the stand base (horizontal rounded plate) to outline points read from brightened s4 crops of
IMG_5711/5712. Params: yaw about gravity up (deg), shift along X and along h (mm), width, depth."""
import json, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, "scripts")
from tools.geom import load_views
V = load_views(4)
fr = json.load(open("config/frames.json"))["frames"]; R = np.array(fr["package"]["R"]); t = np.array(fr["package"]["t_mm"])
up = R.T @ np.array(fr["scene"]["R"])[:, 2]; up /= np.linalg.norm(up)
x0 = np.array([1, 0, 0.]); x0 -= up * (x0 @ up); x0 /= np.linalg.norm(x0)
h0 = np.cross(up, x0); h0 /= np.linalg.norm(h0); h0 = h0 if h0[2] > 0 else -h0
f = np.array([-3, 97.5, -39.3])
# outline targets: (view, edge, u, v) edge in {left(-x), right(+x), front}
T = [("IMG_5711", "left", 1000, 360), ("IMG_5711", "left", 1100, 360), ("IMG_5711", "left", 1165, 362),
     ("IMG_5711", "right", 1050, 693), ("IMG_5711", "right", 1150, 695),
     ("IMG_5711", "front", 1175, 450), ("IMG_5711", "front", 1175, 600),
     ("IMG_5712", "left", 1000, 260), ("IMG_5712", "left", 1135, 330), ("IMG_5712", "left", 1265, 398),
     ("IMG_5712", "front", 1270, 425), ("IMG_5712", "front", 1200, 555), ("IMG_5712", "front", 1135, 675)]
def corners(p):
    yaw, sx, sh, W, D, tx, th = p
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    xb = c * x0 + s * h0; hb = -s * x0 + c * h0
    a, b = np.radians(tx), np.radians(th)   # tilt: hb rotated about xb by tx, xb rotated about hb by th
    hb = np.cos(a) * hb + np.sin(a) * np.cross(xb, hb)
    xb = np.cos(b) * xb + np.sin(b) * np.cross(hb, xb)
    fc = f + sx * x0 + sh * h0
    return {"FL": fc - W / 2 * xb, "FR": fc + W / 2 * xb, "BL": fc - W / 2 * xb - D * hb, "BR": fc + W / 2 * xb - D * hb}
def proj(v, X): return V[v].project((R @ X + t) / 1e3)[0][0]
def seg_d(p, a, b):
    ab = b - a; s = np.clip((p - a) @ ab / (ab @ ab), 0, 1); return np.linalg.norm(p - a - s * ab)
E = {"left": ("FL", "BL"), "right": ("FR", "BR"), "front": ("FL", "FR")}
def loss(p):
    C = corners(p); L = 0
    for v, e, u, w in T:
        a, b = (proj(v, C[k]) for k in E[e]); L += min(seg_d(np.array([u, w], float), a, b), 60) ** 2
    return L
p0 = np.array([0, 0, 0, 72, 72., 0, 0])
sim = np.vstack([p0] + [p0 + np.eye(7)[i] * [10, 5, 5, 8, 15, 10, 10][i] for i in range(7)])
r = minimize(loss, p0, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 0.05, "fatol": 0.05, "initial_simplex": sim})
print("start rms px", np.sqrt(loss(p0) / len(T)).round(1), "fit rms px", np.sqrt(r.fun / len(T)).round(1))
print("yaw, shift_x, shift_h, width, depth, tilt_x, tilt_h =", r.x.round(2))
C = corners(r.x); print({k: v.round(1).tolist() for k, v in C.items()})
