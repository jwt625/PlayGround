"""Like prof.py but edges along a (columns): prof over b-chunks of |d/da|. profa.py VIEW AXIS VALUE a0 a1 b0 b1 [ppmm]"""
import sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.geom import load_views
a = sys.argv[1:]
view, ax, val = a[0], a[1], float(a[2]); a0, a1, b0, b1 = map(float, a[3:7]); ppm = float(a[7]) if len(a) > 7 else 2
v = load_views(2)[view]; img = cv2.cvtColor(v.image(), cv2.COLOR_BGR2GRAY).astype(np.float32)
A = np.arange(a0, a1, 1 / ppm); B = np.arange(b0, b1, 1 / ppm)
AA, BB = np.meshgrid(A, B); V = np.full(AA.shape, val)
X = {"z": (AA, BB, V), "x": (V, AA, BB), "y": (AA, V, BB)}[ax]
P = np.stack([x.ravel() for x in X], 1) / 1e3
uv, d = v.project(P)
inside = ((uv[:, 0] > 0) & (uv[:, 0] < img.shape[1]) & (uv[:, 1] > 0) & (uv[:, 1] < img.shape[0])).reshape(AA.shape)
out = cv2.remap(img, uv[:, 0].reshape(AA.shape).astype(np.float32), uv[:, 1].reshape(AA.shape).astype(np.float32), cv2.INTER_LINEAR)
g = np.abs(np.diff(cv2.GaussianBlur(out, (0, 0), 1.0), axis=1)); gi = inside[:, 1:] & inside[:, :-1]
n = int(a[8]) if len(a) > 8 else 3
for k in range(n):
    sl = slice(k * len(B) // n, (k + 1) * len(B) // n)
    m = gi[sl].sum(0); prof = np.where(m > 5, (g[sl] * gi[sl]).sum(0) / np.maximum(m, 1), 0)
    pk = [i for i in range(1, len(prof) - 1) if prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]]
    pk = sorted(pk, key=lambda i: -prof[i])[:6]
    print(f"b {B[sl][0]:.0f}..{B[sl][-1]:.0f}: " + ", ".join(f"{(A[i] + A[i + 1]) / 2:.1f}({prof[i]:.0f})" for i in sorted(pk)))
