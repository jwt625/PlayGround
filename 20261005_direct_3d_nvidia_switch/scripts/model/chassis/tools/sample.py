import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.geom import load_views
V = load_views(2)
def samp(view, pts):
    v = V[view]; img = v.image()
    uv, d = v.project(np.array(pts) / 1e3)
    out = []
    for (u, w) in uv:
        u, w = int(round(u)), int(round(w))
        if 3 <= u < img.shape[1] - 3 and 3 <= w < img.shape[0] - 3:
            out.append(img[w - 3:w + 4, u - 3:u + 4].reshape(-1, 3))
    a = np.concatenate(out)[:, ::-1]
    return np.median(a, 0), len(out)
def grid(x, y, z):
    return [(a, b, c) for a in np.atleast_1d(x) for b in np.atleast_1d(y) for c in np.atleast_1d(z)]
roll = lambda p: (p[0], p[1], p[2] + 0.0127 * p[0])
tests = {
 'wall_mx_out 5756': ('IMG_5756', grid(-219.5, np.arange(300, 700, 20), np.arange(-50, 0, 10))),
 'wall_mx_out 5755': ('IMG_5755', grid(-219.5, np.arange(300, 700, 20), np.arange(-50, 0, 10))),
 'wall_px_in 5744': ('IMG_5744', grid(217.5, np.arange(400, 560, 15), np.arange(-40, 10, 10))),
 'crossbar 5749': ('IMG_5749', grid(np.arange(-150, 150, 15), [352], [0.5])),
 'crossbar 5751': ('IMG_5751', grid(np.arange(-150, 150, 15), [352], [0.5])),
 'duct_top 5749': ('IMG_5749', grid(np.arange(150, 205, 8), np.arange(120, 300, 15), [-3])),
 'mmc 5740': ('IMG_5740', grid(np.arange(-204, -120, 6), [17], np.arange(-45, -12, 6))),
 'bezel plate 5740': ('IMG_5740', grid(np.arange(90, 115, 4), [23], np.arange(-40, -12, 5))),
 'bezel plate 5738': ('IMG_5738', grid(np.arange(-110, -90, 4), [23], np.arange(-40, -12, 5))),
 'grip 5740': ('IMG_5740', grid(np.arange(-200, 200, 20), [-15], [-38])),
 'lip 5740': ('IMG_5740', grid(np.arange(-200, 200, 20), [12], [22])),
 'floor rear 5749': ('IMG_5749', grid(np.arange(-150, 150, 20), [790], [-60])),
 'divider 5744': ('IMG_5744', grid(np.arange(-150, 150, 20), [578], np.arange(-30, 15, 8))),
}
for k, (v, pts) in tests.items():
    m, n = samp(v, [roll(p) for p in pts]); print(f'{k:20s} n={n:3d} rgb8={m.round(0)}')
