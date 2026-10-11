"""Fit check for the T / X optics subassemblies inside the hero head (no Blender needed).

uv run --with trimesh --with numpy python scripts/assets/fit_eye_subassy.py
Places every vertex of assets/components/optics_<side>/optics_<side>.glb with the same eye frame the build uses
(aperture on the head ellipsoid + offset along the horizontal surface normal, +Z kept up) and reports, per candidate
eye position, the worst ellipsoid level of vertices outside the eye opening (level < 1 = inside the inner skin) and
the clearance to the head board / MCU. Numbers in mm.
Optional args: E_Y E_Z offset yaw_k -> also print the result for that eye placement (build: -49.0 81.5 -1.0 0).
"""
import math
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
COMP = os.path.join(HERE, "..", "..", "assets", "components")
HC = np.array([0.0, -44.0, 81.5])
HA = np.array([12.0, 14.0, 12.0])
SKIN_IN = 0.3 + 0.25          # half skin thickness + clearance
ZB0 = HC[2] + 4.9
R_SOCKET = 4.2                # eye socket / eye-cere radius: vertices inside it and less than 1.6 mm behind the
X_SOCKET = -1.6               # aperture sit in the socket and are exempt from the skin test (D)


def load(side):
    sc = trimesh.load(os.path.join(COMP, "optics_%s" % side, "optics_%s.glb" % side), force="scene")
    vs = []
    for name in sc.graph.nodes_geometry:
        if "beam" in name:
            continue
        T, g = sc.graph[name]
        m = sc.geometry[g].copy()
        m.apply_transform(T)
        v = m.vertices * 1000.0
        vs.append(np.c_[v[:, 0], -v[:, 2], v[:, 1]])
    return np.vstack(vs)


def frame(side, e_y, e_z, off, yaw_k=1.0):
    s = -1 if side == "tx" else 1
    dy, dz = (e_y - HC[1]) / HA[1], (e_z - HC[2]) / HA[2]
    xs = HA[0] * math.sqrt(1 - dy * dy - dz * dz)
    p = np.array([s * xs, e_y, e_z])
    g = (p - HC) / HA ** 2
    n = np.array([g[0], g[1] * yaw_k, 0.0])
    n /= np.linalg.norm(n)
    ex, ez = n, np.array([0.0, 0, 1])
    ey = np.cross(ez, ex)
    return p + n * off, np.c_[ex, ey, ez], xs


def check(side, V, e_y, e_z, off, yaw_k):
    A, R, xs = frame(side, e_y, e_z, off, yaw_k)
    P = V @ R.T + A
    r_ax = np.hypot(V[:, 1], V[:, 2])
    outside_hole = ~((r_ax < R_SOCKET) & (V[:, 0] > X_SOCKET))
    lev = (((P - HC) / (HA - SKIN_IN)) ** 2).sum(1)
    worst = lev[outside_hole].max()
    under_board = (np.abs(P[:, 0]) < 7.0) & (P[:, 1] > HC[1] - 8.2) & (P[:, 1] < HC[1] + 10.4)
    zc = ZB0 - 0.55 - 0.3 - P[under_board, 2].max() if under_board.any() else 99.0   # MCU 0.55 under the board
    return worst, zc, A, xs


if __name__ == "__main__":
    V = {s: load(s) for s in ("tx", "rx")}
    for s in V:
        print(s, "verts", len(V[s]))
    best = []
    for e_y in np.arange(-48.5, -51.6, -0.5):
        for e_z in np.arange(80.0, 82.6, 0.5):
            for off in (-1.2, -1.0, -0.6, -0.3, 0.0):
                for yk in (0.0, 0.5, 1.0):
                    r = [check(s, V[s], e_y, e_z, off, yk) for s in ("tx", "rx")]
                    w = max(r[0][0], r[1][0])
                    best.append((w, e_y, e_z, off, yk, r[0][0], r[1][0], r[0][1], r[1][1]))
    best.sort()
    print("worst_level  E_Y   E_Z  off yaw_k | tx_lev rx_lev | tx_board_clr rx_board_clr")
    for b in best[:25]:
        print("%.3f  %6.1f %5.1f %4.2f %3.1f | %.3f %.3f | %6.2f %6.2f" % b)
    if len(sys.argv) > 1:
        e_y, e_z, off, yk = map(float, sys.argv[1:5])
        for s in ("tx", "rx"):
            print(s, check(s, V[s], e_y, e_z, off, yk))
