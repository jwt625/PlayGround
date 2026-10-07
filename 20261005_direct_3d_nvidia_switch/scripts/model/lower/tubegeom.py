"""Tube geometry shared by build.py (Blender) and bake_lower.py (uv): centripetal Catmull-Rom centerline through
the measured points, parallel-transport frames, and the (along x around) surface grid that the tube mesh and its
texture use. Pure numpy; units mm. Texture convention: column = along the tube (start -> end), row = angle
(0 at the top of the image), angle 0 = frame n1."""

from __future__ import annotations

import numpy as np


def catmull_rom(pts, step=1.5):
    P = np.asarray(pts, float)
    if len(P) < 2:
        return P
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** 0.5 + 1e-9
        t2 = t1 + np.linalg.norm(p2 - p1) ** 0.5 + 1e-9
        t3 = t2 + np.linalg.norm(p3 - p2) ** 0.5 + 1e-9
        n = max(2, int(np.ceil(np.linalg.norm(p2 - p1) / step)))
        for t in np.linspace(t1, t2, n, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(P[-1])
    return np.array(out)


def frames(C):
    """Unit tangents T and parallel-transported normals N1, N2 along centerline C (n, 3)."""
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n1 = ref - (ref @ T[0]) * T[0]
    n1 /= np.linalg.norm(n1)
    N1 = [n1]
    for i in range(1, len(C)):
        v = N1[-1] - (N1[-1] @ T[i]) * T[i]
        N1.append(v / np.linalg.norm(v))
    N1 = np.array(N1)
    N2 = np.cross(T, N1)
    return T, N1, N2


def centerline(P: dict, key: str, step=1.5):
    """Smoothed centerline for a cable entry of lower.toml (braid_b gets its fitted front-part offsets)."""
    e = P[key]
    pts = []
    for x, y, z in e["pts"]:
        if key == "braid_b":
            w = min(1.0, max(0.0, (345.0 - y) / 55.0))
            x = x + w * e.get("dx", 0.0)
            z = z + w * (e.get("dz", 0.0) + e.get("dz_slope", 0.0) * (y - 200.0) / 100.0)
            ky = [125.0, 200.0, 275.0]  # piecewise-linear knot offsets (fit_params keys kx125.., kz125..)
            x = x + w * float(np.interp(y, ky, [e.get(f"kx{int(k)}", 0.0) for k in ky]))
            z = z + w * float(np.interp(y, ky, [e.get(f"kz{int(k)}", 0.0) for k in ky]))
        if key == "braid_a" and any(k.startswith("m") for k in e):  # Wave 4: middle-bay knot offsets mx/mz<Y>
            ky = sorted({int(k[2:]) for k in e if k[:2] in ("mx", "mz")})
            kx = [e.get(f"mx{k}", 0.0) for k in ky]
            kz = [e.get(f"mz{k}", 0.0) for k in ky]
            ky2 = [ky[0] - 25.0] + [float(k) for k in ky] + [ky[-1] + 25.0]  # taper to 0 over 25 mm at both ends
            x = x + float(np.interp(y, ky2, [0.0] + kx + [0.0]))
            z = z + float(np.interp(y, ky2, [0.0] + kz + [0.0]))
        pts.append((x, y, z))
    return catmull_rom(pts, step)


def surface(C, r, n_around):
    """Ring points X (n_around + 1, n, 3) and outward normals; row k = angle 2 pi k / n_around (seam repeated)."""
    T, N1, N2 = frames(C)
    a = 2 * np.pi * np.arange(n_around + 1) / n_around
    Nrm = np.cos(a)[:, None, None] * N1[None] + np.sin(a)[:, None, None] * N2[None]
    return C[None] + r * Nrm, Nrm


def arclen(C):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]
