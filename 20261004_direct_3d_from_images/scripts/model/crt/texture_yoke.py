"""Photo textures for the yoke tape surface, the copper ring side and the round neck parts (holder, clamp, rings,
cap, glass neck), sampled on the model's own lofts/cylinders.

Why not texture_from_photos.py cylinder mode: the yoke is elliptical (ry about 24, rz about 20) and tapered, so a
circular cylinder puts texels up to about 3 mm off the surface (parallax ghosting between views). This script
builds the texel grid from config/model/crt.toml exactly as scripts/model/crt/build.py builds the mesh, and uses
the ID renders of a run (all train views) for visibility: a texel is used from a view only if that view's ID pass
shows the target object at the texel's pixel (occluders like wires, tabs, PCB parts are skipped).

Usage: uv run python scripts/model/crt/texture_yoke.py --idrun outputs/runs/<run with id pass for train views>
       [--ppm 8] [--k 1] [--surfaces yoke,copper]
Outputs: assets/textures/crt_<surface>.png (u = angle from the bottom seam (t = -90 deg) toward +Y, v top = -X end)
and outputs/textures/crt_<surface>_views.jpg (texture | coverage | best-view index).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from common import OUTPUTS  # noqa: E402
from tools.geom import load_views  # noqa: E402
from tools.texture_from_photos import full_res_cams, sample_view  # noqa: E402

ASYM = (1.0, 1.0)  # yoke/copper asymmetry (sy_neg, sz_bot), set by yoke_rings(); same as build._ell
T0 = -math.pi / 2  # texture seam at the bottom of the loft (never seen)
# Color gates per surface (OpenCV HSV: H 0-180): a view's texel is used only if its color is plausible for the
# surface (rejects occluders the model does not have yet and misregistered edges); rejected texels are inpainted.
GATES_1C = {"yoke": dict(h=(10, 32), s=120, v=60), "copper": dict(h=(0, 32), s=90, v=30)}
# Phase 3 (2026-10-05): gates off by default. Self-texture test in IMG_1624: yoke 19.3 dB with the gate vs 29.9 dB
# without (the gate drops highlights, shadows and folds); probe crt_025: yoke 18.4 -> 19.2 dB. --gate re-enables.
GATES = {}


def yoke_rings(P):
    """Same ring list and axis as build.py (mm)."""
    Y = P["yoke"]
    sy, sz = Y.get("scale_y", 1.0), Y.get("scale_z", 1.0)
    ks = [Y.get(f"s{i}", 1.0) for i in range(len(Y["rings"]))]
    rings = [(x, ry * sy * k, rz * sz * k) for (x, ry, rz), k in zip(Y["rings"], ks)]
    global ASYM
    ASYM = (Y.get("sy_neg", 1.0), Y.get("sz_bot", 1.0))
    return rings, P["axis"]["y"] + Y.get("dy", 0.0), P["axis"]["z"] + Y.get("dz", 0.0), sy, sz


def surfaces(P):
    rings, yay, az, sy, sz = yoke_rings(P)
    c = P["copper"]
    ay, az0 = P["axis"]["y"], P["axis"]["z"]
    cyl = {}
    for key, obj in (("holder", "crt.yoke_holder"), ("clamp", "crt.neck_clamp"), ("white_ring", "crt.neck_white_ring"),
                     ("cream_ring", "crt.neck_ring"), ("cap", "crt.socket_cap"), ("neck", "crt.neck")):
        q = P[key]
        cyl[key] = (obj, [(q["x0"], q["r"], q["r"]), (q["x1"], q["r"], q["r"])], ay + q.get("dy", 0.0),
                    az0 + q.get("dz", 0.0))
    g = P["black_ring"]
    cyl["dark_ring"] = ("crt.yoke_ring", [(g["x0"], g["r"], g["r"]), (g["x1"], g["r"], g["r"])], ay, az0)
    b = P["board"]  # +X face of the socket board: plane, TL/TR/BR/BL seen from +X (right = +Y)
    cyl["board"] = ("crt.socket_board_face", {"plane": [[b["x1"], b["y0"], b["z1"]], [b["x1"], b["y1"], b["z1"]],
                                                   [b["x1"], b["y1"], b["z0"]], [b["x1"], b["y0"], b["z0"]]]}, 0, 0)
    cyl["label"] = ("crt.label_yoke", {"label": P["label"], "rings": rings, "asym": ASYM}, yay, az)
    return cyl | {
        "yoke": ("crt.yoke", {"rings": rings, "asym": ASYM}, yay, az),
        "copper": ("crt.yoke_copper", {"rings": [(c["x0"], c["ry"] * sy, c["rz"] * sz),
                                                 (c["x1"], c["ry"] * sy, c["rz"] * sz)], "asym": ASYM}, yay, az),
    }


def interp(rings, x):
    xs = np.array([r[0] for r in rings])
    return np.interp(x, xs, [r[1] for r in rings]), np.interp(x, xs, [r[2] for r in rings])


def plane_grid(C, ppm):
    C = np.array(C, float)
    w = max(2, int(round(np.linalg.norm(C[1] - C[0]) * ppm)))
    h = max(2, int(round(np.linalg.norm(C[3] - C[0]) * ppm)))
    U, Vv = np.meshgrid((np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h)
    top = C[0] * (1 - U[..., None]) + C[1] * U[..., None]
    bot = C[3] * (1 - U[..., None]) + C[2] * U[..., None]
    X = top * (1 - Vv[..., None]) + bot * Vv[..., None]
    n = np.cross(C[3] - C[0], C[1] - C[0])
    n /= np.linalg.norm(n)
    return X / 1e3, np.broadcast_to(n, X.shape).copy()


def grid(rings, ay, az, ppm):
    """Texel world points (h, w, 3) in meters and outward normals."""
    if isinstance(rings, dict) and "plane" in rings:
        return plane_grid(rings["plane"], ppm)
    asym = (1.0, 1.0)
    if isinstance(rings, dict) and "label" not in rings:
        rings, asym = rings["rings"], rings["asym"]
    if isinstance(rings, dict):  # label patch on the yoke ellipse, same parametrization as build._label_patch
        L, rr, asym = rings["label"], rings["rings"], rings["asym"]
        w = int(round((L["y1"] - L["y0"]) * ppm * 1.15))
        h = int(round((L["x1"] - L["x0"]) * ppm))
        xs = L["x0"] + (L["x1"] - L["x0"]) * (np.arange(h) + 0.5) / h
        ry, rz = interp(rr, xs)
        ry, rz = ry + L["offset"], rz + L["offset"]
        t0 = np.arccos(np.clip((L["y0"] - ay) / (ry * (asym[0] if L["y0"] < ay else 1.0)), -1, 1))
        t1 = np.arccos(np.clip((L["y1"] - ay) / (ry * (asym[0] if L["y1"] < ay else 1.0)), -1, 1))
        f = (np.arange(w) + 0.5) / w
        T = t0[:, None] + (t1 - t0)[:, None] * f[None, :]
        Xg = np.repeat(xs[:, None], w, 1)
        kc = np.where(np.cos(T) < 0, asym[0], 1.0)
        Pp = np.stack([Xg, ay + ry[:, None] * kc * np.cos(T), az + rz[:, None] * np.sin(T)], -1)
        Nn = np.stack([np.zeros_like(T), np.cos(T) / (ry[:, None] * kc), np.sin(T) / rz[:, None]], -1)
        Nn /= np.linalg.norm(Nn, axis=-1, keepdims=True)
        return Pp / 1e3, Nn
    x0, x1 = rings[0][0], rings[-1][0]
    rmax = max(max(r[1], r[2]) for r in rings)
    w = int(round(2 * math.pi * rmax * ppm))
    h = max(2, int(round((x1 - x0) * ppm)))
    t = T0 + 2 * math.pi * (np.arange(w) + 0.5) / w
    x = x0 + (x1 - x0) * (np.arange(h) + 0.5) / h
    T, Xg = np.meshgrid(t, x)

    def pt(xx, tt):
        ry, rz = interp(rings, xx)
        c, s_ = np.cos(tt), np.sin(tt)
        return np.stack([xx, ay + ry * c * np.where(c < 0, asym[0], 1.0),
                         az + rz * s_ * np.where(s_ < 0, asym[1], 1.0)], -1)

    P = pt(Xg, T)
    dt = pt(Xg, T + 1e-3) - pt(Xg, T - 1e-3)
    dx = pt(Xg + 1e-2, T) - pt(Xg - 1e-2, T)
    N = np.cross(dt, dx)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    radial = P - np.stack([Xg, np.full_like(Xg, ay), np.full_like(Xg, az)], -1)
    N *= np.sign((N * radial).sum(-1, keepdims=True))
    return P / 1e3, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idrun", required=True)
    ap.add_argument("--ppm", type=float, default=8.0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--surfaces", default="yoke,copper,holder,clamp,white_ring,cream_ring,cap,neck,dark_ring,board")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--gate", action="store_true", help="1C HSV color gates on yoke/copper")
    ap.add_argument("--no-gain", action="store_true")
    ap.add_argument("--min-cos", type=float, default=0.25)
    ap.add_argument("--views", default="", help="comma list: restrict source views (diagnostics)")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--erode", type=int, default=2, help="erode the ID ownership mask (s4 px)")
    a = ap.parse_args()
    P = tomllib.loads((ROOT / "config" / "model" / "crt.toml").read_text())
    idrun = ROOT / a.idrun
    id_map = json.loads((idrun / "id_map.json").read_text())
    V4 = load_views(4)
    intr, img_dir = full_res_cams()
    split = json.loads((ROOT / "data" / "split.json").read_text())
    names = [n for n in split["train"] if (idrun / "id" / f"{n}.png").exists()]
    if a.views:
        names = [n for n in a.views.split(",") if (idrun / "id" / f"{n}.png").exists()]
    for sname in a.surfaces.split(","):
        obj, rings, ay, az = surfaces(P)[sname]
        X, N = grid(rings, ay, az, a.ppm)
        h, w = X.shape[:2]
        if obj not in id_map and obj == "crt.socket_board_face":
            obj = "crt.socket_board"  # before the face quad existed, the box owned its +X face
        key = np.array(id_map[obj][::-1])  # BGR
        cols, scores, used = [], [], []
        for nm in names:
            v = V4[nm]
            idm = cv2.imread(str(idrun / "id" / f"{nm}.png"), cv2.IMREAD_UNCHANGED)
            own = ((np.abs(idm[..., :3].astype(int) - key).sum(2) <= 3) & (idm[..., 3] > 0)).astype(np.uint8)
            if own.sum() < 200:
                continue
            if a.erode:
                own = cv2.erode(own, np.ones((2 * a.erode + 1, 2 * a.erode + 1), np.uint8))
            uv, z = v.project(X.reshape(-1, 3))
            uu = np.round(uv[:, 0]).astype(int)
            vv = np.round(uv[:, 1]).astype(int)
            inside = (uu >= 0) & (vv >= 0) & (uu < own.shape[1]) & (vv < own.shape[0]) & (z > 0)
            vis = np.zeros(len(uu), bool)
            vis[inside] = own[vv[inside], uu[inside]] > 0
            vis = vis.reshape(h, w)
            d = v.center[None, None] - X
            dist = np.linalg.norm(d, axis=2)
            cosang = (d * N).sum(2) / dist
            sc = (cosang * intr[v.camera_id][0] / (dist * 1e3)).astype(np.float32)
            sc[~vis | (cosang < a.min_cos)] = -1
            if (sc > 0).sum() < 0.002 * h * w:
                continue
            t, inb = sample_view(v, intr[v.camera_id], img_dir, X)
            sc[~inb] = -1
            g = GATES_1C.get(sname) if a.gate else GATES.get(sname)
            if g and not a.no_gate:
                hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
                ok = (hsv[..., 0] >= g["h"][0]) & (hsv[..., 0] <= g["h"][1]) & (hsv[..., 1] >= g["s"]) & \
                     (hsv[..., 2] >= g["v"])
                sc[~ok] = -1
            cols.append(t)
            scores.append(sc)
            used.append(nm)
        if not cols:
            print(json.dumps({"surface": sname, "error": "no view sees it"}))
            continue
        C = np.stack(cols)
        S = np.stack(scores)
        if not a.no_gain:  # per-view exposure gain toward a robust 5-view reference (removes seams)
            o5 = np.argsort(-S, axis=0)[:5]
            C5 = np.take_along_axis(C, o5[..., None], 0).astype(np.float32)
            C5[np.take_along_axis(S, o5, 0) <= 0] = np.nan
            with np.errstate(all="ignore"):
                ref = np.nanmedian(C5, axis=0)
            del C5
            okr = ~np.isnan(ref).any(2)
            for j in range(len(C)):
                mj = (S[j] > 0) & okr
                if mj.sum() < 200:
                    continue
                g = np.clip(np.median(ref[mj], 0) / np.maximum(np.median(C[j][mj].astype(np.float32), 0), 1), 0.6, 1.6)
                C[j] = np.clip(C[j] * g, 0, 255).astype(np.uint8)
        order = np.argsort(-S, axis=0)[: a.k]
        Ck = np.take_along_axis(C, order[..., None], 0).astype(np.float32)
        Sk = np.take_along_axis(S, order, 0)
        Ck[Sk <= 0] = np.nan
        with np.errstate(all="ignore"):
            blend = np.nanmedian(Ck, axis=0)
        hole = np.isnan(blend).any(2)
        blend = np.nan_to_num(blend, nan=0).astype(np.uint8)
        if hole.any():  # large holes (never seen): median surface color; borders: short inpaint blend
            med = np.median(blend[~hole], axis=0)
            fill = blend.copy()
            fill[hole] = med
            near = hole & (cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 3) < 6)
            inp = cv2.inpaint(blend, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)
            fill[near] = inp[near]
            soft = cv2.GaussianBlur(fill, (0, 0), 3)
            blend = np.where(hole[..., None], soft, blend)
        out = ROOT / "assets" / "textures" / f"crt_{sname}{a.out_suffix}.png"
        cv2.imwrite(str(out), blend)
        best = order[0].astype(np.float32)
        best[Sk[0] <= 0] = -1
        bv = cv2.applyColorMap(((best + 1) * 255 / (len(used) + 1)).astype(np.uint8), cv2.COLORMAP_JET)
        cov = cv2.cvtColor(np.clip((Sk > 0).sum(0) * 255 // a.k, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        (OUTPUTS / "textures").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUTPUTS / "textures" / f"crt_{sname}{a.out_suffix}_views.jpg"), np.vstack([blend, cov, bv]))
        print(json.dumps({"surface": sname, "texture": str(out.relative_to(ROOT)), "size_px": [h, w],
                          "n_views": len(used), "hole_frac": round(float(hole.mean()), 3),
                          "views_used_best": sorted({used[i] for i in np.unique(order[0][Sk[0] > 0])})}))


if __name__ == "__main__":
    main()
