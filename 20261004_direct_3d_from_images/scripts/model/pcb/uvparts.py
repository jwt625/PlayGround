"""Analytic surface patches + texture-atlas layout for pcb components (shared by build.py inside Blender and by
bake_parts.py outside). numpy only, no bpy.

A part = list of primitives (box or cylinder) -> list of patches. Each patch maps (s, t) in [0, 1]^2 (s = image
column, t = image row, t = 0 at the top of the patch image) to world points (mm) and outward normals. Patches are
shelf-packed into one atlas per part at PPM texels per mm; build.py makes the mesh from the same patches, so the
baked image and the mesh UVs agree by construction.
"""

from __future__ import annotations

import math

import numpy as np

PPM = 6.0  # texels per mm
PAD = 2  # texels between patches


def _rot(rx, ry, rz):
    """Same order as lib.place: Rz @ Ry @ Rx (degrees)."""
    a, b, c = (math.radians(v) for v in (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def box(center, half, R=None, skip_bottom=True):
    R = np.eye(3) if R is None else np.asarray(R, float)
    return {"type": "box", "c": np.asarray(center, float), "h": np.asarray(half, float), "R": R,
            "skip_bottom": skip_bottom}


def cyl(base, axis, r, length, ref=None, cap_top=True):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    if ref is None:
        ref = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    r0 = np.asarray(ref, float) - (np.asarray(ref, float) @ a) * a
    r0 /= np.linalg.norm(r0)
    return {"type": "cyl", "b": np.asarray(base, float), "a": a, "r0": r0, "r1": np.cross(a, r0), "r": float(r),
            "L": float(length), "cap_top": cap_top}


def patches(prims):
    """List of patch dicts: name, w, h (texels), fn(s, t) -> (P (..., 3) mm, N (..., 3)), seg (mesh segments)."""
    out = []
    for k, p in enumerate(prims):
        if p["type"] == "box":
            c, h, R = p["c"], p["h"], p["R"]
            for ax in range(3):
                for sg in (1, -1):
                    if ax == 2 and sg == -1 and p["skip_bottom"]:
                        continue
                    ia, ib = [i for i in range(3) if i != ax]
                    n = R[:, ax] * sg
                    A, B = R[:, ia] * (sg if ax != 1 else -sg), R[:, ib]
                    ha, hb = h[ia], h[ib]

                    def fn(s, t, c=c, n=n, A=A, B=B, ha=ha, hb=hb, hn=h[ax]):
                        s = np.asarray(s, float)[..., None]
                        t = np.asarray(t, float)[..., None]
                        P = c + n * hn + A * ha * (2 * s - 1) + B * hb * (1 - 2 * t)
                        return P, np.broadcast_to(n, P.shape)

                    out.append({"name": f"{k}_box{ax}{'p' if sg > 0 else 'm'}", "fn": fn,
                                "w": max(2, round(2 * ha * PPM)), "h": max(2, round(2 * hb * PPM)), "seg": (1, 1),
                                "smooth": False})
        else:
            b, a, r0, r1, r, L = p["b"], p["a"], p["r0"], p["r1"], p["r"], p["L"]

            def side(s, t, b=b, a=a, r0=r0, r1=r1, r=r, L=L):
                th = 2 * np.pi * np.asarray(s, float)[..., None]
                t = np.asarray(t, float)[..., None]
                N = np.cos(th) * r0 + np.sin(th) * r1
                return b + a * L * (1 - t) + r * N, N

            out.append({"name": f"{k}_side", "fn": side, "w": max(8, round(2 * np.pi * r * PPM)),
                        "h": max(2, round(L * PPM)), "seg": (32, 1), "smooth": True})
            if p["cap_top"]:
                def top(s, t, b=b, a=a, r0=r0, r1=r1, r=r, L=L):
                    th = 2 * np.pi * np.asarray(s, float)[..., None]
                    rho = r * (1 - 0.98 * np.asarray(t, float)[..., None])
                    P = b + a * L + rho * (np.cos(th) * r0 + np.sin(th) * r1)
                    return P, np.broadcast_to(a, P.shape)

                out.append({"name": f"{k}_top", "fn": top, "w": max(8, round(np.pi * r * PPM)),
                            "h": max(2, round(r * PPM)), "seg": (32, 3), "smooth": False})
    return out


def layout(pl):
    """Shelf packing; adds x0, y0 to each patch, returns atlas (W, H)."""
    area = sum((p["w"] + PAD) * (p["h"] + PAD) for p in pl)
    W = max(max(p["w"] for p in pl) + PAD, int(math.ceil(math.sqrt(area) * 1.2)))
    x = y = shelf = 0
    for p in sorted(pl, key=lambda q: (-q["h"], q["name"])):
        if x + p["w"] + PAD > W:
            x, y, shelf = 0, y + shelf, 0
        p["x0"], p["y0"] = x, y
        x += p["w"] + PAD
        shelf = max(shelf, p["h"] + PAD)
    return W, y + shelf


def parts_from_params(P):
    """Textured-part specs from config/model/pcb.toml: dict(name, mat, prims, owners_prefix)."""
    zb = P["board"]["top"]
    F = P.get("fit", {})

    def off(n):
        return F.get(f"{n}_dx", 0.0), F.get(f"{n}_dy", 0.0)

    out = []
    for blk in P.get("block", []):
        x0, x1, y0, y1 = blk["xy"]
        dx, dy = off(blk["name"])
        out.append({"name": blk["name"], "mat": blk["mat"],
                    "prims": [box(((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, (zb + blk["top"]) / 2),
                                  ((x1 - x0) / 2, (y1 - y0) / 2, (blk["top"] - zb) / 2))]})
    for sl in P.get("slab", []):
        R = _rot(sl.get("rot_x_deg", 0.0), 0.0, sl.get("rot_z_deg", 0.0))
        out.append({"name": sl["name"], "mat": sl["mat"],
                    "prims": [box(sl["center"], np.asarray(sl["size"], float) / 2, R, skip_bottom=False)]})
    t = P["trimmer"]
    s2 = t["side"] / 2
    for i, (x, y) in enumerate(t["xy"]):
        dx, dy = off(f"trim_{i}")
        out.append({"name": f"trim_{i}", "mat": "trim_base",
                    "prims": [box((x + dx, y + dy, (zb + t["rotor_top"]) / 2), (s2, s2, (t["rotor_top"] - zb) / 2))]})
    for c in P.get("cap", []):
        dx, dy = off(c["name"])
        out.append({"name": c["name"], "mat": c["mat"],
                    "prims": [cyl((c["x"] + dx, c["y"] + dy, zb), (0, 0, 1), c["d"] / 2, c["top"] - zb)]})
    for a in P.get("axial", []):
        dx, dy = off(a["name"])
        cen = np.array(a["center"], float) + [dx, dy, F.get(f"{a['name']}_dz", 0.0)]
        ax = {"x": (1.0, 0, 0), "y": (0, 1.0, 0), "z": (0, 0, 1.0)}[a["axis"]]
        out.append({"name": a["name"], "mat": a["mat"], "leads": a.get("lead_len", 0.0), "axis": a["axis"],
                    "center": cen.tolist(),
                    "prims": [cyl(cen - np.array(ax) * a["len"] / 2, ax, a["r"], a["len"], ref=(0, 0, 1.0))]})
    f = P["flyback"]
    out.append({"name": "flyback", "mat": f.get("body_mat", "beige"),
                "prims": [box(((f["x0"] + f["x1"]) / 2, (f["y0"] + f["y1"]) / 2, (zb + f["base_h"] + f["top"]) / 2),
                              ((f["x1"] - f["x0"]) / 2, (f["y1"] - f["y0"]) / 2, (f["top"] - zb - f["base_h"]) / 2))]})
    out.append({"name": "flyback_base", "mat": "black",
                "prims": [box(((f["x0"] + f["x1"]) / 2, (f["y0"] + f["y1"]) / 2, zb + f["base_h"] / 2),
                              ((f["x1"] - f["x0"]) / 2, (f["y1"] - f["y0"]) / 2, f["base_h"] / 2))]})
    b = P["board"]  # board as one box; top texture per-texel over all train views (replaces label_board_top)
    out.append({"name": "board", "mat": "board", "owners": ["pcb.label_board_top"], "remove": ["pcb.label_board_top"],
                "prims": [box(((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, zb - b["thick"] / 2),
                              ((b["x1"] - b["x0"]) / 2, (b["y1"] - b["y0"]) / 2, b["thick"] / 2))]})
    pt = P["plate"]  # heat-sink plate as one box (vertical plate + top flange; chamfer dropped when textured)
    y0 = pt.get("flange_y0", pt["y0"])
    out.append({"name": "heatsink_plate", "mat": "metal",
                "owners": ["pcb.label_plate_top", "pcb.label_plate_face"],
                "remove": ["pcb.label_plate_top", "pcb.label_plate_face"],
                "prims": [box(((pt["x0"] + pt["x1"]) / 2, (y0 + pt["y1"]) / 2, (zb + pt["top"]) / 2),
                              ((pt["x1"] - pt["x0"]) / 2, (pt["y1"] - y0) / 2, (pt["top"] - zb) / 2))]})
    return out


def texture_path(name):
    return f"assets/textures/pcb_parts/{name}.png"
