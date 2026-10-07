"""Photo-textured surfaces of the chassis group (pure Python: used by build.py inside Blender and by bake.py).

Each surface is a quad in the chassis LOCAL frame (before the roll correction), corners TL, TR, BR, BL in image
orientation; normal = (down) x (right) points toward the viewer. The texture file is
assets/textures/chassis_<name>.png; owners are the model objects whose ID pixels may be sampled.
Parameters for baking (sessions, statistic, px_per_mm) live in config/model/chassis.toml [tex.<name>].
"""

from __future__ import annotations

import math


def roll_to_world(p, roll_deg: float):
    """Local (mm) -> world (mm): optional extra roll about Y (lib.place(rot_deg=(0, roll_deg, 0)); 0 in use),
    then the shared "tray" frame of config/frames.json (as lib.apply_frame(objs, "tray"))."""
    t = math.radians(roll_deg)
    x, y, z = p
    x, z = x * math.cos(t) + z * math.sin(t), -x * math.sin(t) + z * math.cos(t)
    fr = _tray_frame()
    R, tt = fr["R"], fr["t_mm"]
    return tuple(R[i][0] * x + R[i][1] * y + R[i][2] * z + tt[i] for i in range(3))


_FR: dict = {}


def _tray_frame() -> dict:
    if not _FR:
        import json
        from pathlib import Path
        root = next(q for q in Path(__file__).resolve().parents if (q / "config" / "frames.json").exists())
        _FR.update(json.loads((root / "config" / "frames.json").read_text())["frames"]["tray"])
    return _FR


def surfaces(P: dict) -> dict:
    b, f, c, w = P["body"], P["front"], P["crossbar"], P["wall"]
    X, t, zb, zr, yr = b["x_outer"], b["t"], b["z_bot"], b["z_rim"], b["y_rear"]
    yf = f["y_face"]
    xi = X - f["bezel_inset_x"]
    du = P["ducts_front"]
    S = {}
    # crossbar top (Z = 0), seen from +Z: right = +X, down = -Y
    zc = c.get("z_top", 0.0)
    er = c.get("edge_r", 0.0)  # textured flat top between the rounded edges
    S["crossbar_top"] = dict(corners=[(-c["x_half"], c["y1"] - er, zc), (c["x_half"], c["y1"] - er, zc),
                                      (c["x_half"], c["y0"] + er, zc), (-c["x_half"], c["y0"] + er, zc)],
                             owners=["chassis.crossbar_top"])
    # front port plate face (Y = yf) between the grip top and the lip, facing -Y: right = +X, down = -Z
    zpb = f.get("plate_tex_z0", f["bezel_z1"])  # lower edge of the textured panel face
    S["front_plate"] = dict(corners=[(-X, yf, f["z_top"]), (X, yf, f["z_top"]), (X, yf, zpb),
                                     (-X, yf, zpb)], owners=["chassis.front_plate"])
    m = f["mmc"]
    for gi, xc in enumerate(m["x_centers"]):
        y = yf - m["depth"]
        S[f"mmc_g{gi}"] = dict(corners=[(xc - m["w"] / 2, y, m["z1"]), (xc + m["w"] / 2, y, m["z1"]),
                                        (xc + m["w"] / 2, y, m["z0"]), (xc - m["w"] / 2, y, m["z0"])],
                               owners=[f"chassis.mmc_g{gi}"])
    # diagnostic strip over the whole MMC row (bake only; [tex.mmc_strip] skip = true keeps it out of the build)
    ym = yf - m["depth"]
    S["mmc_strip"] = dict(corners=[(-X, ym, m["z1"] + 4), (X, ym, m["z1"] + 4), (X, ym, m["z0"] - 4), (-X, ym, m["z0"] - 4)],
                          owners=[f"chassis.mmc_g{i}" for i in range(len(m["x_centers"]))] + ["chassis.front_plate",
                          "chassis.front_grip"] + [f"chassis.tex_mmc_g{i}" for i in range(len(m["x_centers"]))] +
                          ["chassis.tex_front_plate", "chassis.tex_wing"])
    # grip front (Y = bezel_y0) and grip top (Z = bezel_z1, seen from +Z)
    if "nose" in f["grip"]:  # planar nose facet of the wing profile, facing forward (-Y) and slightly up
        (yt, zt), (yb_, zbn) = f["grip"]["nose"]
        S["grip_front"] = dict(corners=[(-xi, yt, zt), (xi, yt, zt), (xi, yb_, zbn), (-xi, yb_, zbn)],
                               owners=["chassis.front_grip"])
    else:
        S["grip_front"] = dict(corners=[(-xi, f["bezel_y0"], f["bezel_z1"]), (xi, f["bezel_y0"], f["bezel_z1"]),
                                        (xi, f["bezel_y0"], f["grip_z0"]), (-xi, f["bezel_y0"], f["grip_z0"])],
                               owners=["chassis.front_grip"])
    S["grip_top"] = dict(corners=[(-xi, yf, f["bezel_z1"]), (xi, yf, f["bezel_z1"]),
                                  (xi, f["bezel_y0"], f["bezel_z1"]), (-xi, f["bezel_y0"], f["bezel_z1"])],
                         owners=["chassis.front_grip"])
    (ya, za), (yb, zb_) = f["grip"]["slope"]
    S["wing"] = dict(corners=[(xi, ya, za), (-xi, ya, za), (-xi, yb, zb_), (xi, yb, zb_)],
                     owners=["chassis.front_grip"])  # seen from the front/top: right = -X, normal up and back
    S["lip_top"] = dict(corners=[(-X, yf + f["lip_d"], f["z_top"]), (X, yf + f["lip_d"], f["z_top"]), (X, yf, f["z_top"]),
                                 (-X, yf, f["z_top"])], owners=["chassis.front_lip", "chassis.front_plate"])
    # side walls: outer faces (normal -+X) and inner faces
    S["wall_mx_out"] = dict(corners=[(-X, yr, zr), (-X, w["y0"], zr), (-X, w["y0"], zb), (-X, yr, zb)],
                            owners=["chassis.wall_mx"])
    S["wall_px_out"] = dict(corners=[(X, w["y0"], zr), (X, yr, zr), (X, yr, zb), (X, w["y0"], zb)],
                            owners=["chassis.wall_px"])
    S["wall_mx_in"] = dict(corners=[(-X + t, w["y0"], zr - t), (-X + t, yr, zr - t), (-X + t, yr, zb + t),
                                    (-X + t, w["y0"], zb + t)], owners=["chassis.wall_mx"])
    S["wall_px_in"] = dict(corners=[(X - t, yr, zr - t), (X - t, w["y0"], zr - t), (X - t, w["y0"], zb + t),
                                    (X - t, yr, zb + t)], owners=["chassis.wall_px"])
    # rim folds (top faces of the wall flanges), seen from +Z: right = +X, down = -Y
    fw = w["flange_w"]
    S["rim_mx"] = dict(corners=[(-X, yr, zr), (-X + fw, yr, zr), (-X + fw, w["y0"], zr), (-X, w["y0"], zr)],
                       owners=["chassis.wall_mx_flange", "chassis.wall_mx"])
    S["rim_px"] = dict(corners=[(X - fw, yr, zr), (X, yr, zr), (X, w["y0"], zr), (X - fw, w["y0"], zr)],
                       owners=["chassis.wall_px_flange", "chassis.wall_px"])
    # crossbar flanges: front face (normal -Y) and rear face (normal +Y)
    if c.get("y_ext", 0.0) > c["y1"]:  # rear ledge top, seen from +Z
        ze = c.get("z_ext", zc)
        S["crossbar_ext"] = dict(corners=[(-c["x_half"], c["y_ext"], ze), (c["x_half"], c["y_ext"], ze),
                                          (c["x_half"], c["y1"], ze), (-c["x_half"], c["y1"], ze)],
                                 owners=["chassis.crossbar_ext"])
    if c.get("y_ext_front", c["y0"]) < c["y0"]:  # front ledge top, seen from +Z
        zf_ = c.get("z_ext_front", zc)
        S["crossbar_ext_front"] = dict(corners=[(-c["x_half"], c["y0"], zf_), (c["x_half"], c["y0"], zf_),
                                                (c["x_half"], c["y_ext_front"], zf_), (-c["x_half"], c["y_ext_front"], zf_)],
                                       owners=["chassis.crossbar_ext_front"])
    if er > 0 and int(c.get("edge_n", 8)) == 1:  # 45 deg chamfers of the crossbar long edges (planar, textured)
        xh = c["x_half"]
        S["crossbar_ch_front"] = dict(corners=[(-xh, c["y0"] + er, zc), (xh, c["y0"] + er, zc), (xh, c["y0"], zc - er),
                                               (-xh, c["y0"], zc - er)], owners=["chassis.crossbar_fl_front"])
        S["crossbar_ch_rear"] = dict(corners=[(xh, c["y1"] - er, zc), (-xh, c["y1"] - er, zc), (-xh, c["y1"], zc - er),
                                              (xh, c["y1"], zc - er)], owners=["chassis.crossbar_fl_rear"])
    fd = c["flange_depth"]
    yff = min(c.get("y_ext_front", c["y0"]), c["y0"])
    S["crossbar_fl_front"] = dict(corners=[(-c["x_half"], yff, zc), (c["x_half"], yff, zc),
                                           (c["x_half"], yff, -fd), (-c["x_half"], yff, -fd)],
                                  owners=["chassis.crossbar_fl_front"])
    yfr = max(c.get("y_ext", 0.0), c["y1"])
    S["crossbar_fl_rear"] = dict(corners=[(c["x_half"], yfr, zc), (-c["x_half"], yfr, zc),
                                          (-c["x_half"], yfr, -fd), (c["x_half"], yfr, -fd)],
                                 owners=["chassis.crossbar_fl_rear"])
    # divider (both faces) and rear wall inner face
    d = P["divider"]
    xd = X - t
    zdt = d["z_top"] - (t if "flange_y0" in d else 0.0)
    S["divider_front"] = dict(corners=[(-xd, d["y"], zdt), (xd, d["y"], zdt), (xd, d["y"], zb + t),
                                       (-xd, d["y"], zb + t)], owners=["chassis.divider"])
    S["divider_rear"] = dict(corners=[(xd, d["y"] + t, d["z_top"]), (-xd, d["y"] + t, d["z_top"]),
                                      (-xd, d["y"] + t, zb + t), (xd, d["y"] + t, zb + t)], owners=["chassis.divider"])
    if "flange_y0" in d:  # top flange, seen from +Z (rounded corners: corner texels are never owned)
        fx, zt = d["flange_x"], d["z_top"]
        S["divider_top"] = dict(corners=[(-fx, d["flange_y1"], zt), (fx, d["flange_y1"], zt), (fx, d["flange_y0"], zt),
                                         (-fx, d["flange_y0"], zt)], owners=["chassis.divider_flange"])
    zrt = P["rear"]["z_top"]
    S["rear_in"] = dict(corners=[(-xd, yr - t, zrt), (xd, yr - t, zrt), (xd, yr - t, zb + t), (-xd, yr - t, zb + t)],
                        owners=["chassis.rear_wall"])
    S["rear_out"] = dict(corners=[(X, yr, zrt), (-X, yr, zrt), (-X, yr, zb), (X, yr, zb)],
                         owners=["chassis.rear_wall"])  # outer face (normal +Y), seen from behind: right = -X
    # floor pan top face (Z = zb + t) per bay, seen from +Z: right = +X, down = -Y
    zf = zb + t
    for nm, y0, y1 in (("floor_front", yf + f["plate_t"], c["y0"]), ("floor_mid", c["y1"], d["y"]),
                       ("floor_rear", d["y"] + t, yr - t)):
        z_ = zf + P.get("tex", {}).get(nm, {}).get("dz", 0.0)  # dz: textured plane above the floor top (sweeps)
        S[nm] = dict(corners=[(-xd, y1, z_), (xd, y1, z_), (xd, y0, z_), (-xd, y0, z_)], owners=["chassis.floor"])
    # black duct tops (Z = du.z_top), seen from +Z
    for s, nm in ((-1, "duct_front_mx"), (1, "duct_front_px")):
        x0, x1 = sorted((s * du["x_in"], s * du["x_out"]))
        S[nm + "_top"] = dict(corners=[(x0, du["y1"], du["z_top"]), (x1, du["y1"], du["z_top"]),
                                       (x1, du["y0"], du["z_top"]), (x0, du["y0"], du["z_top"])],
                              owners=[f"chassis.{nm}"])
    # duct inner side faces (X = -+x_in, facing the bay center): right = +Y on the -X duct, -Y on the +X duct
    zd0 = zb + t
    S["duct_front_mx_in"] = dict(corners=[(-du["x_in"], du["y0"], du["z_top"]), (-du["x_in"], du["y1"], du["z_top"]),
                                          (-du["x_in"], du["y1"], zd0), (-du["x_in"], du["y0"], zd0)],
                                 owners=["chassis.duct_front_mx"])
    S["duct_front_px_in"] = dict(corners=[(du["x_in"], du["y1"], du["z_top"]), (du["x_in"], du["y0"], du["z_top"]),
                                          (du["x_in"], du["y0"], zd0), (du["x_in"], du["y1"], zd0)],
                                 owners=["chassis.duct_front_px"])
    sh = P.get("shelf_px", {})
    if sh.get("enabled", False):  # shelf top, seen from +Z
        x1s, zs = X - t, sh["z_top"]
        S["shelf_px_top"] = dict(corners=[(sh["x0"], sh["y1"], zs), (x1s, sh["y1"], zs), (x1s, sh["y0"], zs),
                                          (sh["x0"], sh["y0"], zs)], owners=["chassis.shelf_px"])
    rr = P.get("rail_rear_px", {})
    if rr.get("enabled", False):
        zr_ = rr["z_top"]
        S["rail_rear_px_top"] = dict(corners=[(rr["x0"], rr["y1"], zr_), (rr["x1"], rr["y1"], zr_), (rr["x1"], rr["y0"], zr_),
                                              (rr["x0"], rr["y0"], zr_)], owners=["chassis.rail_rear_px"])
    for k, v in S.items():
        v["owners"] = v["owners"] + [f"chassis.tex_{k}"]
    return S
