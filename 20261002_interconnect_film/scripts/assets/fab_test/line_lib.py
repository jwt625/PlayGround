"""Shared helpers for the line assets (reflow oven, conveyor modules, WIP carrier)."""
import math
import ft

MM = ft.MM


def make_board(A, prefix, cx, cy, z0, size=(300, 220), parent=None, coll=None, pkg=True):
    """Substrate board: green PCB with a black BGA/ASIC package, gold fiducials, MLCCs and a connector. Returns list of objects."""
    sx, sy = size
    objs = []
    objs.append(A.box(prefix + "_pcb", (sx, sy, 1.6), (cx, cy, z0), "pcb_green", anchor="b", bev=0.3, parent=parent, coll=coll))
    if pkg:
        objs.append(A.box(prefix + "_package_substrate", (70, 70, 1.2), (cx, cy, z0 + 1.6), "pcb_blue", anchor="b", bev=0.3, parent=parent, coll=coll))
        objs.append(A.box(prefix + "_package_lid", (46, 46, 3.5), (cx, cy, z0 + 2.8), "black_plastic", anchor="b", bev=1.0, parent=parent, coll=coll))
        for sgx in (-1, 1):
            for sgy in (-1, 1):
                objs.append(A.box(prefix + "_hbm_proxy", (16, 24, 2.2), (cx + sgx * 28, cy + sgy * 6, z0 + 2.8), "black_plastic", anchor="b", bev=0.5, parent=parent, coll=coll))
    mlcc = A.box(prefix + "_mlcc_proto", (1.0, 0.5, 0.5), (cx, cy, z0 + 1.6), "ceramic", anchor="b", parent=parent, coll=coll)
    mlcc.hide_render = mlcc.hide_viewport = True
    for k in range(24):
        a = 2 * math.pi * k / 24
        A.dup(mlcc, prefix + "_mlcc", (cx + 62 * math.cos(a), cy + 62 * math.sin(a), z0 + 1.6 + 0.25), rot_z=a, parent=parent, coll=coll)
    for k in range(4):
        objs.append(A.cyl(prefix + "_fiducial", 1.2, 0.05, (cx + (sx / 2 - 8) * (1 if k % 2 else -1), cy + (sy / 2 - 8) * (1 if k // 2 else -1), z0 + 1.6),
                          "gold", anchor="b", seg=16, parent=parent, coll=coll))
    objs.append(A.box(prefix + "_edge_connector", (60, 8, 6), (cx, cy + sy / 2 - 10, z0 + 1.6), "black_plastic", anchor="b", bev=0.5, parent=parent, coll=coll))
    return objs


def belt_conveyor(A, tag, x0, x1, y, z, width=600, rail_h=40, parent=None, coll=None, legs=True, leg_h=None):
    """Belt conveyor segment from x0 to x1 (mm) at belt-top height z centred on y: frame, two rails, belt, rollers, legs."""
    L = x1 - x0
    xc = (x0 + x1) / 2
    objs = []
    objs.append(A.box(tag + "_belt", (L, width - 40, 3), (xc, y, z - 3), "mesh_belt", anchor="b", parent=parent, coll=coll))
    for sg in (-1, 1):
        objs.append(A.box(tag + "_rail", (L, 25, rail_h), (xc, y + sg * (width / 2 + 5), z - 25), "steel", anchor="b", bev=1.5, parent=parent, coll=coll))
        objs.append(A.box(tag + "_guide", (L, 6, 22), (xc, y + sg * (width / 2 - 18), z), "silver", anchor="b", bev=1, parent=parent, coll=coll))
    objs.append(A.box(tag + "_frame_beam", (L, width - 60, 60), (xc, y, z - 90), "steel_dark", anchor="b", bev=2, parent=parent, coll=coll))
    if legs:
        lh = leg_h or (z - 90)
        for xx in (x0 + 80, x1 - 80):
            for sg in (-1, 1):
                objs.append(A.box(tag + "_leg", (50, 50, lh), (xx, y + sg * (width / 2 - 30), 0), "steel_dark", anchor="b", bev=2, parent=parent, coll=coll))
    return objs
