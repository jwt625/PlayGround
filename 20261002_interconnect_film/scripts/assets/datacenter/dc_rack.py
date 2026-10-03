"""Rack hardware generators shared by rack_nvl72_style, rack_pair_for_cable_gag, datahall_environment, tray_fiber_pullout.

Rack local frame: x centre, y centre of footprint (-y front), z=0 floor. Dimensions in mm unless noted (sources in DevLog/meta).
"""
import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
sys.path.insert(0, HERE)
import common as C  # noqa: E402
import dc_common as D  # noqa: E402
import dc_parts as P  # noqa: E402
from dc_common import MB, mm  # noqa: E402

RACK_W, RACK_D, RACK_H = 600.0, 1068.0, 2236.0   # NVL72 / MGX ORV3 external (Supermicro, OCP listings): B
BASE_H = 160.0                                    # casters + base frame (estimate C)
OPEN_W = 537.0                                    # ORV3 21-inch opening
OU_MM = 48.0                                      # OpenU pitch
FRONT_Y = -505.0                                  # tray front plane (estimate)
SPINE_Y0 = 340.0                                  # cartridge bay front plane (tray rear connector tips)
N_OU = 42                                         # usable OU between base frame and top frame (2036/48)


def ou_z(n):
    """Bottom z (mm) of OU n (1-based)."""
    return BASE_H + (n - 1) * OU_MM


def rack_frame_parts(L, with_manifold=True, n_trays_z=(5, 31)):
    """Static rack hardware: base, casters, posts, rails, busbar, manifolds + QD heads, cartridge bay frames. Returns list[P.Part]."""
    parts = []
    hw, hd = RACK_W / 2, RACK_D / 2
    # --- frame (black powder coat)
    fr = MB()
    # base frame perimeter
    fr.boxmm((0, -hd + 18, BASE_H - 30), (RACK_W - 8, 36, 60), 0)
    fr.boxmm((0, hd - 18, BASE_H - 30), (RACK_W - 8, 36, 60), 0)
    for s in (-1, 1):
        fr.boxmm((s * (hw - 18), 0, BASE_H - 30), (36, RACK_D - 80, 60), 0)
    fr.boxmm((0, 0, BASE_H - 10), (RACK_W - 80, 80, 20), 0)
    # corner posts
    for sx in (-1, 1):
        for sy in (-1, 1):
            fr.boxmm((sx * (hw - 14), sy * (hd - 20), (BASE_H + RACK_H - 40) / 2), (28, 40, RACK_H - 40 - BASE_H + 0), 0)
    # top frame
    fr.boxmm((0, -hd + 18, RACK_H - 20), (RACK_W, 36, 40), 0)
    fr.boxmm((0, hd - 18, RACK_H - 20), (RACK_W, 36, 40), 0)
    for s in (-1, 1):
        fr.boxmm((s * (hw - 14), 0, RACK_H - 20), (28, RACK_D - 72, 40), 0)
    # inner side rail plates carrying the tray slides (x = +-272.5), and front/rear uprights
    z0, z1 = BASE_H, RACK_H - 40
    ytr = FRONT_Y + P.TRAY_D
    for s in (-1, 1):
        for yy in (FRONT_Y + 20, FRONT_Y + P.TRAY_D / 2, ytr - 20):
            fr.boxmm((s * (OPEN_W / 2 + 4), yy, (z0 + z1) / 2), (8, 40, z1 - z0), 0)
        fr.boxmm((s * (OPEN_W / 2 + 12), FRONT_Y - 5, (z0 + z1) / 2), (16, 12, z1 - z0), 0)
        fr.boxmm((s * (OPEN_W / 2 + 12), 0.5 * (FRONT_Y - 5 + ytr), z1 - 10), (16, ytr - FRONT_Y, 20), 0)
        fr.boxmm((s * (OPEN_W / 2 + 12), 0.5 * (FRONT_Y - 5 + ytr), z0 + 10), (16, ytr - FRONT_Y, 20), 0)
    parts.append(P.Part("frame", fr, [L["frame"]], bevel=(mm(1.2), 1)))
    # --- casters (rubber wheel mat0, fork steel mat1) and leveling feet
    cs = MB()
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * 250, sy * 440
            cs.cyl((x * 0.001, y * 0.001, 0.045), 0.045, 0.028, "x", 20, 0)
            cs.boxmm((x - 17, y, 80), (4, 36, 60), 1)
            cs.boxmm((x + 17, y, 80), (4, 36, 60), 1)
            cs.boxmm((x, y, 108), (50, 60, 6), 1)
    parts.append(P.Part("casters", cs, [L["hose"], L["dark_steel"]], smooth=False))
    ft = MB()
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = sx * 280, sy * 480
            ft.cyl((x * 0.001, y * 0.001, 0.075), 0.012, 0.09, "z", 12, 0)
            ft.cyl((x * 0.001, y * 0.001, 0.0225), 0.028, 0.012, "z", 16, 0)
    parts.append(P.Part("leveling_feet", ft, [L["nickel"]], smooth=True))
    # --- busbar with insulating shroud (rear centre)
    zb0, zb1 = BASE_H + 30, RACK_H - 60
    bb = MB()
    for dx in (-13, 13):
        bb.boxmm((dx, 352, (zb0 + zb1) / 2), (22, 28, zb1 - zb0), 0)
    parts.append(P.Part("busbar", bb, [L["copper"]], bevel=(mm(1.0), 1)))
    sh = MB()
    for s in (-1, 1):
        sh.boxmm((s * 38, 358, (zb0 + zb1) / 2), (5, 44, zb1 - zb0 + 20), 0)
    sh.boxmm((0, 381, (zb0 + zb1) / 2), (81, 4, zb1 - zb0 + 20), 0)
    sh.boxmm((0, 352, zb0 - 6), (81, 44, 8), 0)
    sh.boxmm((0, 352, zb1 + 6), (81, 44, 8), 0)
    parts.append(P.Part("busbar_shroud", sh, [L["black"]], bevel=(mm(1.0), 1)))
    # --- coolant manifolds + QD heads
    if with_manifold:
        for s, nm, key in ((-1, "supply", "qd_blue"), (1, "return", "qd_red")):
            mf = MB()
            x = s * 222
            mf.cyl((x * 0.001, 0.405, (BASE_H + 20 + RACK_H - 70) / 2 * 0.001), 0.022, (RACK_H - 90 - BASE_H) * 0.001, "z", 20, 0)
            mf.boxmm((x, 405, BASE_H + 20), (60, 60, 8), 1)
            mf.boxmm((x, 405, RACK_H - 70), (60, 60, 8), 1)
            # rack-level inlet/outlet stub at the bottom rear
            mf.cyl((x * 0.001, (405 + 60) * 0.001, (BASE_H + 60) * 0.001), 0.020, 0.12, "y", 16, 0)
            mf.cyl((x * 0.001, (405 + 128) * 0.001, (BASE_H + 60) * 0.001), 0.027, 0.022, "y", 16, 1)
            parts.append(P.Part("manifold_" + nm, mf, [L["nickel"], L[key]], smooth=True))
        # QD heads on the manifold, one per tray OU (27 trays), built per OU in the rack script via tray_qd_heads()
    # --- cartridge bay frames (empty; cartridge wall asset mounts here): x in [35,185] on each side of the busbar, y 340..460
    bz0, bz1 = ou_z(n_trays_z[0]), ou_z(n_trays_z[1] + 1)
    for s, nm in ((-1, "L"), (1, "R")):
        bf = MB()
        xc = s * 110
        for dx in (-75, 75):
            bf.boxmm((xc + dx, 400, (bz0 + bz1) / 2), (6, 120, bz1 - bz0 + 30), 0)
        bf.boxmm((xc, 400, bz0 - 12), (156, 120, 6), 0)
        bf.boxmm((xc, 400, bz1 + 12), (156, 120, 6), 0)
        bf.boxmm((xc, 458, (bz0 + bz1) / 2), (150, 3, bz1 - bz0), 1)   # mounting backplate (empty bay)
        parts.append(P.Part("cartridge_bay_" + nm, bf, [L["dark_steel"], L["panel"]], bevel=(mm(0.8), 1)))
    return parts


def tray_qd_heads(L, ou_list):
    """Female QD heads + stubs for each tray OU: returns (supply_mb, return_mb) with colour ring mat1."""
    out = []
    for s in (-1, 1):
        mb = MB()
        x = s * 222
        for n in ou_list:
            zc = ou_z(n) + 24.0
            mb.cyl((x * 0.001, 0.3665, zc * 0.001), 0.0115, 0.041, "y", 14, 0)
            mb.cyl((x * 0.001, 0.351, zc * 0.001), 0.0135, 0.006, "y", 14, 1)
            mb.cyl((x * 0.001, 0.3905, zc * 0.001), 0.0075, 0.030, "y", 10, 0)   # stub to manifold
        out.append(mb)
    return out


def power_shelf_parts(L, name="ps"):
    """1 OU power shelf (33 kW class, 6 PSU modules): front modules with handles, LEDs, vents; rear copper blades."""
    W2 = P.TRAY_BODY_W / 2
    H = P.TRAY_H
    Dm = (SPINE_Y0 - 5) - FRONT_Y   # flush front, rear at the busbar
    parts = []
    bd = MB()
    bd.boxmm((0, 0, H / 2), (P.TRAY_BODY_W, Dm - 4, H - 2))
    parts.append(P.Part("body", bd, [L["steel"]], bevel=(mm(0.5), 1)))
    fp = MB()
    mod = P.TRAY_BODY_W / 6
    fp.boxmm((0, -Dm / 2 - 1.5, H / 2), (P.TRAY_BODY_W + 12, 3, H - 1.5), 0)
    parts.append(P.Part("front_panel", fp, [L["tray_front"]], bevel=(mm(0.6), 2)))
    pm = D.perforated_material("perforated_front", (0.09, 0.095, 0.1), 0.8, 0.4, pitch=0.0042, radius=0.0014,
                               scale_axes=(1, 0, 1))
    vt = MB()
    hd = MB()
    lg = MB()
    for i in range(6):
        x = -W2 + mod * (i + 0.5)
        vt.boxmm((x - 8, -Dm / 2 - 3.35, H / 2), (mod - 34, 0.7, H - 12))
        pts = [(x + mod / 2 - 14, -Dm / 2 - 3.0, 8), (x + mod / 2 - 14, -Dm / 2 - 14, 11),
               (x + mod / 2 - 14, -Dm / 2 - 14, H - 11), (x + mod / 2 - 14, -Dm / 2 - 3.0, H - 8)]
        hd.tube([tuple(a * 0.001 for a in p) for p in pts], 0.0024, 8, 0)
        lg.boxmm((x + mod / 2 - 24, -Dm / 2 - 3.4, H - 9), (3, 0.8, 2.5), 0)
        fp.boxmm((x + mod / 2, -Dm / 2 - 3.1, H / 2), (0.8, 0.6, H - 4), 0)  # module seams
    parts[-1] = P.Part("front_panel", fp, [L["tray_front"]], bevel=(mm(0.6), 2))
    parts.append(P.Part("vents", vt, [pm]))
    parts.append(P.Part("handles", hd, [L["alu"]], smooth=True))
    parts.append(P.Part("leds", lg, [L["led_g"]]))
    rr = MB()
    for dx in (-13, 13):
        rr.boxmm((dx, Dm / 2 + 5, H / 2), (22, 10, 30), 0)
    parts.append(P.Part("rear_blades", rr, [L["copper"]], bevel=(mm(0.5), 1)))
    return parts, Dm


def blank_panel_parts(L):
    H = P.TRAY_H
    Dm = 120.0
    mb = MB()
    mb.boxmm((0, 0, H / 2), (P.TRAY_BODY_W + 10, 3, H - 1.5), 0)
    return [P.Part("panel", mb, [L["panel"]], bevel=(mm(0.6), 1))], Dm


def door_parts(L, hinge_side=-1, name="door"):
    """Perforated rack door (596 x 2030 mm), hinge at local origin on the hinge edge; opens about z."""
    Wd, Hd = 596.0, 2036.0
    parts = []
    xs = -hinge_side  # door extends away from hinge
    cx = xs * Wd / 2
    fr = MB()
    for dx in (-Wd / 2 + 12, Wd / 2 - 12):
        fr.boxmm((cx + dx, 0, Hd / 2), (24, 24, Hd))
    for dz in (12, Hd - 12):
        fr.boxmm((cx, 0, dz), (Wd - 48, 24, 24))
    parts.append(P.Part("frame", fr, [L["frame"]], bevel=(mm(1.0), 1)))
    pm = D.perforated_material("perforated_door", (0.045, 0.047, 0.052), 0.7, 0.45, pitch=0.0072, radius=0.0027,
                               scale_axes=(1, 0, 1))
    pl = MB()
    pl.boxmm((cx, 0, Hd / 2), (Wd - 44, 2.5, Hd - 44))
    parts.append(P.Part("perf_panel", pl, [pm]))
    hd = MB()
    hx = cx - xs * (Wd / 2 - 36)
    hd.boxmm((hx, -22, Hd * 0.52), (14, 8, 160), 0)
    hd.boxmm((hx, -14, Hd * 0.52 + 70), (14, 14, 12), 0)
    hd.boxmm((hx, -14, Hd * 0.52 - 70), (14, 14, 12), 0)
    parts.append(P.Part("handle", hd, [L["alu"]], bevel=(mm(1.5), 2)))
    return parts


def side_panel_parts(L):
    mb = MB()
    zc = (BASE_H + RACK_H - 40) / 2
    for s in (-1, 1):
        mb.boxmm((s * (RACK_W / 2 - 1.0), 0, zc), (2.0, RACK_D - 40, RACK_H - 40 - BASE_H - 6), 0)
    return [P.Part("panels", mb, [L["panel"]], bevel=(mm(0.5), 1))]


def rack_stub(coll, root, prefix, L, loc=(0, 0, 0), rot=(0, 0, 0), door=True, solid=True, cache_key="stub"):
    """Simplified rack (body + perforated front/rear doors + top/casters), shared mesh data via cache_key.

    Returns the stub empty. Used for rows in the data hall and for the cable-gag / fiber-tray racks.
    """
    e = bpy.data.objects.new(prefix, None)
    e.empty_display_type = "PLAIN_AXES"
    e.empty_display_size = 0.1
    coll.objects.link(e)
    e.parent = root
    e.location = loc
    e.rotation_euler = rot
    mb = MB()
    hw, hd = RACK_W / 2, RACK_D / 2
    # sides, top, bottom, back (solid carcass)
    mb.boxmm((-hw + 1, 0, (BASE_H + RACK_H) / 2), (2, RACK_D, RACK_H - BASE_H), 0)
    mb.boxmm((hw - 1, 0, (BASE_H + RACK_H) / 2), (2, RACK_D, RACK_H - BASE_H), 0)
    mb.boxmm((0, 0, RACK_H - 1), (RACK_W, RACK_D, 2), 0)
    mb.boxmm((0, 0, BASE_H + 1), (RACK_W, RACK_D, 2), 0)
    mb.boxmm((0, hd - 1, (BASE_H + RACK_H) / 2), (RACK_W, 2, RACK_H - BASE_H), 0)
    mb.boxmm((0, -hd + 1, (BASE_H + RACK_H) / 2), (RACK_W, 2, RACK_H - BASE_H), 0)  # hidden behind the door
    mb.boxmm((0, 0, BASE_H / 2), (RACK_W - 20, RACK_D - 20, BASE_H - 20), 1)
    parts = [P.Part("body", mb, [L["frame"], L["dark_steel"]], bevel=(mm(1.5), 1))]
    cs = MB()
    for sx in (-1, 1):
        for sy in (-1, 1):
            cs.cyl((sx * 0.25, sy * 0.44, 0.04), 0.04, 0.026, "x", 16, 0)
    parts.append(P.Part("casters", cs, [L["hose"]]))
    if door:
        fd = MB()
        pm = D.perforated_material("perforated_door", (0.045, 0.047, 0.052), 0.7, 0.45, pitch=0.0072, radius=0.0027,
                                   scale_axes=(1, 0, 1))
        fd.boxmm((0, -hd - 2.5, (BASE_H + RACK_H) / 2), (RACK_W - 4, 2.5, RACK_H - BASE_H - 6), 0)
        parts.append(P.Part("door_front", fd, [pm]))
        hd_ = MB()
        hd_.boxmm((-hw + 36, -hd - 14, 1150), (14, 8, 160), 0)
        parts.append(P.Part("door_handle", hd_, [L["alu"]], bevel=(mm(1.5), 2)))
    P.instantiate(parts, coll, e, prefix, key=cache_key)
    return e
