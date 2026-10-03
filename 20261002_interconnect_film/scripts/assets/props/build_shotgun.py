"""Double-barrel break-action 12 gauge shotgun, two variants (clay gag, dark wood + steel), with break-open rig.

Run: Blender -b --python scripts/assets/props/build_shotgun.py -- assets/components/props

Axes: barrels point along -Y (front), Z up. Origin = stock wrist (HOOK_grip_R), the point the assembler parents to the hand.
All build coordinates are in mm; Z0 shifts the wrist centre to z = 0.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
import props_lib as L  # noqa: E402
from props_lib import C, mm  # noqa: E402

OUT, PREV = L.outdirs()
Z0 = 37.0  # wrist centre height in the build frame
AXIS_Z = 57.0  # barrel pair axis height (build frame)
Y_BREECH = -170.0  # standing breech face
BARREL_LEN = 710.0  # 28 in
HINGE = (0.0, -240.0, 32.0)


def P(x, y, z):
    return Vector((mm(x), mm(y), mm(z - Z0)))


def interp(pts, x):
    for k in range(len(pts) - 1):
        if pts[k][0] <= x <= pts[k + 1][0]:
            t = (x - pts[k][0]) / (pts[k + 1][0] - pts[k][0])
            return pts[k][1] * (1 - t) + pts[k + 1][1] * t
    return pts[-1][1] if x > pts[-1][0] else pts[0][1]


# barrel outer radius along the barrel (u = distance from breech face, mm). ESTIMATE: typical SxS 12 ga OD 27 mm at the chamber, 22 mm at the muzzle.
R_OUT = [(0, 13.5), (80, 12.9), (300, 11.9), (710, 11.0)]
R_HOLLOW = [(0, 10.75), (68, 10.75), (80, 9.8), (710, 9.8)]  # wall inner radius (tube liner embeds into it)
R_BORE = [(0, 10.2), (68, 10.2), (80, 9.25), (710, 9.25)]  # chamber 20.4 mm, bore 18.5 mm (12 ga nominal bore 18.5 mm)


def barrel_stations(side, chunk):
    """side = -1 (left, -x) or +1; returns stations (center, r_out, r_hollow, r_bore) along -Y."""
    us = [0, 4, 30, 68, 80, 150, 300, 450, 600, 690, 706, 710]
    st = []
    for u in us:
        r = interp(R_OUT, u) * chunk
        xc = side * (interp(R_OUT, u) - 0.8) * (1.0 + (chunk - 1.0) * 0.0) * chunk
        rr = r
        if u >= 706:
            rr = r - (0.0 if u == 706 else 0.7)
        st.append((u, Vector((mm(xc), mm(Y_BREECH - u), mm(AXIS_Z - Z0))), mm(rr), mm(interp(R_HOLLOW, u)), mm(interp(R_BORE, u))))
    return st


def hollow_tube(name, stations, which, mat, n=36, end_inset=0.0):
    """Hollow tube along -Y. which = 'barrel' (outer r_out, inner r_hollow) or 'bore' (outer r_hollow+0.15, inner r_bore)."""
    bm = bmesh.new()
    outer, inner = [], []
    ux, vz = Vector((1, 0, 0)), Vector((0, 0, 1))
    for u, c, ro, rh, rb in stations:
        if which == "barrel":
            a, b = ro, rh
        else:
            a, b = rh + mm(0.15), rb
        cc = c.copy()
        if which == "bore" and u >= 706:
            cc.y += mm(end_inset) * 1  # recess liner end behind the muzzle face
        outer.append([bm.verts.new(L.ring_frame(cc, ux, vz, L.circle2d(a, n))[i]) for i in range(n)])
        inner.append([bm.verts.new(L.ring_frame(cc, ux, vz, L.circle2d(b, n))[i]) for i in range(n)])
    for k in range(len(stations) - 1):
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((outer[k][i], outer[k][j], outer[k + 1][j], outer[k + 1][i]))
            bm.faces.new((inner[k][j], inner[k][i], inner[k + 1][i], inner[k + 1][j]))
    for k in (0, len(stations) - 1):
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((outer[k][i], outer[k][j], inner[k][j], inner[k][i]))
    return L.obj_from_bm(name, bm, mat, True, True, 40)


def rib_obj(name, mat, chunk):
    bm = bmesh.new()
    rings = []
    us = [0, 40, 150, 300, 450, 600, 700]
    ux, vz = Vector((1, 0, 0)), Vector((0, 0, 1))
    for u in us:
        ro = interp(R_OUT, u) * chunk
        zt = AXIS_Z + ro - 0.5
        zb = AXIS_Z + 2.0
        pts = [(-4, zb), (4, zb), (4.3, zt - 0.9), (3.2, zt), (-3.2, zt), (-4.3, zt - 0.9)]
        pts = [(mm(a), mm(b - Z0)) for a, b in pts]
        rings.append(L.ring_frame(Vector((0, mm(Y_BREECH - u), 0)), ux, vz, pts))
    L.loft_bm(bm, rings, True, True)
    return L.obj_from_bm(name, bm, mat, True, True, 40)


def path_obj(name, pts_yz, prof_wh, mat, x=0.0, n_per=6, taper=None, close_cap=True):
    """Sweep a rounded-rect profile (w along x, h normal to the path) along a y-z polyline (mm, build frame)."""
    pts = [Vector((mm(x), mm(y), mm(z - Z0))) for y, z in pts_yz]
    pts = L.catmull(pts, n_per)
    w, h = prof_wh
    prof = [(mm(a), mm(b)) for a, b in L.superellipse2d(w, h, 12, 2.6)]
    bm = bmesh.new()
    L.tube_path_bm(bm, pts, prof, binormal=Vector((1, 0, 0)), cap=close_cap, scale_fn=taper)
    return L.obj_from_bm(name, bm, mat, True, True, 50)


def stock_curves():
    ys = [-90, -40, 0, 60, 130, 200, 252, 270]
    top = [68, 65, 60, 52, 40, 28, 19, 15]
    bot = [-8, 8, 14, 8, -20, -72, -112, -125]
    wid = [40, 34, 30, 32, 36, 40, 43, 44]
    return ys, top, bot, wid


def stock_obj(name, mat, y0, y1, grow=0.0, p=3.0, nst=40):
    ys, top, bot, wid = stock_curves()
    X = [y0 + (y1 - y0) * i / (nst - 1) for i in range(nst)]
    T = L.smooth_curve(ys, top, 80)
    B = L.smooth_curve(ys, bot, 80)
    W = L.smooth_curve(ys, wid, 80)

    def at(curve, x):
        return interp(list(zip(curve[0], curve[1])), x)
    bm = bmesh.new()
    ux, vz = Vector((1, 0, 0)), Vector((0, 0, 1))
    rings = []
    # path runs toward +Y: u x v = +Y needs u = z, v = x  (z x x = +y)
    for y in X:
        t, b, w = at(T, y), at(B, y), at(W, y)
        pts = L.superellipse2d(w + grow, t - b + grow, 24, p)
        # (a, b) -> (z, x): a is the vertical half-axis dir in u = Z, b is lateral in v = X
        pts2 = [(mm((z + (t + b) / 2) - Z0), mm(x)) for x, z in pts]
        rings.append(L.ring_frame(Vector((0, mm(y), 0)), Vector((0, 0, 1)), Vector((1, 0, 0)), pts2))
    L.loft_bm(bm, rings, True, True)
    return L.obj_from_bm(name, bm, mat, True, True, 40)


def forend_obj(name, mat, chunk):
    y0, y1 = -258.0, -482.0
    bm = bmesh.new()
    rings = []
    ux, vz = Vector((1, 0, 0)), Vector((0, 0, 1))
    n = 14
    for i in range(n):
        y = y0 + (y1 - y0) * i / (n - 1)
        t = i / (n - 1)
        w = 50 - 7 * t
        top = 45.5
        bot = 9 + 6 * t
        pts = L.superellipse2d(w * (1 + (chunk - 1) * 0.6), top - bot, 24, 2.8)
        pts = [(mm(a), mm(b + (top + bot) / 2 - Z0)) for a, b in pts]
        if i == n - 1:
            pts = [(a * 0.93, (b - mm(top - Z0)) * 0.93 + mm(top - Z0)) for a, b in pts]
        rings.append(L.ring_frame(Vector((0, mm(y), 0)), ux, vz, pts))
    L.loft_bm(bm, rings, True, True)
    return L.obj_from_bm(name, bm, mat, True, True, 40)


def hammer_obj(name, side, mat, chunk):
    # cocked pose: spur rises up and back (+y)
    path = [(0, 0), (7, 11), (13, 21), (21, 28), (28, 29)]
    pivot = (side * 28.5, -108.0, 50.0)
    pts = [Vector((mm(pivot[0]), mm(pivot[1] + dy), mm(pivot[2] + dz - Z0))) for dy, dz in path]
    pts = L.catmull(pts, 6)
    prof = [(mm(a), mm(b)) for a, b in L.superellipse2d(7.0 * chunk, 9.5 * chunk, 10, 2.4)]
    bm = bmesh.new()
    L.tube_path_bm(bm, pts, prof, binormal=Vector((1, 0, 0)), cap=True, scale_fn=lambda t: 1.0 - 0.45 * t)
    r, hw = mm(5.5 * chunk), mm(3.0)
    xf = Matrix.Translation(P(*pivot)) @ Matrix.Rotation(math.pi / 2, 4, "Y")
    L.lathe_bm(bm, [(0, -hw), (r, -hw), (r, hw), (0, hw)], 20, xf=xf)
    o = L.obj_from_bm(name, bm, mat, True, True, 50)
    return o, pivot


def build():
    L.C.reset()
    scn = bpy.context.scene
    coll, root = C.new_asset("shotgun", accuracy="B")
    root.empty_display_size = 0.05

    # ------------------------------------------------------------ rig
    bones = [
        dict(name="body", head=P(0, 0, Z0), tail=P(0, 0, Z0 + 15), roll_axis=(0, -1, 0)),
        dict(name="break", head=P(*HINGE), tail=P(HINGE[0], HINGE[1], HINGE[2] + 12), parent="body", roll_axis=(0, -1, 0)),
        dict(name="top_lever", head=P(0, -112, 71.5), tail=P(12, -112, 71.5), parent="body", roll_axis=(0, 0, 1)),
        dict(name="hammer_L", head=P(-28.5, -108, 50), tail=P(-28.5, -108, 62), parent="body", roll_axis=(0, -1, 0)),
        dict(name="hammer_R", head=P(28.5, -108, 50), tail=P(28.5, -108, 62), parent="body", roll_axis=(0, -1, 0)),
        dict(name="trigger_R", head=P(0, -82, -8), tail=P(0, -82, 4), parent="body", roll_axis=(0, -1, 0)),
        dict(name="trigger_F", head=P(0, -108, -8), tail=P(0, -108, 4), parent="body", roll_axis=(0, -1, 0)),
    ]
    arm = L.make_armature("shotgun_rig", bones, coll, root)
    L.add_prop(root, "p_break", 0.0, 0.0, 1.0, "0 closed, 1 fully broken open (barrels tip down about 28 deg about HOOK_break, local +X)")
    L.add_prop(root, "p_hammer_cock_L", 1.0, 0.0, 1.0, "1 cocked (modelled pose), 0 dropped (tilted forward 40 deg)")
    L.add_prop(root, "p_hammer_cock_R", 1.0, 0.0, 1.0, "1 cocked (modelled pose), 0 dropped")
    L.add_prop(root, "p_trigger", 0.0, 0.0, 1.0, "0 rest, 1 both triggers pulled back about 17 deg")
    L.add_prop(root, "p_top_lever", 0.0, 0.0, 1.0, "top lever swing (also follows p_break): 0 closed, 1 thrown right 30 deg")
    pb = arm.pose.bones
    L.driver(pb["break"], "rotation_euler", 0, root, ["p_break"], "v0*0.49")
    L.driver(pb["top_lever"], "rotation_euler", 2, root, ["p_top_lever", "p_break"], "-0.52*max(v0, v1)")
    L.driver(pb["hammer_L"], "rotation_euler", 0, root, ["p_hammer_cock_L"], "(1-v0)*0.70")
    L.driver(pb["hammer_R"], "rotation_euler", 0, root, ["p_hammer_cock_R"], "(1-v0)*0.70")
    L.driver(pb["trigger_R"], "rotation_euler", 0, root, ["p_trigger"], "v0*0.30")
    L.driver(pb["trigger_F"], "rotation_euler", 0, root, ["p_trigger"], "v0*0.30")

    # ------------------------------------------------------------ hooks
    hooks = {}

    def mk_hook(name, loc, rot=(0, 0, 0), bone="body", size=0.03):
        h = L.hook_at(name, coll, root, loc, rot, size)
        L.bone_parent(h, arm, bone)
        hooks[name] = h
        return h
    mk_hook("break", P(*HINGE), size=0.04)
    xm = interp(R_OUT, 710) - 0.8
    ymuz = Y_BREECH - BARREL_LEN
    mk_hook("muzzle_L", P(-xm, ymuz, AXIS_Z), rot=(math.pi / 2, 0, 0), bone="break")
    mk_hook("muzzle_R", P(xm, ymuz, AXIS_Z), rot=(math.pi / 2, 0, 0), bone="break")
    xb = interp(R_OUT, 0) - 0.8
    mk_hook("chamber_L", P(-xb, Y_BREECH - 2, AXIS_Z), rot=(math.pi / 2, 0, 0), bone="break", size=0.015)
    mk_hook("chamber_R", P(xb, Y_BREECH - 2, AXIS_Z), rot=(math.pi / 2, 0, 0), bone="break", size=0.015)
    mk_hook("grip_R", P(0, 0, Z0), size=0.04)
    mk_hook("foregrip_L", P(0, -370, 6), bone="break", size=0.04)
    mk_hook("shoulder", P(0, 268, -55), size=0.04)
    mk_hook("trigger_finger", P(0, -108, -22), bone="trigger_F", size=0.02)

    # ------------------------------------------------------------ materials
    variants = {}

    def mats_clay():
        return dict(
            wood=L.clay("MAT_props_clay_stock", (0.52, 0.30, 0.15)),
            steel=L.clay("MAT_props_clay_barrel", (0.26, 0.30, 0.38)),
            recv=L.clay("MAT_props_clay_receiver", (0.46, 0.47, 0.50)),
            dark=L.clay("MAT_props_clay_dark", (0.16, 0.16, 0.18)),
            bore=C.principled("MAT_props_clay_bore", (0.02, 0.02, 0.025), rough=0.9),
            pad=L.clay("MAT_props_clay_pad", (0.10, 0.10, 0.11)),
            brass=L.clay("MAT_props_clay_brass", (0.85, 0.65, 0.20)),
        )

    def mats_wood():
        return dict(
            wood=L.wood_mat("MAT_props_wood_walnut"),
            steel=C.principled("MAT_props_wood_blued_steel", (0.035, 0.04, 0.05), metallic=1.0, rough=0.26),
            recv=C.principled("MAT_props_wood_receiver", (0.30, 0.31, 0.33), metallic=1.0, rough=0.34),
            dark=C.principled("MAT_props_wood_trigger", (0.05, 0.05, 0.055), metallic=1.0, rough=0.4),
            bore=C.principled("MAT_props_wood_bore", (0.015, 0.015, 0.02), metallic=1.0, rough=0.35),
            pad=C.principled("MAT_props_wood_recoil_pad", (0.02, 0.02, 0.02), rough=0.9),
            brass=C.principled("MAT_props_wood_brass", (0.8, 0.6, 0.2), metallic=1.0, rough=0.3),
        )

    def build_variant(vname, M, chunk, p_stock, tg=1.0):
        vc = L.C.sub_collection(coll, "VARIANT_" + vname)
        parts = []

        def put(o, bone=None):
            C.add(o, vc, root)
            L.bone_parent(o, arm, bone or "body")
            parts.append(o)
            return o
        pre = "shotgun_%s_" % vname
        # barrels (rotate with the break bone)
        for s, nm in ((-1, "L"), (1, "R")):
            st = barrel_stations(s, chunk)
            put(hollow_tube(pre + "barrel_" + nm, st, "barrel", M["steel"]), "break")
            put(hollow_tube(pre + "bore_" + nm, st, "bore", M["bore"], end_inset=0.2), "break")
        put(rib_obj(pre + "rib", M["steel"], chunk), "break")
        for nm, u, r in (("bead_front", 697.0, 1.9), ("bead_mid", 380.0, 1.1)):
            ro = interp(R_OUT, u) * chunk
            b = C.sphere(pre + nm, mm(r), loc=P(0, Y_BREECH - u, AXIS_Z + ro - 0.5 + r * 0.5), mat=M["brass"], seg=16, rings=10)
            put(b, "break")
        lump = L.rbox(pre + "lumps", (mm(38), mm(80), mm(14)), P(0, -207, 39.5), mm(2.5), 2, M["steel"])
        put(lump, "break")
        put(forend_obj(pre + "forend", M["wood"], chunk), "break")
        iron = L.rbox(pre + "forend_tip", (mm(37 * chunk), mm(14), mm(11)), P(0, -486, 33), mm(3), 3, M["dark"])
        put(iron, "break")
        # fixed frame
        put(L.rbox(pre + "receiver", (mm(50), mm(86), mm(81)), P(0, -128, 30.5), mm(5 if vname == "clay" else 3), 3, M["recv"]))
        put(L.rbox(pre + "frame_bar", (mm(45), mm(84), mm(38)), P(0, -210, 11), mm(4), 3, M["recv"]))
        pin = C.cylinder(pre + "hinge_pin", mm(3.4), mm(50), loc=P(*HINGE), axis="X", verts=20, mat=M["dark"])
        put(pin)
        for s, nm in ((-1, "L"), (1, "R")):
            sp = C.cylinder(pre + "sideplate_" + nm, mm(21), mm(1.6), loc=P(s * 25.4, -122, 28), axis="X", verts=40, mat=M["recv"] if vname == "clay" else M["steel"])
            put(sp)
            for k, (dy, dz) in enumerate(((-13, 10), (14, 8), (0, -10))):
                sc = C.cylinder(pre + "screw_%s%d" % (nm, k), mm(2.1), mm(1.2), loc=P(s * 26.2, -122 + dy, 28 + dz), axis="X", verts=12, mat=M["dark"])
                put(sc)
            fp = C.cylinder(pre + "firing_pin_" + nm, mm(2.0), mm(1.0), loc=P(s * xb, Y_BREECH + 0.4, AXIS_Z), axis="Y", verts=14, mat=M["dark"])
            put(fp)
        # stock, pad, tang
        put(stock_obj(pre + "stock", M["wood"], -90, 252, 0.0, p_stock))
        padm = stock_obj(pre + "butt_pad", M["pad"], 252, 270, 1.4, p_stock, nst=6)
        put(padm)
        tang_pts = [(-92, 68.5), (-75, 67.5), (-50, 65.5), (-25, 62.5)]
        put(path_obj(pre + "tang", tang_pts, (16 * chunk, 3.2), M["recv"], 0, 6))
        # lever (rotates about vertical axis at the pivot): modelled closed, thumb piece rearward
        lev = L.rbox(pre + "top_lever", (mm(7 * chunk), mm(52), mm(3.4)), P(0, -112 + 26, 74.5), mm(1.2), 2, M["recv"])
        thumb = L.rbox(pre + "top_lever_thumb", (mm(13 * chunk), mm(15), mm(3.4)), P(0, -112 + 56, 71.4), mm(1.2), 2, M["recv"])
        lev_j = C.join([lev, thumb], pre + "top_lever")
        put(lev_j, "top_lever")
        # trigger guard and triggers
        put(path_obj(pre + "trigger_guard", [(-60, 3), (-66, -22), (-92, -48), (-132, -50), (-158, -30), (-172, -6)], (7.2 * tg, 4.6 * tg), M["recv"], 0, 6))
        for nm, yy in (("R", -82.0), ("F", -108.0)):
            tr = path_obj(pre + "trigger_" + nm, [(yy, -6), (yy - 1, -17), (yy - 6, -29), (yy - 12, -34)], (5.0 * tg, 4.0 * tg), M["dark"], 0, 5)
            put(tr, "trigger_" + nm)
        for s, nm in ((-1, "L"), (1, "R")):
            h, piv = hammer_obj(pre + "hammer_" + nm, s, M["dark"], chunk)
            put(h, "hammer_" + nm)
        return vc, parts

    clay_v, clay_parts = build_variant("clay", mats_clay(), 1.14, 2.4, 1.8)
    wood_v, wood_parts = build_variant("wood", mats_wood(), 1.0, 3.2)
    # clay variant default visible, wood variant default hidden
    wood_v.hide_viewport = True
    wood_v.hide_render = True
    return coll, root, arm, hooks, clay_v, wood_v, clay_parts, wood_parts


def main():
    coll, root, arm, hooks, clay_v, wood_v, clay_parts, wood_parts = build()
    bpy.context.view_layer.update()
    # verify placements: hooks follow rest transforms, muzzle at 710 mm from breech, overall length
    for k, h in hooks.items():
        w = h.matrix_world.translation
        print("HOOK", k, [round(c * 1000, 1) for c in w])
    bb = C.bbox_mm(clay_v)
    print("CLAY bbox", bb)
    # break pose test
    L.set_prop(root, "p_break", 1.0)
    print("OPEN muzzle_L", [round(c * 1000, 1) for c in L.world_pos(hooks["muzzle_L"])])
    print("OPEN chamber_L", [round(c * 1000, 1) for c in L.world_pos(hooks["chamber_L"])])
    L.set_prop(root, "p_break", 0.0)

    meta = dict(
        category="props", asset_family="shotgun", date="2026-10-02", accuracy="B (proportions of a typical 28 in side-by-side 12 gauge; clay variant stylised)",
        description="Double-barrel side-by-side break-action 12 gauge shotgun at real size: clay gag variant (VARIANT_clay, visible) and dark walnut + blued steel variant (VARIANT_wood, hidden by default; unhide the collection). One shared rig.",
        origin="Stock wrist centre = HOOK_grip_R at the root origin (mounting origin, not on z=0). Barrels point along -Y, up is +Z.",
        sources=[
            {"what": "12 gauge nominal bore 18.5 mm, shell 2-3/4 in (70 mm)", "url": "https://en.wikipedia.org/wiki/12-gauge_shotgun", "accessed": "2026-10-02"},
            {"what": "SAAMI 12 ga hull base diameter 0.809 in (secondary source)", "url": "https://www.shotgunworld.com/bbs/viewtopic.php?f=13&t=72561", "accessed": "2026-10-02"},
        ],
        dimensions_mm=[
            {"item": "overall length (butt pad to muzzle)", "value": 1150, "provenance": "task brief (about 1.15 m)", "level": "B"},
            {"item": "barrel length breech face to muzzle", "value": 710, "provenance": "task brief (28 in)", "level": "B"},
            {"item": "bore diameter", "value": 18.5, "provenance": "12 gauge nominal bore, Wikipedia", "level": "A"},
            {"item": "chamber diameter", "value": 20.4, "provenance": "estimate, +-0.5 (chamber for 2-3/4 in shell, hull 20.55)", "level": "C"},
            {"item": "barrel OD breech / muzzle", "value": "27 / 22", "provenance": "estimate +-2 (typical SxS)", "level": "C"},
            {"item": "length of pull (rear trigger to pad)", "value": 352, "provenance": "estimate; typical 14 in (356 mm)", "level": "C"},
            {"item": "receiver width x height", "value": "50 x 81", "provenance": "estimate +-5", "level": "C"},
            {"item": "stock drop at heel below rib line", "value": "about 55", "provenance": "estimate +-8", "level": "C"},
            {"item": "break-open angle at p_break=1", "value": "28 deg (0.49 rad)", "provenance": "design choice, muzzle drops about 304 mm", "level": "C"},
        ],
        hooks={
            "HOOK_break": "hinge pin (0,-240,-5 mm in root frame); barrels pivot about local +X; positive angle = muzzles down (rig bone 'break')",
            "HOOK_muzzle_L / HOOK_muzzle_R": "muzzle face centres; local +Z = firing direction (-Y world at rest), ring/flash planes are local XY; follow the barrels when broken open",
            "HOOK_chamber_L / HOOK_chamber_R": "breech-face centres for loading shells (+Z along the barrel)",
            "HOOK_grip_R": "right hand at the stock wrist (= root origin)",
            "HOOK_foregrip_L": "left hand under the fore-end (follows barrels)",
            "HOOK_shoulder": "butt pad centre for shouldering",
            "HOOK_trigger_finger": "trigger tip (follows trigger_F bone)",
        },
        custom_properties={
            "p_break": "0..1 break-open (driver on bone 'break', 0.49 rad)",
            "p_top_lever": "0..1 top lever swing (also follows p_break)",
            "p_hammer_cock_L / p_hammer_cock_R": "1 cocked (modelled pose), 0 dropped (0.70 rad forward)",
            "p_trigger": "0..1 both triggers pulled (0.30 rad)",
        },
        rig="Armature 'shotgun_rig': bones body, break, top_lever, hammer_L, hammer_R, trigger_R, trigger_F. Parts are bone-parented; drivers read the root props. Headless: after changing a root prop call root.update_tag() on the root and armature then view_layer.update().",
        variants={"VARIANT_clay": "clay (MAT_props_clay_*: stock, barrel, receiver, dark, bore, pad, brass), chunkier barrels (x1.14), thick guard; visible",
                  "VARIANT_wood": "procedural walnut stock/fore-end (MAT_props_wood_walnut), blued steel barrels, steel receiver, brass beads, rubber pad; hidden by default"},
        materials=["MAT_props_clay_stock", "MAT_props_clay_barrel", "MAT_props_clay_receiver", "MAT_props_clay_dark", "MAT_props_clay_bore", "MAT_props_clay_pad", "MAT_props_clay_brass",
                   "MAT_props_wood_walnut", "MAT_props_wood_blued_steel", "MAT_props_wood_receiver", "MAT_props_wood_trigger", "MAT_props_wood_bore", "MAT_props_wood_recoil_pad", "MAT_props_wood_brass"],
        simplifications=["boxlock receiver is a rounded slab with side plates (no internal mechanism)", "wood variant has no checkering or engraving (bump only)", "no choke, no ejectors, no sling swivels",
                         "bore liner is an inner tube (r 9.25 mm bore, 10.2 mm chamber) recessed 0.2 mm behind the muzzle face; no rifling or choke taper"],
        intended_usage="S1-S5 Manager shotgun (clay), S3 vendors and Gary (clay); wood variant for cutaways or alternative look. Muzzle flash and smoke ring from HOOK_muzzle_*.",
        previews="previews/shotgun_{clay,wood}_{front,side,three_quarter,top,closeup_action,closeup_muzzle,closeup_stock,broken_open}.png (front = muzzle end-on, side = broadside)",
    )
    out_blend = os.path.join(OUT, "shotgun.blend")
    items = [dict(asset_id="shotgun", coll=coll, root=root, meta={})]
    # variant meta
    meta_items_extra = {}
    L.write_family(out_blend, items, meta, PREV, previews=False)
    L.reopen(out_blend)
    root = bpy.data.objects["ROOT_shotgun"]
    clay_v = bpy.data.collections["VARIANT_clay"]
    wood_v = bpy.data.collections["VARIANT_wood"]
    hooks = {h.name[5:]: h for h in bpy.data.objects if h.name.startswith("HOOK_")}
    L.set_prop(root, "p_break", 1.0)
    print("OPEN muzzle_L (after reload)", [round(c * 1000, 1) for c in L.world_pos(hooks["muzzle_L"])])
    L.set_prop(root, "p_break", 0.0)
    # previews
    gun_views = [
        {"name": "front", "dir": "front", "fit": 2.4},
        {"name": "side", "dir": "side", "fit": 0.85},
        {"name": "three_quarter", "dir": (0.7, -0.8, 0.5), "fit": 0.95},
        {"name": "top", "dir": "top", "fit": 0.7},
        {"name": "closeup_action", "target": P(0, -130, 40), "cam_dir": Vector((0.7, -0.5, 0.45)), "dist": 0.42, "lens": 50},
        {"name": "closeup_muzzle", "target": P(0, -870, 57), "cam_dir": Vector((0.4, -0.9, 0.35)), "dist": 0.20, "lens": 50},
        {"name": "closeup_stock", "target": P(0, 120, 0), "cam_dir": Vector((0.9, 0.35, 0.4)), "dist": 0.5, "lens": 50},
    ]
    for vname, vc, hide in (("clay", clay_v, wood_v), ("wood", wood_v, clay_v)):
        vc.hide_viewport = False
        vc.hide_render = False
        hide.hide_viewport = True
        hide.hide_render = True
        pn = L.render_views([vc], PREV, "shotgun_" + vname, gun_views, floor=True)
        L.set_prop(root, "p_break", 1.0)
        pn += L.render_views([vc], PREV, "shotgun_" + vname, [{"name": "broken_open", "dir": (0.9, -0.3, 0.25), "fit": 0.8}], floor=True)
        L.set_prop(root, "p_break", 0.0)
        L.contact_sheet(pn, os.path.join(L.SCRATCH, "sheet_shotgun_%s.png" % vname), 3, (400, 300))


main()
