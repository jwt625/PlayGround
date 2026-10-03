"""Cable ties and hand tools: cable_tie (200 x 4.8 mm), hook_loop_strap, cable_tie_gun, side_cutters, cable_tie_box,
screwdriver, esd_wrist_strap. Each is an ASSET_<id> collection with ROOT_<id>; roots are laid out along +X in the file.

Run: Blender -b --python scripts/assets/props/build_cable_ties_tools.py -- assets/components/props
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402
import props_lib as L  # noqa: E402
from props_lib import C, mm  # noqa: E402

OUT, PREV = L.outdirs()
PI = math.pi
V3 = lambda x, y, z: Vector((mm(x), mm(y), mm(z)))  # noqa: E731

# cable tie dimensions (mm): length and width from manufacturer listings (200 x 4.8 mm), the rest estimates
TIE_L, TIE_W, TIE_T = 200.0, 4.8, 1.2
HEAD_L, HEAD_W, HEAD_H = 7.0, 6.0, 4.2
TOOTH_P, TOOTH_H = 1.2, 0.35


class Ctx:
    def __init__(self):
        self.items = []
        self.xcur = 0.0

    def item(self, aid, width_mm, accuracy="B"):
        coll, root = C.new_asset(aid, accuracy=accuracy)
        root.location = (mm(self.xcur), 0, 0)
        self.xcur += mm(width_mm) + 0.1
        self.cur = (aid, coll, root)
        return coll, root

    def done(self, meta):
        aid, coll, root = self.cur
        self.items.append(dict(asset_id=aid, coll=coll, root=root, meta=meta))


def A(o, coll, root):
    C.add(o, coll, root)
    return o


def mats():
    M = {}
    M["nylon"] = C.principled("MAT_props_tie_nylon", (0.88, 0.86, 0.8), rough=0.45)
    M["nylon_black"] = C.principled("MAT_props_tie_nylon_black", (0.03, 0.03, 0.035), rough=0.4)
    M["steel"] = C.principled("MAT_props_tool_steel", (0.72, 0.72, 0.74), metallic=1.0, rough=0.28)
    M["dark_steel"] = C.principled("MAT_props_tool_blackened", (0.12, 0.12, 0.14), metallic=1.0, rough=0.4)
    M["rubber_red"] = C.principled("MAT_props_grip_red", (0.75, 0.06, 0.04), rough=0.6)
    M["rubber_black"] = C.principled("MAT_props_grip_black", (0.03, 0.03, 0.035), rough=0.65)
    M["orange"] = C.principled("MAT_props_handle_orange", (0.9, 0.3, 0.04), rough=0.4)
    M["blue"] = C.principled("MAT_props_handle_blue", (0.05, 0.15, 0.55), rough=0.4)
    M["fabric"] = C.principled("MAT_props_strap_loop", (0.04, 0.04, 0.045), rough=0.95)
    M["hook"] = C.principled("MAT_props_strap_hook", (0.25, 0.27, 0.3), rough=0.8)
    M["carton"] = C.principled("MAT_props_box_carton", (0.55, 0.42, 0.26), rough=0.9)
    M["label"] = C.principled("MAT_props_box_label", (0.93, 0.93, 0.9), rough=0.8)
    M["ink"] = C.principled("MAT_props_ink", (0.04, 0.04, 0.05), rough=0.7)
    M["brass"] = C.principled("MAT_props_brass", (0.8, 0.6, 0.2), metallic=1.0, rough=0.3)
    M["cord"] = C.principled("MAT_props_esd_cord", (0.05, 0.3, 0.7), rough=0.45)
    M["band"] = C.principled("MAT_props_esd_band", (0.1, 0.1, 0.12), rough=0.9)
    return M


# ------------------------------------------------------------------ cable tie
def tie_mesh(M, teeth=True, name="cable_tie"):
    """Tie lying flat: strap along +X, head at -X end, teeth up. Centre of the strap at x = 0, bottom at z = 0."""
    bm = bmesh.new()
    x_head = -TIE_L / 2 + HEAD_L / 2
    x0, x1 = -TIE_L / 2 + HEAD_L - 1.0, TIE_L / 2
    rings = []
    n = 40
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        f = max(0.0, (x - (x1 - 16.0)) / 16.0)  # tip taper over the last 16 mm
        w = TIE_W * (1 - 0.5 * f)
        t = TIE_T * (1 - 0.45 * f)
        pts = [(-w / 2 + 0.4, 0), (w / 2 - 0.4, 0), (w / 2, 0.4), (w / 2, t - 0.3), (w / 2 - 0.3, t), (-w / 2 + 0.3, t), (-w / 2, t - 0.3), (-w / 2, 0.4)]
        rings.append(L.ring_frame(Vector((mm(x), 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1)), [(mm(a), mm(b)) for a, b in pts]))
    L.loft_bm(bm, rings, True, True)
    # head: left/right blocks and front/back walls around the through slot (slot runs along Z)
    hz = HEAD_H
    for sx in (-1, 1):
        L.box_bm(bm, (mm(2.5), mm(HEAD_W), mm(hz)), (mm(x_head + sx * (HEAD_L / 2 - 1.25)), 0, mm(hz / 2)), mm(0.4), 1)
    for sy in (-1, 1):
        L.box_bm(bm, (mm(HEAD_L), mm(0.5), mm(hz)), (mm(x_head), mm(sy * (HEAD_W / 2 - 0.25)), mm(hz / 2)), mm(0.1), 1)
    # pawl: small wedge inside the slot
    L.box_bm(bm, (mm(1.0), mm(3.4), mm(1.4)), (mm(x_head - 0.6), 0, mm(2.2)), 0, 1, rot=Matrix.Rotation(0.5, 3, "Y"))
    if teeth:
        xa, xb = -TIE_L / 2 + HEAD_L + 3.0, TIE_L / 2 - 18.0
        k = 0
        x = xa
        while x < xb:
            ta, tb = mm(TIE_T), mm(TIE_T + TOOTH_H)
            ys = (-mm(1.8), mm(1.8))
            a = [bm.verts.new((mm(x), y, ta)) for y in ys]
            b = [bm.verts.new((mm(x + TOOTH_P * 0.85), y, tb)) for y in ys]
            c = [bm.verts.new((mm(x + TOOTH_P), y, ta)) for y in ys]
            bm.faces.new((a[0], b[0], c[0]))
            bm.faces.new((c[1], b[1], a[1]))
            bm.faces.new((a[0], a[1], b[1], b[0]))
            bm.faces.new((b[0], b[1], c[1], c[0]))
            bm.faces.new((c[0], c[1], a[1], a[0]))
            x += TOOTH_P
            k += 1
    return L.obj_from_bm(name, bm, M["nylon"], True, True, 45)


def cable_tie(cx, M):
    coll, root = cx.item("cable_tie", 220)
    o = tie_mesh(M)
    A(o, coll, root)
    L.hook_at("head", coll, root, V3(-TIE_L / 2 + HEAD_L / 2, 0, HEAD_H / 2), size=0.01)
    L.hook_at("tail_tip", coll, root, V3(TIE_L / 2, 0, 0.5), size=0.01)
    cx.done(dict(accuracy="B", origin="strap centre, bottom at z = 0, lying flat, teeth up, head at -X",
                 hooks={"HOOK_head": "centre of the head slot", "HOOK_tail_tip": "end of the tapered tail"},
                 dims=[{"item": "length", "value": 200, "provenance": "manufacturer listing 200 x 4.8 mm (hwlok GT-200ST: 200 mm +-5, 4.8 mm +-0.2)", "level": "A"},
                       {"item": "strap width", "value": 4.8, "provenance": "same listing", "level": "A"},
                       {"item": "strap thickness", "value": 1.2, "provenance": "typical for 4.8 mm ties in manufacturer data (search result), not for this exact model", "level": "B"},
                       {"item": "head L x W x H", "value": "7.0 x 6.0 x 4.2", "provenance": "estimate +-1 (head dims not found for this exact tie)", "level": "C"},
                       {"item": "tooth pitch / height", "value": "1.2 / 0.35", "provenance": "estimate +-0.2", "level": "C"}],
                 notes="One mesh: tapered strap, head with through slot (along Z) and pawl wedge, 140+ sawtooth teeth over the usable length."))
    return o


# ------------------------------------------------------------------ hook-and-loop strap
def strap(cx, M):
    coll, root = cx.item("hook_loop_strap", 230, "C")
    A(L.rbox("hook_loop_strap_band", (mm(200), mm(12), mm(1.6)), (0, 0, mm(0.8)), mm(0.5), 1, M["fabric"]), coll, root)
    A(L.rbox("hook_loop_strap_hook_patch", (mm(80), mm(12), mm(0.7)), (mm(50), 0, mm(1.6 + 0.35)), mm(0.2), 1, M["hook"]), coll, root)
    ring = L.lathe("hook_loop_strap_ring", [(mm(3.0), -mm(6)), (mm(4.6), -mm(6)), (mm(4.6), mm(6)), (mm(3.0), mm(6))], 28,
                   xf=Matrix.Translation((-mm(96), 0, mm(2.2))) @ Matrix.Rotation(PI / 2, 4, "X") @ Matrix.Diagonal((1.35, 1.0, 1.0, 1.0)), closed_profile=True, mat=M["steel"])
    A(ring, coll, root)
    L.hook_at("middle", coll, root, V3(0, 0, 1.6), size=0.01)
    cx.done(dict(accuracy="C", origin="band centre, bottom at z = 0, lying flat; 200 x 12 x 1.6 mm band with a 80 mm hook patch and a steel D-ring (generic; no brand name)",
                 hooks={"HOOK_middle": "band centre"},
                 dims=[{"item": "band", "value": "200 x 12 x 1.6", "provenance": "typical reusable cable strap, estimate", "level": "C"}]))


# ------------------------------------------------------------------ cable tie gun
def tie_gun(cx, M):
    coll, root = cx.item("cable_tie_gun", 220, "C")
    # origin: bottom of the grip, front towards -Y. Frame along Y.
    A(L.rbox("tie_gun_body", (mm(18), mm(130), mm(30)), (0, -mm(10), mm(98)), mm(4), 2, M["steel"]), coll, root)
    A(L.rbox("tie_gun_nose", (mm(14), mm(46), mm(22)), (0, -mm(98), mm(96)), mm(3), 2, M["dark_steel"]), coll, root)
    A(L.rbox("tie_gun_nose_slot", (mm(5.4), mm(14), mm(1.8)), (0, -mm(110), mm(107.2)), 0, 1, M["ink"]), coll, root)
    pts = [Vector((0, mm(y), mm(z))) for y, z in ((40, 90), (55, 60), (60, 30), (56, 4))]
    bm = bmesh.new()
    L.tube_path_bm(bm, L.catmull(pts, 6), [(mm(a), mm(b)) for a, b in L.superellipse2d(26, 22, 14, 2.4)], binormal=Vector((1, 0, 0)), cap=True)
    A(L.obj_from_bm("tie_gun_grip", bm, M["rubber_red"], True, True, None), coll, root)
    # trigger lever pivots on a pin at the body front-bottom
    trig = L.rbox("tie_gun_trigger", (mm(10), mm(14), mm(52)), (0, 0, -mm(26)), mm(2.5), 2, M["steel"], rot=Matrix.Rotation(0.0, 3, "X"))
    trig.location = V3(0, 22, 84)
    A(trig, coll, root)
    A(C.cylinder("tie_gun_pivot", mm(3), mm(24), loc=V3(0, 22, 84), axis="X", verts=16, mat=M["dark_steel"]), coll, root)
    A(C.cylinder("tie_gun_tension_knob", mm(9), mm(7), loc=V3(0, 58, 113), axis="Z", verts=24, mat=M["rubber_black"]), coll, root)
    A(L.rbox("tie_gun_cutter_cover", (mm(20), mm(24), mm(6)), (0, -mm(60), mm(116)), mm(2), 1, M["dark_steel"]), coll, root)
    L.add_prop(root, "p_trigger", 0.0, 0.0, 1.0, "0 rest, 1 trigger squeezed (0.45 rad about the pivot pin, rotates towards the grip)")
    L.driver(trig, "rotation_euler", 0, root, ["p_trigger"], "v0*0.45")
    L.hook_at("nose", coll, root, V3(0, -118, 107.2), rot=(PI / 2, 0, 0), size=0.02)
    L.hook_at("grip", coll, root, V3(0, 56, 40), size=0.03)
    L.hook_at("trigger", coll, root, V3(0, 22, 62), size=0.02)
    cx.done(dict(accuracy="C", origin="bottom of the grip, nose towards -Y; about 190 x 30 x 125 mm; generic tensioning gun",
                 hooks={"HOOK_nose": "tie slot, +Z along the tie (feed direction towards +Z)", "HOOK_grip": "palm", "HOOK_trigger": "trigger finger pad"},
                 custom_properties={"p_trigger": "0..1 trigger squeeze (driver on tie_gun_trigger rotation X)"},
                 dims=[{"item": "overall length x height", "value": "about 190 x 125", "provenance": "estimate +-20 (generic tensioning tool)", "level": "C"}]))


# ------------------------------------------------------------------ side cutters
def side_cutters(cx, M):
    coll, root = cx.item("side_cutters", 80, "C")
    # origin at the pivot ground point; tool lies flat along Y, jaws towards -Y, pivot at origin; total length about 160 mm
    pivot_z = 7.0
    halves = []
    for side, nm in ((-1, "L"), (1, "R")):
        parts = []
        parts.append(L.rbox("s", (mm(7), mm(52), mm(12)), (mm(side * 3.5), mm(-33), mm(0)), mm(1.5), 1, M["steel"], rot=Matrix.Rotation(side * -0.08, 3, "Z")))
        parts.append(L.rbox("g", (mm(15), mm(80), mm(11)), (mm(side * 7.5), mm(48), mm(0)), mm(4.5), 2, M["rubber_red"], rot=Matrix.Rotation(side * -0.06, 3, "Z")))
        parts.append(L.rbox("p", (mm(10), mm(16), mm(12)), (0, 0, 0), mm(3), 2, M["steel"]))
        jaw = L.rbox("j", (mm(6), mm(22), mm(10)), (mm(-side * 0.0 + side * 1.0), mm(-66), mm(0)), mm(1.0), 1, M["dark_steel"], rot=Matrix.Rotation(side * -0.1, 3, "Z"))
        parts.append(jaw)
        o = C.join(parts, "side_cutters_half_" + nm)
        o.location = V3(0, 0, pivot_z)
        A(o, coll, root)
        halves.append(o)
    A(C.cylinder("side_cutters_pivot", mm(3.2), mm(14), loc=V3(0, 0, pivot_z), axis="Z", verts=16, mat=M["steel"]), coll, root)
    L.add_prop(root, "p_open", 0.0, 0.0, 1.0, "0 closed, 1 jaws open (0.2 rad each half about the pivot)")
    L.driver(halves[0], "rotation_euler", 2, root, ["p_open"], "v0*0.2")
    L.driver(halves[1], "rotation_euler", 2, root, ["p_open"], "-v0*0.2")
    L.hook_at("cut_point", coll, root, V3(0, -76, pivot_z), size=0.015)
    L.hook_at("grip", coll, root, V3(0, 50, pivot_z), size=0.03)
    cx.done(dict(accuracy="C", origin="pivot, lying flat, jaws towards -Y; about 160 x 38 x 14 mm; diagonal cutter (generic)",
                 hooks={"HOOK_cut_point": "jaw cutting edge (cut here)", "HOOK_grip": "handle centre"},
                 custom_properties={"p_open": "0..1 (drivers rotate the two halves about the pivot, origins at the pivot)"},
                 dims=[{"item": "overall length", "value": 160, "provenance": "6 in diagonal cutter, typical", "level": "B"}]))


# ------------------------------------------------------------------ box of ties
def tie_box(cx, M):
    coll, root = cx.item("cable_tie_box", 240, "C")
    Lx, Ly, Hz, t = 215.0, 70.0, 48.0, 1.5
    A(L.rbox("tie_box_floor", (mm(Lx), mm(Ly), mm(t)), (0, 0, mm(t / 2)), 0, 1, M["carton"]), coll, root)
    for nm, sx, sy, x, y in (("front", Lx, t, 0, -Ly / 2 + t / 2), ("back", Lx, t, 0, Ly / 2 - t / 2), ("left", t, Ly - 2 * t, -Lx / 2 + t / 2, 0), ("right", t, Ly - 2 * t, Lx / 2 - t / 2, 0)):
        A(L.rbox("tie_box_wall_" + nm, (mm(sx), mm(sy), mm(Hz)), (mm(x), mm(y), mm(Hz / 2)), 0, 1, M["carton"]), coll, root)
    A(L.rbox("tie_box_label", (mm(90), mm(0.3), mm(26)), (0, -mm(Ly / 2 + 0.15), mm(Hz / 2)), 0, 1, M["label"]), coll, root)
    A(L.text_obj("tie_box_label_text", "NYLON TIES 4.8 x 200", mm(7.5), loc=(0, -mm(Ly / 2 + 0.4), mm(Hz / 2 + 5)), rot=(PI / 2, 0, 0), extrude=mm(0.1), mat=M["ink"]), coll, root)
    A(L.text_obj("tie_box_label_text2", "100 PCS", mm(8), loc=(0, -mm(Ly / 2 + 0.4), mm(Hz / 2 - 6)), rot=(PI / 2, 0, 0), extrude=mm(0.1), mat=M["ink"]), coll, root)
    # ties: low-detail linked duplicates
    lo = tie_mesh(M, teeth=False, name="tie_box_tie_lowdetail")
    mesh = lo.data
    bpy.data.objects.remove(lo, do_unlink=True)
    rnd = random.Random(3)
    for layer in range(5):
        for k in range(12):
            o = bpy.data.objects.new("tie_box_tie_%d_%d" % (layer, k), mesh)
            bpy.context.scene.collection.objects.link(o)
            o.location = V3(rnd.uniform(-4, 4) + 0.0, -Ly / 2 + 6 + k * 5.2 + rnd.uniform(-0.3, 0.3), t + layer * 1.4 + 0.1)
            o.rotation_euler = (0, 0, rnd.uniform(-0.01, 0.01))
            A(o, coll, root)
    lid = L.rbox("tie_box_lid", (mm(Lx), mm(Ly), mm(1.5)), (0, -mm(Ly / 2), mm(0.75)), 0, 1, M["carton"])
    lid.location = V3(0, Ly / 2, Hz + 0.2)
    A(lid, coll, root)
    L.add_prop(root, "p_open", 1.0, 0.0, 1.0, "lid: 0 closed, 1 open (hinged at the back top edge, 109 deg)")
    L.driver(lid, "rotation_euler", 0, root, ["p_open"], "-v0*1.9")
    L.hook_at("pick", coll, root, V3(0, 0, Hz), size=0.02)
    cx.done(dict(accuracy="C", origin="bottom centre, front towards -Y; 215 x 70 x 48 mm carton, 60 low-detail ties (shared mesh) lying in 5 layers",
                 hooks={"HOOK_pick": "above the ties (pull one out here)"},
                 custom_properties={"p_open": "lid angle 0..1 (driver on tie_box_lid, hinge at the back top edge)"},
                 dims=[{"item": "box", "value": "215 x 70 x 48", "provenance": "estimate +-10 for a 100-pack of 200 mm ties", "level": "C"}],
                 notes="Label text is generic (no brand). tie_box_tie_* share one mesh without teeth."))


# ------------------------------------------------------------------ screwdriver
def screwdriver(cx, M):
    coll, root = cx.item("screwdriver", 260, "C")
    zc = 14.0
    hprof = [(0, 0), (8, 0), (12.5, 6), (14, 18), (14, 70), (12, 92), (9, 100), (0, 100)]
    xf = Matrix.Translation((0, mm(40), mm(zc))) @ Matrix.Rotation(-PI / 2, 4, "X")  # axis along +Y, handle end at y = 40 .. 140? (Z -> +Y)
    h = L.lathe("screwdriver_handle", [(mm(r), mm(z)) for r, z in hprof], 48, xf=xf, mat=M["orange"], sharp_deg=None)
    for v in h.data.vertices:  # six flutes
        a = math.atan2(v.co.z - mm(zc), v.co.x)
        r = math.hypot(v.co.x, v.co.z - mm(zc))
        k = 1.0 - 0.045 * (1 - math.cos(6 * a)) / 2 * (1 if 0.0 < v.co.y - mm(40) < mm(92) else 0)
        v.co.x = r * k * math.cos(a)
        v.co.z = mm(zc) + r * k * math.sin(a)
    A(h, coll, root)
    A(C.cylinder("screwdriver_cap", mm(9.5), mm(2), loc=V3(0, 140.6, zc), axis="Y", verts=24, mat=M["blue"]), coll, root)
    shaft = L.lathe("screwdriver_shaft", [(mm(3), -mm(100)), (mm(3), 0), (mm(2.4), mm(1))], 20, xf=Matrix.Translation((0, mm(40), mm(zc))) @ Matrix.Rotation(-PI / 2, 4, "X"), mat=M["steel"], sharp_deg=None)
    A(shaft, coll, root)
    # Phillips tip: cone + four wings
    A(L.lathe("screwdriver_tip", [(0, 0), (mm(2.6), mm(7)), (mm(3.0), mm(8))], 20, xf=Matrix.Translation((0, -mm(60), mm(zc))) @ Matrix.Rotation(PI / 2, 4, "X"), mat=M["dark_steel"], sharp_deg=None), coll, root)
    for k in range(2):
        A(L.rbox("screwdriver_wing_%d" % k, (mm(0.9 if k == 0 else 6.0), mm(7), mm(6.0 if k == 0 else 0.9)), (0, -mm(63.5), mm(zc)), 0, 1, M["dark_steel"]), coll, root)
    L.hook_at("grip", coll, root, V3(0, 100, zc), size=0.03)
    L.hook_at("tip", coll, root, V3(0, -68, zc), rot=(PI / 2, 0, 0), size=0.015)
    cx.done(dict(accuracy="C", origin="axis along Y, lying on its side (lowest point z = 0), tip towards -Y; about 208 mm long, handle 28 mm diameter, 6 mm shaft (Phillips #1 size, typical)",
                 hooks={"HOOK_grip": "handle centre", "HOOK_tip": "tip, +Z along the screw axis away from the tool"},
                 dims=[{"item": "handle diameter x length", "value": "28 x 100", "provenance": "typical, estimate", "level": "C"}, {"item": "shaft", "value": "6 mm x 100 mm", "provenance": "typical PH1, estimate", "level": "C"}]))


# ------------------------------------------------------------------ ESD wrist strap
def esd_strap(cx, M):
    coll, root = cx.item("esd_wrist_strap", 260, "C")
    A(L.lathe("esd_band", [(mm(28), 0), (mm(29.4), 0), (mm(29.4), mm(10)), (mm(28), mm(10))], 64, closed_profile=True, mat=M["band"], sharp_deg=None), coll, root)
    A(L.rbox("esd_plate", (mm(2.2), mm(20), mm(16)), (mm(30.5), 0, mm(5)), mm(0.6), 1, M["steel"]), coll, root)
    A(C.cylinder("esd_stud", mm(2.2), mm(5), loc=V3(33.8, 0, 5), axis="X", verts=16, mat=M["steel"]), coll, root)
    A(C.sphere("esd_stud_head", mm(3.4), loc=V3(36.5, 0, 5), seg=16, rings=8, mat=M["steel"]), coll, root)
    # coiled cord: 82 turns, coil diameter 7 mm, pitch 2.1 mm (1.8 m of wire)
    turns, rc, pitch = 82, 3.5, 2.1
    npt = turns * 12
    x0 = 52.0
    pts = []
    for i in range(npt + 1):
        a = 2 * PI * i / 12
        pts.append(Vector((mm(x0 + pitch * i / 12.0), mm(rc * math.cos(a)), mm(rc + 1.0 + rc * math.sin(a)))))
    lead_in = [Vector((mm(36.5 + (x0 - 36.5) * f), 0, mm(5 + (rc + 1.0 + 0 - 5) * f))) for f in (0.0, 0.5, 1.0)]
    bm = bmesh.new()
    L.tube_path_bm(bm, lead_in[:-1] + pts, [(mm(a), mm(b)) for a, b in L.circle2d(0.9, 8)], binormal=Vector((0, 1, 0)), cap=True)
    A(L.obj_from_bm("esd_cord_coil", bm, M["cord"], True, True, None), coll, root)
    xe = x0 + pitch * turns
    A(C.cylinder("esd_plug_body", mm(5.0), mm(14), loc=V3(xe + 9.0, 0, rc + 1.0), axis="X", verts=24, mat=M["rubber_black"]), coll, root)
    A(C.cylinder("esd_plug_pin", mm(2.0), mm(14), loc=V3(xe + 23.0, 0, rc + 1.0), axis="X", verts=16, mat=M["brass"]), coll, root)
    L.hook_at("wrist", coll, root, V3(0, 0, 5), size=0.03)
    L.hook_at("plug", coll, root, V3(xe + 30, 0, rc + 1.0), size=0.02)
    L.hook_at("snap", coll, root, V3(36.5, 0, 5), size=0.01)
    cx.done(dict(accuracy="C", origin="band centre on the ground (z = 0), band axis along Z, cord coil along +X; coil 82 turns of 7 mm diameter, 2.1 mm pitch (about 1.8 m of wire)",
                 hooks={"HOOK_wrist": "band centre", "HOOK_snap": "snap stud on the band", "HOOK_plug": "tip of the ground plug"},
                 dims=[{"item": "band inner diameter", "value": 56, "provenance": "typical adjustable wrist strap, estimate", "level": "C"},
                       {"item": "coiled cord extended length", "value": 1800, "provenance": "typical product spec (1.8 m), not verified; modelled retracted", "level": "C"},
                       {"item": "built-in series resistor", "value": "about 1 MOhm", "provenance": "typical ESD wrist strap practice (not verified, not modelled)", "level": "C"}],
                 notes="No cord resistor modelled; cord is one tube mesh."))


def main():
    C.reset()
    M = mats()
    cx = Ctx()
    for fn in (cable_tie, strap, tie_gun, side_cutters, tie_box, screwdriver, esd_strap):
        fn(cx, M)
    meta = dict(
        category="props", asset_family="cable_ties_tools", date="2026-10-02", accuracy="B for the 200 x 4.8 mm tie, C for the tools (generic proportions)",
        description="Cable tie (200 x 4.8 mm, head, tapered tail, ratchet teeth), hook-and-loop strap, tensioning gun, side cutters, box of ties, screwdriver, ESD wrist strap. Each item is an ASSET_ collection with its own root; roots are spread along +X in the file (reset on append).",
        sources=[{"what": "cable tie 200 mm x 4.8 mm: width 4.8 +-0.2 mm, length 200 +-5 mm, PA66", "url": "https://www.hwlok.com/en/product/GT-200ST.html", "accessed": "2026-10-02"},
                 {"what": "typical strap thickness 1.2 mm (search-result summary of manufacturer datasheets, not tied to one model)", "url": "https://www.hellermanntyton.com/products/cable-ties-inside-serrated/x80s/108-00003 (listed in search; not opened)", "accessed": "2026-10-02"}],
        simplifications=["tie head is a block frame around the slot plus one pawl wedge", "tools are generic and built from boxes and swept bars", "ESD cord is one tube without the resistor"],
        intended_usage="S6: Gary ties fiber bundles (tie, strap, gun, cutters, box); screwdriver and ESD strap as lab-tech props.",
    )
    blend = os.path.join(OUT, "cable_ties_tools.blend")
    L.write_family(blend, cx.items, meta, PREV, previews=False)
    L.reopen(blend)
    pn = []
    for it in cx.items:
        aid = it["asset_id"]
        coll = bpy.data.collections["ASSET_" + aid]
        views = L.std_views()
        if aid in ("cable_tie", "hook_loop_strap", "esd_wrist_strap", "screwdriver"):
            views = [{"name": "front", "dir": "front", "fit": 0.9}, {"name": "three_quarter", "dir": "three_quarter", "fit": 0.9}, {"name": "top", "dir": "top", "fit": 0.7}]
        pn += L.render_views([coll], PREV, "cable_ties_tools_" + aid, views)
    root = bpy.data.objects["ROOT_cable_tie"]
    pn += L.render_views([bpy.data.collections["ASSET_cable_tie"]], PREV, "cable_ties_tools_cable_tie",
                         [{"name": "closeup_head", "target": root.location + V3(-92, 0, 2), "cam_dir": Vector((0.3, -0.8, 0.7)), "dist": 0.04, "lens": 50}])
    L.set_prop(bpy.data.objects["ROOT_side_cutters"], "p_open", 1.0)
    pn += L.render_views([bpy.data.collections["ASSET_side_cutters"]], PREV, "cable_ties_tools_side_cutters", [{"name": "open", "dir": "top", "fit": 1.0}])
    L.contact_sheet(pn, os.path.join(L.SCRATCH, "sheet_tools.png"), 6, (300, 225))


main()
