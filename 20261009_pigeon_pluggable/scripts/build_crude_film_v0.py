"""Build the crude storyboard film v0 (DevLog-002 rev 2) as a single .blend.

Usage: blender -b --python scripts/build_crude_film_v0.py -- <out.blend>   (frozen copy of crude v0, 2026-10-09)
Crude on purpose: primitives, one continuous camera, captions, and [FX: ...] labels where effects will go.
World coordinates are absolute (no per-scene offset) so the camera can stay continuous:
  data hall: aisle along +y (racks at x = +-1.6, faces at x = +-1.1), end door at y = 13, street beyond;
  TX-eye macro set around x = 1000, RX-eye macro set around x = 2000 (entered through color fills).
Bird local frame: facing +x, up +z, right = -y. Front face at x = +0.21 (bird length 0.42 m in the hall).
"""
import math
import os
import random
import sys

import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_lib as L  # noqa: E402
from blender_lib import F, K, V, VA, box, cyl, sph, empty, mat, plane, narr, ovt, key_emit, set_blend, key_alpha, _text, curve_line  # noqa: E402

random.seed(11)
PI = math.pi
TOTAL = 1860
C = {
    "gun": (0.16, 0.17, 0.19), "gun2": (0.24, 0.25, 0.28), "dark": (0.05, 0.05, 0.06), "black": (0.01, 0.01, 0.012),
    "brass": (0.75, 0.55, 0.2), "gold": (0.95, 0.72, 0.25), "red": (1.0, 0.08, 0.05), "cyan": (0.1, 0.85, 1.0),
    "pcb": (0.05, 0.3, 0.12), "si": (0.18, 0.22, 0.3), "glass": (0.6, 0.85, 1.0), "white": (0.92, 0.92, 0.92),
    "orange": (1.0, 0.45, 0.05), "mag": (1.0, 0.1, 0.7), "holo": (0.2, 0.9, 1.0), "skin": (0.92, 0.72, 0.55),
    "hood": (0.16, 0.14, 0.2), "rack": (0.07, 0.075, 0.085), "floor": (0.09, 0.09, 0.1),
}
HERO_CELL = ("L", 7, 1, 1)
PARTNER_CELL = ("R", 7, 1, 1)
COMP_CELLS = [("L", 7, 2, 1), ("R", 7, 0, 1)]
ROWS_Z = (1.15, 1.45, 1.75)
COLS_DY = (-0.24, 0.0, 0.24)
N_RACKS = 13


def lab(i, t0, t1, s): ovt("LAB", s, i, t0, t1)
def fx(i, t0, t1, s): ovt("FX", s, i, t0, t1)
def big(i, t0, t1, s): ovt("BIG", s, i, t0, t1)


def rack_y(k):
    return 1.4 + 0.8 * k


def cell_pos(side, k, row, col):
    """Body centre of a bird sitting in a pigeonhole (front face 4 cm proud of the faceplate), and its yaw."""
    y = rack_y(k) + COLS_DY[col]
    z = ROWS_Z[row]
    if side == "L":
        return Vector((-1.06 - 0.21, y, z)), 0.0
    return Vector((1.06 + 0.21, y, z)), PI


def bw(pos, yaw, local):
    """Bird-local point to world."""
    return Vector(pos) + Matrix.Rotation(yaw, 3, "Z") @ Vector(local)


EYE_TX_L = (0.2165, -0.026, 0.004)   # pigeon's right eye (TX); seen from the front it is on the viewer's left
EYE_RX_L = (0.2165, 0.026, 0.004)


# ------------------------------------------------------------------ helpers
def new_objs(fn, *a, **k):
    before = set(bpy.data.objects)
    r = fn(*a, **k)
    return r, [o for o in bpy.data.objects if o not in before]


def move_to(coll, objs):
    for o in objs:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)


def instancer(name, coll, loc, yaw):
    o = bpy.data.objects.new(name, None)
    o.instance_type = "COLLECTION"
    o.instance_collection = coll
    o.location = loc
    o.rotation_euler = (0, 0, yaw)
    bpy.context.scene.collection.objects.link(o)
    return o


def beam(name, p0, p1, r, color, emit=6.0):
    p0, p1 = Vector(p0), Vector(p1)
    v = p1 - p0
    o = cyl(name, (p0 + p1) / 2, r, v.length, color, verts=8, emit=emit)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = v.to_track_quat("Z", "Y")
    return o


def wlab(body, pos, idx, t0, t1, size=0.05, color=(1, 1, 1), parent=None):
    """World billboard label (absolute or parent-relative position), with a dark shadow copy."""
    o = _text(body, size, color, pos, parent, "CENTER", "CENTER", billboard=True, fnt="Arial Bold")
    sh = _text(body, size, (0, 0, 0), (size * 0.05, -size * 0.05, -0.002), o, "CENTER", "CENTER", fnt="Arial Bold")
    V(o, idx, t0, t1)
    V(sh, idx, t0, t1)
    return o


def cam(idx, t, pos, tgt, lens=28.0, interp="BEZIER"):
    f = F(idx, t)
    K(L.CAM, f, loc=pos, interp=interp)
    K(L.TGT, f, loc=tgt, interp=interp)
    L.CAM.data.lens = lens
    L.CAM.data.keyframe_insert("lens", frame=f)
    L._interp(L.CAM.data, "lens", f, interp)
    L.HOLD.scale = (L.HOLD_S0 * L.LENS0 / lens,) * 3
    L.HOLD.keyframe_insert("scale", frame=f)
    L._interp(L.HOLD, "scale", f, interp)


def cam_jump(idx, t, prev, new):
    """Teleport (hidden by a fill): hold `prev` until the frame before t, then `new` at t."""
    f = F(idx, t)
    for (pos, tgt, lens), ff, it in ((prev, f - 1, "CONSTANT"), (new, f, "BEZIER")):
        K(L.CAM, ff, loc=pos, interp=it)
        K(L.TGT, ff, loc=tgt, interp=it)
        L.CAM.data.lens = lens
        L.CAM.data.keyframe_insert("lens", frame=ff)
        L._interp(L.CAM.data, "lens", ff, it)
        L.HOLD.scale = (L.HOLD_S0 * L.LENS0 / lens,) * 3
        L.HOLD.keyframe_insert("scale", frame=ff)
        L._interp(L.HOLD, "scale", ff, it)


def fill(idx, color, keys):
    """Full-frame color fill (behind the captions); keys = [(t, alpha)] in scene seconds."""
    m = mat(color, 1.0, 3.0, unique=True)
    set_blend(m)
    q = plane("fill", (0, 0, -0.15), (30, 30), m, rot=(0, 0, 0), parent=L.HOLD)
    for t, a in keys:
        key_alpha(m, F(idx, t), a)
    VA(q, F(idx, keys[0][0]), F(idx, keys[-1][0]) + 1)
    return q


def key_strength(m, frame, st, interp="LINEAR"):
    inp = m.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"]
    inp.default_value = st
    inp.keyframe_insert("default_value", frame=frame)
    for fc in m.node_tree.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            if int(round(kp.co[0])) == frame:
                kp.interpolation = interp


def add_light(name, kind, loc, energy, color=(1, 1, 1), size=None, rot=None):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color
    if size is not None:
        ld.size = size
    lo = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(lo)
    lo.location = loc
    if rot:
        lo.rotation_euler = rot
    return lo


def fade_mats(mats, idx, keys):
    for m in mats:
        set_blend(m)
        for t, a in keys:
            key_alpha(m, F(idx, t), a)


# ------------------------------------------------------------------ bird
class Bird:
    pass


def bird_body(p, hero=False):
    """Body parts in body-local coordinates under parent p. Returns dict of groups (empties) for the hero."""
    g = {}

    def grp(name):
        e = empty("grp_" + name, (0, 0, 0), parent=p)
        g[name] = e
        return e

    sb, st, fn = grp("shell_bottom"), grp("shell_top"), grp("fins")
    box("shell_b", (0, 0, -0.015), (0.42, 0.1, 0.03), C["gun"], sb, rough=0.35)
    box("shell_t", (0, 0, 0.015), (0.42, 0.1, 0.03), C["gun2"], st, rough=0.35)
    for k in range(8):
        box("fin", (-0.02, -0.042 + 0.012 * k, 0.041), (0.36, 0.005, 0.022), C["gun2"], fn, rough=0.3)
    box("emi_ruff", (0.12, 0, 0), (0.012, 0.108, 0.068), (0.55, 0.56, 0.6), st, rough=0.25)
    band = grp("band")
    box("leg_band", (-0.165, 0, 0), (0.01, 0.104, 0.064), C["orange"], band, rough=0.5)
    lt = grp("leash")
    box("leash", (0.265, 0, -0.034), (0.11, 0.014, 0.002), (0.05, 0.05, 0.05), lt)
    box("leash_loop", (0.322, 0, -0.036), (0.012, 0.02, 0.004), C["red"], lt)
    ft = grp("feet")
    mf = {}
    if hero:
        mf = {k: mat(c, 0.3, 0.0, unique=True) for k, c in (("gnd", C["brass"]), ("pwr", C["gold"]), ("sig", C["gold"]))}
    for side, y0 in (("tx", -0.0285), ("rx", 0.0285)):
        for k in range(8):
            y = y0 + (-0.0145 + k * 0.0041)
            o = box("toe_%s" % side, (-0.197, y, -0.0312), (0.026, 0.0024, 0.0016), C["gold"], ft, rough=0.3)
            if hero:
                o.data.materials[0] = mf["sig"]
        for k in range(3):
            y = y0 + (-0.016 + k * 0.016)
            o = box("claw_%s" % side, (-0.2, y, -0.0315), (0.04, 0.0022, 0.0018), C["brass"], ft, rough=0.3)
            if hero:
                o.data.materials[0] = mf["gnd"]
    for k in range(4):
        o = box("pad_ls", (-0.199, -0.006 + 0.004 * k, -0.0312), (0.032, 0.0024, 0.0016), C["gold"], ft, rough=0.3)
        if hero:
            o.data.materials[0] = mf["pwr"]
    if hero:
        # crude internals (placeholders; the detailed hero GLB comes from DevLog-003)
        pcb = grp("pcb")
        box("pcb", (-0.02, 0, -0.004), (0.39, 0.088, 0.003), C["pcb"], pcb)
        dsp = grp("dsp")
        box("dsp", (-0.04, 0, 0.002), (0.05, 0.05, 0.008), C["dark"], dsp, rough=0.4)
        box("thermal_pad", (-0.04, 0, 0.0075), (0.05, 0.05, 0.003), (0.6, 0.6, 0.62), dsp)
        tx = grp("tx_optics")
        box("tx_pic", (0.11, -0.024, 0.0), (0.06, 0.03, 0.003), C["si"], tx, rough=0.25)
        for k in range(8):
            box("mzm", (0.115, -0.036 + k * 0.0034, 0.0018), (0.04, 0.0012, 0.0006), C["gold"], tx, rough=0.3)
        for yy in (-0.031, -0.017):
            box("dfb", (0.078, yy, 0.001), (0.006, 0.003, 0.002), C["red"], tx, emit=3.0)
        box("tx_prism", (0.19, -0.026, 0.004), (0.014, 0.03, 0.014), C["glass"], tx, rough=0.05)
        rx = grp("rx_optics")
        box("pd_array", (0.168, 0.026, 0.0), (0.012, 0.03, 0.004), (0.12, 0.08, 0.1), rx, rough=0.3)
        box("tia", (0.14, 0.026, 0.0), (0.02, 0.025, 0.003), C["dark"], rx)
        box("rx_prism", (0.19, 0.026, 0.004), (0.014, 0.03, 0.014), C["glass"], rx, rough=0.05)
        el = grp("electronics")
        box("mcu", (-0.11, 0.025, 0.0), (0.016, 0.016, 0.003), C["dark"], el)
        box("eeprom", (-0.11, -0.02, 0.0), (0.008, 0.006, 0.002), C["dark"], el)
        box("crystal", (-0.075, 0.033, 0.0), (0.008, 0.006, 0.003), (0.75, 0.75, 0.8), el, rough=0.2)
        box("pmic", (-0.145, -0.025, 0.0), (0.012, 0.012, 0.003), C["dark"], el)
        box("inductor", (-0.145, 0.004, 0.002), (0.012, 0.012, 0.008), (0.3, 0.3, 0.32), el)
        box("load_switch", (-0.125, -0.036, 0.0), (0.005, 0.005, 0.002), C["dark"], el)
        for k in range(10):
            box("mlcc", (-0.17 + 0.03 * (k % 5), -0.04 + 0.075 * (k // 5), -0.001), (0.004, 0.002, 0.002), (0.6, 0.5, 0.35), el)
    return g, mf


def bird_head(p, eye_tx_m, eye_rx_m):
    """Head parts in head-local coordinates (body-local minus (0.15, 0, 0)) under parent p."""
    o = -0.15
    sph("skull", (0.17 + o, 0, 0.03), 1.0, C["gun"], p, scale=(0.06, 0.055, 0.035), rough=0.35, seg=16, rings=10)
    box("face", (0.21 + o, 0, 0.0), (0.006, 0.1, 0.058), C["dark"], p, rough=0.3)
    box("brow", (0.213 + o, 0, 0.027), (0.012, 0.1, 0.008), C["gun2"], p, rot=(0.35, 0, 0))
    for ey, m in ((EYE_TX_L[1], eye_tx_m), (EYE_RX_L[1], eye_rx_m)):
        cyl("iris", (0.2135 + o, ey, 0.004), 0.021, 0.004, C["black"], axis="X", parent=p, verts=20)
        for r in range(2):
            for c in range(4):
                s = sph("lenslet", (0.2165 + o, ey - 0.0135 + 0.009 * c, 0.004 - 0.0048 + 0.0096 * r), 0.0042,
                        C["glass"], p, seg=10, rings=6)
                s.data.materials[0] = m
    hinge_u = empty("beak_hinge_u", (0.212 + o, 0, -0.016), parent=p)
    box("beak_u", (0.014, 0, 0), (0.03, 0.016, 0.008), C["brass"], hinge_u, rough=0.3)
    hinge_l = empty("beak_hinge_l", (0.212 + o, 0, -0.022), parent=p)
    box("beak_l", (0.012, 0, -0.002), (0.026, 0.014, 0.006), C["brass"], hinge_l, rough=0.3)
    lids = []
    for ey in (EYE_TX_L[1], EYE_RX_L[1]):
        piv = empty("lid_pivot", (0.218 + o, ey, 0.026), parent=p)
        box("lid", (0, 0, -0.011), (0.004, 0.044, 0.022), C["gun2"], piv)
        piv.scale = (1, 1, 0.0)
        lids.append(piv)
    return hinge_l, lids


def make_bird(name, hero=False, unique_eyes=False):
    b = Bird()
    b.root = empty(name, (0, -500, 0))
    b.groups, b.feet_mats = bird_body(b.root, hero=hero)
    b.head_grp = empty(name + "_head_grp", (0, 0, 0), parent=b.root)   # explode group for the head
    b.head = empty(name + "_head", (0.15, 0, 0), parent=b.head_grp)
    if unique_eyes:
        b.eye_tx = mat(C["red"], 0.1, 6.0, unique=True)
        b.eye_rx = mat(C["cyan"], 0.1, 6.0, unique=True)
    else:
        b.eye_tx, b.eye_rx = mat(C["red"], 0.1, 6.0), mat(C["cyan"], 0.1, 6.0)
    b.beak_l, b.lids = bird_head(b.head, b.eye_tx, b.eye_rx)
    b.groups["head"] = b.head_grp
    return b


def bird_at(b, idx, t, pos, yaw=0.0, pitch=0.0, interp="BEZIER"):
    K(b.root, F(idx, t), loc=tuple(pos), rot=(0, pitch, yaw), interp=interp)


def beak(b, idx, t, open_amt, interp="BEZIER"):
    K(b.beak_l, F(idx, t), rot=(0, 0.6 * open_amt, 0), interp=interp)


def blink(b, idx, t0, t1):
    for lid in b.lids:
        K(lid, F(idx, t0), scale=(1, 1, 0.0), interp="LINEAR")
        K(lid, F(idx, (t0 + t1) / 2), scale=(1, 1, 1.0), interp="LINEAR")
        K(lid, F(idx, t1), scale=(1, 1, 0.0), interp="LINEAR")


# ------------------------------------------------------------------ subject (Gary-model stand-in)
class Subject:
    pass


def make_subject():
    s = Subject()
    s.root = empty("SUBJECT", (0, -500, 0))
    s.bob = empty("subj_bob", parent=s.root)
    sph("s_body", (0, 0, 0.95), 1.0, C["hood"], s.bob, scale=(0.24, 0.18, 0.42))
    sph("s_hood", (0, 0.05, 1.52), 0.21, C["hood"], s.bob)
    sph("s_head", (0, -0.03, 1.55), 0.16, C["skin"], s.bob)
    sph("s_nose", (0, -0.19, 1.53), 0.03, (0.95, 0.65, 0.5), s.bob)
    for sx in (-1, 1):
        sph("s_eye", (sx * 0.06, -0.17, 1.58), 0.035, (1, 1, 1), s.bob)
        sph("s_pup", (sx * 0.06, -0.2, 1.58), 0.018, (0, 0, 0), s.bob)
        cyl("s_leg", (sx * 0.1, 0, 0.3), 0.075, 0.6, (0.1, 0.1, 0.14), parent=s.bob)
    cyl("s_arm_l", (0.27, 0, 0.98), 0.055, 0.5, C["hood"], parent=s.bob)
    cyl("s_cup", (0.3, -0.08, 0.74), 0.04, 0.12, C["white"], parent=s.bob)
    s.rarm = empty("s_rarm", (-0.27, 0, 1.3), parent=s.bob)
    cyl("s_arm_r", (0, 0, -0.25), 0.055, 0.5, C["hood"], parent=s.rarm)
    sph("s_hand_r", (0, 0, -0.52), 0.06, C["skin"], s.rarm)
    return s


def subj_at(s, idx, t, pos, yaw, interp="LINEAR"):
    K(s.root, F(idx, t), loc=(pos[0], pos[1], 0.0), rot=(0, 0, yaw), interp=interp)


def subj_walk(s, idx, t0, t1, p0, p1, yaw):
    subj_at(s, idx, t0, p0, yaw)
    subj_at(s, idx, t1, p1, yaw)
    n = max(1, int((t1 - t0) / 0.25))
    for k in range(n + 1):
        t = t0 + (t1 - t0) * k / n
        K(s.bob, F(idx, t), loc=(0, 0, 0.03 * (k % 2)), interp="LINEAR")


def reach(s, idx, t, amt, interp="BEZIER"):
    K(s.rarm, F(idx, t), rot=(-1.4 * amt, 0, 0), interp=interp)


def yaw_face(src, dst):
    """Subject yaw (front = -y) to face dst."""
    return math.atan2(dst[0] - src[0], -(dst[1] - src[1]))


# ------------------------------------------------------------------ sets
def build_hall():
    objs = []
    fl = plane("hall_floor", (0, 6.0, 0), (6.4, 13.6), mat(C["floor"], 0.4), rot=(0, 0, 0))
    ce = plane("hall_ceiling", (0, 6.0, 3.6), (6.4, 13.6), mat(C["black"], 0.9), rot=(PI, 0, 0))
    objs += [fl, ce]
    for k in range(4):  # aisle floor tiles lines
        objs.append(box("tile_line", (-0.5 + k * 0.33, 6.0, 0.001), (0.008, 13.0, 0.001), (0.15, 0.15, 0.17)))
    objs.append(box("wall_back", (0, -0.8, 1.8), (6.4, 0.1, 3.6), C["black"]))
    objs.append(box("wall_end_l", (-1.9, 13.0, 1.8), (2.4, 0.12, 3.6), C["black"]))
    objs.append(box("wall_end_r", (1.9, 13.0, 1.8), (2.4, 0.12, 3.6), C["black"]))
    objs.append(box("wall_end_top", (0, 13.0, 3.05), (1.4, 0.12, 1.1), C["black"]))
    objs.append(box("door_frame", (0, 12.95, 2.52), (1.5, 0.05, 0.04), C["cyan"], emit=4.0))
    for k in range(4):
        objs.append(box("ceil_strip", (0, 1.5 + 3.2 * k, 3.55), (0.12, 2.2, 0.03), (0.85, 0.9, 1.0), emit=3.0))
    racks = {}
    for side, sx in (("L", -1), ("R", 1)):
        for k in range(N_RACKS):
            y = rack_y(k)
            rb = box("rack_%s%d" % (side, k), (sx * 1.6, y, 1.1), (1.0, 0.78, 2.2), C["rack"], rough=0.6)
            pn = box("panel_%s%d" % (side, k), (sx * 1.1, y, 1.45), (0.02, 0.76, 0.9), (0.12, 0.12, 0.14), rough=0.5)
            racks[(side, k)] = [rb, pn]
            objs += [rb, pn]
            for row in range(3):
                for col in range(3):
                    objs.append(box("hole", (sx * 1.088, y + COLS_DY[col], ROWS_Z[row]), (0.004, 0.2, 0.24), C["black"]))
            for zz in (0.35, 0.55, 2.0):  # blinking-LED strips (static in v0)
                objs.append(box("led", (sx * 1.098, y, zz), (0.004, 0.5, 0.012), C["cyan"] if k % 2 else C["mag"], emit=2.5))
    for o in objs:
        V(o, 0, 0, 10); V(o, 1, 0, 10); V(o, 2, 0, 10); V(o, 3, 0, 10); V(o, 4, 0, 10); V(o, 5, 0, 4.0)
    # lights
    for k in range(4):
        ld = bpy.data.lights.new("hall_area", "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = 0.6, 2.6
        ld.energy = 160
        ld.color = (0.8, 0.88, 1.0)
        lo = bpy.data.objects.new("hall_area", ld)
        bpy.context.scene.collection.objects.link(lo)
        lo.location = (0, 1.5 + 3.2 * k, 3.45)
    for k, (x, y, col) in enumerate([(-1.0, 3.0, C["mag"]), (1.0, 6.0, C["cyan"]), (-1.0, 9.0, C["cyan"]), (1.0, 11.5, C["mag"])]):
        pd = bpy.data.lights.new("accent", "POINT")
        pd.energy = 40
        pd.color = col
        po = bpy.data.objects.new("accent", pd)
        bpy.context.scene.collection.objects.link(po)
        po.location = (x, y, 2.6)
    return racks


def build_crowd(subject_track):
    """Crowd birds as collection instances (body + head); heads track the Subject. Returns head instancers."""
    cb = bpy.data.collections.new("CROWD_BODY")
    ch = bpy.data.collections.new("CROWD_HEAD")
    proot = empty("crowd_body_root")
    _, ob = new_objs(bird_body, proot)
    move_to(cb, [proot] + ob)
    hroot = empty("crowd_head_root")
    _, oh = new_objs(bird_head, hroot, mat(C["red"], 0.1, 6.0), mat(C["cyan"], 0.1, 6.0))
    move_to(ch, [hroot] + oh)
    skip = {HERO_CELL, PARTNER_CELL, *COMP_CELLS, ("L", 7, 1, 0)}  # (L,7,1,0): cutaway camera path
    heads, eyes = [], []
    for side in ("L", "R"):
        for k in range(N_RACKS):
            for row in range(3):
                for col in range(3):
                    if (side, k, row, col) in skip:
                        continue
                    pos, yaw = cell_pos(side, k, row, col)
                    bi = instancer("cb", cb, pos, yaw)
                    hi = instancer("ch", ch, bw(pos, yaw, (0.15, 0, 0)), yaw)
                    for o in (bi, hi):
                        for i in range(5):
                            V(o, i, 0, 10)
                        V(o, 5, 0, 4.0)
                    heads.append((hi, pos, yaw))
                    if row == 2:
                        eyes.append((side, pos, yaw))
    # head tracking: yaw toward the Subject at keyed times (clamped +-70 deg)
    for hi, pos, yaw in heads:
        for idx, t, sp in subject_track:
            ang = math.atan2(sp[1] - pos[1], sp[0] - pos[0])
            d = (ang - yaw + PI) % (2 * PI) - PI
            d = max(-1.2, min(1.2, d))
            K(hi, F(idx, t), rot=(0, 0, yaw + d), interp="BEZIER")
    return eyes


def build_hall_beams(eyes):
    """Cyberpunk laser web: top-row birds beam across the aisle, TX (right eye) to the partner's RX (left eye)."""
    left = [e for e in eyes if e[0] == "L"]
    right = [e for e in eyes if e[0] == "R"]
    random.shuffle(right)
    out = []
    for (s1, p1, y1), (s2, p2, y2) in zip(left[::2], right[:12]):
        a, b = bw(p1, y1, EYE_TX_L), bw(p2, y2, EYE_RX_L)
        o = beam("hall_beam", a, b, 0.0025, C["red"], emit=5.0)
        out.append(o)
    for (s2, p2, y2), (s1, p1, y1) in zip(right[1::2][:6], left[1::2][:6]):
        o = beam("hall_beam", bw(p2, y2, EYE_TX_L), bw(p1, y1, EYE_RX_L), 0.0025, C["red"], emit=5.0)
        out.append(o)
    for o in out:
        V(o, 0, 6.0, 10); V(o, 1, 0, 10); V(o, 2, 0, 10); V(o, 3, 0, 10); V(o, 4, 0, 10)
    return out


TREE = [  # (label, x, z, color, extinct)
    ("THEROPOD\n1x9 SOLDERED", 0.0, 0.55, C["holo"], False),
    ("ARCHAEOPTERYX\nGBIC (2000 SPEC REV)", -0.48, 0.82, C["holo"], False),
    ("MOA (EXTINCT)\nXENPAK 2001", 0.5, 1.02, (0.5, 0.6, 0.7), True),
    ("ELEPHANT BIRD (EXTINCT)\nCFP 2010", -0.5, 1.22, (0.5, 0.6, 0.7), True),
    ("EMU\nCFP2-DCO 2018", 0.5, 1.42, C["holo"], False),
    ("OSTRICH\nXPO 2026", -0.55, 1.62, C["holo"], False),
    ("CHICKEN\nSFP 2001", 0.5, 1.82, C["holo"], False),
    ("HUMMINGBIRD\nQSFP-DD 2016", -0.5, 2.02, C["holo"], False),
    ("ALBATROSS\n400ZR 2020", 0.55, 2.22, C["holo"], False),
    ("PARROT\n'COMPATIBLE' MODULES", -0.5, 2.42, C["holo"], False),
    ("CUCKOO\nAEC 2019", 0.5, 2.6, C["holo"], False),
    ("PASSENGER PIGEON (EXTINCT)\nQSFP+ (LEGACY)", -0.5, 2.78, (0.5, 0.6, 0.7), True),
    ("ROCK PIGEON\nQSFP28 2014", 0.5, 2.95, C["holo"], False),
    ("DODO (EXTINCT)\nCPO", -0.45, 3.1, (0.4, 0.4, 0.45), True),
]
CLADES = [("PALAEOGNATHAE", 1.02, 1.62), ("GALLOANSERAE", 1.82, 1.82), ("NEOAVES (UNRESOLVED)", 2.02, 2.42),
          ("COLUMBAVES", 2.6, 3.28)]
TREE_Y = 5.0


def build_tree():
    hm = mat(C["holo"], 0.5, 2.5, unique=True)
    gm = mat((0.5, 0.6, 0.7), 0.5, 1.2, unique=True)
    nm = mat(C["red"], 0.3, 6.0, unique=True)
    objs = []
    trunk = cyl("trunk", (0, TREE_Y, 1.9), 0.012, 2.8, C["holo"], verts=8)
    trunk.data.materials[0] = hm
    objs.append(trunk)
    for k, (lbl, x, z, col, ext) in enumerate(TREE):
        m = gm if ext else hm
        if k > 0:
            br = beam("branch", (0, TREE_Y, z - 0.08), (x, TREE_Y, z), 0.006, col, emit=2.0)
            br.data.materials[0] = m
            objs.append(br)
        ic = sph("leaf", (x, TREE_Y, z), 0.035, col, scale=(1.4, 1, 0.9), seg=12, rings=8)
        hd = sph("leaf_head", (x + 0.045, TREE_Y, z + 0.03), 0.018, col, seg=10, rings=6)
        ic.data.materials[0] = m
        hd.data.materials[0] = m
        objs += [ic, hd]
        objs.append(wlab(lbl, (x + (0.0 if x == 0 else 0.0), TREE_Y - 0.02, z - 0.085), 0, 0, 6.2, size=0.042,
                         color=(0.75, 0.95, 1.0) if not ext else (0.65, 0.7, 0.75)))
    for lbl, z0, z1 in CLADES:
        objs.append(wlab(lbl, (0.0, TREE_Y - 0.03, (z0 + z1) / 2 + 0.05), 0, 0, 6.2, size=0.03, color=(1.0, 0.85, 0.3)))
    node = sph("hero_node", (0, TREE_Y, 3.3), 0.06, C["red"], scale=(1.4, 1, 0.9))
    node.data.materials[0] = nm
    objs.append(node)
    objs.append(wlab("CYBER PIGEON\n800G OSFP DR8", (0.0, TREE_Y - 0.03, 3.42), 0, 0, 6.2, size=0.05, color=(1, 0.5, 0.45)))
    objs.append(wlab("RFC 1149\n1990-04-01\nIP OVER AVIAN CARRIERS", (0.78, TREE_Y - 0.02, 3.05), 0, 0, 6.2, size=0.032,
                     color=(1.0, 0.85, 0.3)))
    for o in objs:
        V(o, 0, 0, 6.2)
    fade_mats([hm, gm, nm], 0, [(5.3, 1.0), (6.1, 0.0)])


def build_tx_macro():
    """TX eye interior at x ~ 1000 (light travels +x). Returns nothing; all visible S3 1.8-8.8."""
    i, t0, t1 = 2, 1.8, 8.8
    X = 1000.0
    objs = [box("pic", (X + 5, 0, -0.15), (10, 5, 0.3), C["si"], rough=0.3),
            box("macro_floor", (X + 6, 0, -0.6), (40, 30, 0.2), C["black"])]
    for yy in (-1.0, 1.0):
        objs.append(box("dfb_sub", (X - 0.45, yy, 0.05), (0.9, 0.5, 0.2), (0.85, 0.85, 0.8)))
        objs.append(box("dfb", (X - 0.4, yy, 0.27), (0.8, 0.3, 0.22), C["red"], emit=4.0))
    lanes = [s * (0.25 + 0.5 * k) for s in (-1, 1) for k in range(4)]
    wg = (0.85, 0.9, 1.0)
    for yy in (-1.0, 1.0):  # splitter tree 1 -> 2 -> 4 per laser
        for d1 in (-0.5, 0.5):
            objs.append(curve_line("wg", [(X, yy, 0.02), (X + 1.2, yy + d1, 0.02)], 0.03, wg, emit=1.5))
            for d2 in (-0.25, 0.25):
                objs.append(curve_line("wg", [(X + 1.2, yy + d1, 0.02), (X + 2.4, yy + d1 + d2, 0.02)], 0.03, wg, emit=1.5))
    pam = []
    for k, ly in enumerate(lanes):
        objs.append(curve_line("wg", [(X + 2.4, ly, 0.02), (X + 3.4, ly, 0.02)], 0.03, wg, emit=1.5))
        for a in (-0.08, 0.08):
            objs.append(curve_line("arm", [(X + 3.4, ly, 0.02), (X + 3.6, ly + a, 0.02), (X + 6.8, ly + a, 0.02),
                                            (X + 7.0, ly, 0.02)], 0.025, wg, emit=1.5))
            objs.append(box("electrode", (X + 5.2, ly + a * 2.1, 0.03), (3.0, 0.07, 0.04), C["gold"], rough=0.3))
        seg = box("mzm_out", (X + 8.5, ly, 0.03), (3.0, 0.06, 0.04), C["red"], emit=3.0)
        m = mat(C["red"], 0.3, 3.0, unique=True)
        seg.data.materials[0] = m
        pam.append(m)
        objs.append(seg)
        mpd = box("mon_pd", (X + 7.4, ly + 0.17, 0.04), (0.14, 0.14, 0.05), (0.2, 0.15, 0.25))
        mm = mat((0.3, 1.0, 0.4), 0.3, 0.0, unique=True)
        mpd.data.materials[0] = mm
        objs.append(mpd)
        for f in range(F(i, 4.5), F(i, 8.0), 9):
            key_emit(mm, f, (0.3, 1.0, 0.4), random.choice([0.0, 0.0, 4.0]))
        # driver wire from the DSP block
        objs.append(curve_line("drv", [(X + 5.0, -3.6, 0.15), (X + 4.6, (ly - 3.6) / 2, 0.9), (X + 4.0, ly + 0.2, 0.05)], 0.02,
                               C["gold"], emit=0.3))
        # light pulses travelling along each lane
        for n in range(3):
            p = sph("pulse", (X, ly, 0.06), 0.07, (1.0, 0.3, 0.2), emit=8.0, seg=8, rings=6)
            for c in range(4):
                ta = 2.2 + c * 1.6 + n * 0.53
                if ta > t1:
                    break
                K(p, F(i, ta), loc=(X + 2.4, ly, 0.06), interp="LINEAR")
                K(p, F(i, min(ta + 1.6, t1)), loc=(X + 10.0, ly, 0.06), interp="LINEAR")
            objs.append(p)
    for f in range(F(i, 4.0), F(i, 8.8), 3):  # PAM4: four brightness levels per symbol (slowed down)
        for m in pam:
            key_emit(m, f, C["red"], random.choice([0.2, 1.5, 3.5, 7.0]))
    objs.append(box("dsp_block", (X + 5.0, -4.1, 0.2), (3.0, 1.2, 0.6), C["dark"]))
    objs.append(wlab("DSP (DRIVER INSIDE)", (X + 5.0, -4.1, 0.9), i, t0, t1, size=0.2, color=C["gold"]))
    pr = box("prism_tx", (X + 11.1, 0, 0.4), (1.0, 4.2, 1.6), C["glass"], rough=0.05)
    pm = mat(C["glass"], 0.05, 0.4, unique=True)
    pm.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.35
    set_blend(pm)
    pr.data.materials[0] = pm
    objs.append(pr)
    objs.append(box("lens_plate", (X + 12.4, 0, 0.4), (0.12, 4.0, 2.0), C["dark"]))
    for r in range(2):
        for c in range(4):
            yy, zz = -1.2 + 0.8 * c, 0.0 + 0.8 * r
            ln = sph("lenslet_m", (X + 12.5, yy, zz), 0.32, C["glass"], scale=(0.5, 1, 1), rough=0.05, emit=0.8)
            objs.append(ln)
            objs.append(beam("tx_beam", (X + 12.6, yy, zz), (X + 40, yy, zz), 0.12, C["red"], emit=4.0))
    objs.append(box("edge_couplers", (X + 9.95, 0, 0.03), (0.1, 4.0, 0.06), (0.9, 0.9, 1.0), emit=2.0))
    lt = bpy.data.lights.new("tx_area", "AREA")
    lt.size = 12
    lt.energy = 2500
    lo = bpy.data.objects.new("tx_area", lt)
    bpy.context.scene.collection.objects.link(lo)
    lo.location = (X + 6, 0, 6)
    for o in objs:
        V(o, i, t0, t1)
    wl = [("CW DFB LASER x2", (X - 0.4, 0, 0.9), 2.0, 5.2), ("SPLITTER TREE", (X + 1.4, 0, 0.6), 3.0, 5.5),
          ("8 MACH-ZEHNDER MODULATORS", (X + 5.2, 2.5, 0.8), 4.5, 7.4), ("MONITOR PDs", (X + 7.4, 2.4, 0.5), 5.0, 7.4),
          ("EDGE COUPLERS", (X + 9.9, 2.4, 0.5), 6.6, 8.2), ("PRISM: 1x8 ROW -> 2x4", (X + 11.1, 2.6, 1.4), 7.0, 8.6),
          ("2x4 LENSLET ARRAY", (X + 12.5, 2.4, 1.5), 7.2, 8.6)]
    for s, p, a, b in wl:
        wlab(s, p, i, a, b, size=0.18, color=(0.75, 0.95, 1.0))


def build_rx_macro():
    i, t0, t1 = 3, 1.8, 6.8
    X = 2000.0
    objs = [box("macro_floor_rx", (X + 4, 0, -0.6), (40, 30, 0.2), C["black"])]
    objs.append(box("lens_plate_rx", (X, 0, 0.4), (0.12, 4.0, 2.0), C["dark"]))
    for r in range(2):
        for c in range(4):
            yy, zz = -1.2 + 0.8 * c, 0.0 + 0.8 * r
            objs.append(sph("lenslet_rx", (X - 0.1, yy, zz), 0.32, C["glass"], scale=(0.5, 1, 1), rough=0.05, emit=0.8))
            objs.append(beam("rx_in", (X - 30, yy, zz), (X - 0.2, yy, zz), 0.12, C["red"], emit=4.0))
            objs.append(beam("rx_conv", (X + 0.1, yy, zz), (X + 1.4, -1.75 + 0.5 * (r * 4 + c), 0.4), 0.05, C["red"], emit=3.0))
    pr = box("prism_rx", (X + 1.6, 0, 0.4), (0.6, 4.2, 1.4), C["glass"], rough=0.05)
    pm = mat(C["glass"], 0.05, 0.4, unique=True)
    pm.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.35
    set_blend(pm)
    pr.data.materials[0] = pm
    objs.append(pr)
    objs.append(box("pd_chip", (X + 3.0, 0, 0.4), (0.3, 4.4, 1.0), (0.12, 0.08, 0.1)))
    pds = []
    for k in range(8):
        ly = -1.75 + 0.5 * k
        objs.append(beam("rx_row", (X + 1.9, ly, 0.4), (X + 2.84, ly, 0.4), 0.05, C["red"], emit=3.0))
        pd = box("pd", (X + 2.84, ly, 0.4), (0.04, 0.35, 0.35), C["cyan"], emit=1.0)
        m = mat(C["cyan"], 0.3, 1.0, unique=True)
        pd.data.materials[0] = m
        pds.append(m)
        objs.append(pd)
        objs.append(curve_line("bond", [(X + 3.15, ly, 0.85), (X + 3.7, ly, 1.3), (X + 4.3, ly * 0.8, 0.12)], 0.02, C["gold"]))
        objs.append(curve_line("trace", [(X + 5.6, ly * 0.8, 0.02), (X + 8.4, ly * 0.4, 0.02)], 0.03, C["gold"], emit=0.5))
        for n in range(2):
            p = sph("sig", (X + 5.6, ly * 0.8, 0.05), 0.06, C["cyan"], emit=8.0, seg=8, rings=6)
            for c in range(3):
                ta = 4.4 + c * 0.8 + n * 0.4
                if ta > t1:
                    break
                K(p, F(i, ta), loc=(X + 5.6, ly * 0.8, 0.05), interp="LINEAR")
                K(p, F(i, min(ta + 0.8, t1)), loc=(X + 8.4, ly * 0.4, 0.05), interp="LINEAR")
            objs.append(p)
    for f in range(F(i, 2.0), F(i, 6.8), 3):
        for m in pds:
            key_emit(m, f, C["cyan"], random.choice([0.2, 2.0, 4.5, 8.0]))
    objs.append(box("tia_chip", (X + 5.0, 0, 0.0), (1.4, 3.6, 0.2), C["dark"]))
    for k in range(16):
        objs.append(sph("bump", (X + 4.4 + 0.4 * (k % 4), -1.2 + 0.8 * (k // 4), -0.12), 0.06, (0.8, 0.8, 0.85), seg=8, rings=6))
    objs.append(box("dsp_rx", (X + 9.0, 0, 0.2), (1.4, 2.0, 0.5), C["dark"]))
    lt = bpy.data.lights.new("rx_area", "AREA")
    lt.size = 10
    lt.energy = 2000
    lo = bpy.data.objects.new("rx_area", lt)
    bpy.context.scene.collection.objects.link(lo)
    lo.location = (X + 3, 0, 6)
    for o in objs:
        V(o, i, t0, t1)
    for s, p, a, b in [("2x4 LENSLET ARRAY", (X, 2.4, 1.5), 1.9, 3.6), ("PRISM: 2x4 -> 1x8 ROW", (X + 1.6, 2.6, 1.3), 2.2, 3.8),
                       ("8 PIN PHOTODIODES", (X + 3.0, 2.6, 1.2), 2.6, 4.8), ("OCTAL TIA", (X + 5.0, 2.4, 0.6), 4.4, 6.6),
                       ("TO THE DSP", (X + 9.0, 1.4, 0.8), 5.2, 6.6)]:
        wlab(s, p, i, a, b, size=0.18, color=(0.75, 0.95, 1.0))


def build_hatchery():
    """Assembly line that fades in behind the hero during S5 8.3-9.9 (aisle, y ~ 8.6-9.8)."""
    i = 4
    belt_m = mat((0.15, 0.15, 0.17), 0.5, 0.0, unique=True)
    robot_m = mat((0.9, 0.5, 0.1), 0.4, 0.0, unique=True)
    item_m = mat(C["pcb"], 0.5, 0.0, unique=True)
    chip_m = mat(C["dark"], 0.4, 0.0, unique=True)
    shell_m = mat(C["gun2"], 0.3, 0.0, unique=True)
    eye_m = mat(C["red"], 0.2, 5.0, unique=True)
    back_m = mat((0.02, 0.02, 0.03), 0.9, 0.0, unique=True)
    sign_m = mat(C["mag"], 0.3, 4.0, unique=True)
    objs = []
    bd = box("hatch_backdrop", (0, 10.1, 1.5), (3.0, 0.05, 2.2), (0.02, 0.02, 0.03))
    bd.data.materials[0] = back_m
    objs.append(bd)
    belt = box("belt", (0, 9.2, 1.25), (1.8, 0.3, 0.05), (0.15, 0.15, 0.17))
    belt.data.materials[0] = belt_m
    objs.append(belt)
    for k in range(5):
        x = -0.8 + 0.4 * k
        rb = box("robot", (x, 9.2, 1.75 + 0.05 * (k % 2)), (0.05, 0.05, 0.4), (0.9, 0.5, 0.1))
        rb.data.materials[0] = robot_m
        objs.append(rb)
        for tt in (8.2, 8.6, 9.0, 9.4, 9.8):
            K(rb, F(i, tt + 0.1 * k), loc=(x, 9.2, 1.7 + 0.08 * ((int(tt * 5) + k) % 2)), interp="LINEAR")
        for part, col_m, size, dz, lvl in (("p_pcb", item_m, (0.2, 0.06, 0.008), 0.03, 0), ("p_dsp", chip_m, (0.04, 0.04, 0.01), 0.04, 1),
                                           ("p_shell", shell_m, (0.22, 0.07, 0.04), 0.06, 3), ("p_eye", eye_m, (0.01, 0.05, 0.02), 0.06, 4)):
            if k < lvl:
                continue
            o = box(part, (x, 9.2, 1.28 + dz), size, (0.5, 0.5, 0.5))
            o.data.materials[0] = col_m
            if part == "p_eye":
                o.location.x += 0.115
            K(o, F(i, 8.2), loc=(o.location.x, 9.2, o.location.z), interp="LINEAR")
            K(o, F(i, 9.9), loc=(o.location.x + 0.4, 9.2, o.location.z), interp="LINEAR")
            objs.append(o)
    sg = _text("OUTNOLIGHT HATCHERY", 0.12, C["mag"], (0, 10.05, 2.25), None, "CENTER", "CENTER", rot=(PI / 2, 0, 0), fnt="Arial Black")
    sg.data.materials[0] = sign_m
    objs.append(sg)
    for o in objs:
        V(o, i, 8.0, 10.0)
    fade_mats([belt_m, robot_m, item_m, chip_m, shell_m, eye_m, back_m, sign_m], i,
              [(8.2, 0.0), (8.6, 1.0), (9.5, 1.0), (9.95, 0.0)])


def build_street():
    i = 5
    objs = []
    gm = mat((0.04, 0.045, 0.06), 0.12)
    objs.append(plane("street", (0, 23, 0.0), (18, 20), gm, rot=(0, 0, 0)))
    for sx in (-1, 1):
        for k in range(5):
            h = 6 + 3 * ((k * 7 + (sx > 0)) % 3)
            objs.append(box("bldg", (sx * 7.5, 15 + 4 * k, h / 2), (3.0, 3.6, h), (0.05, 0.05, 0.07)))
            col = [C["mag"], C["cyan"], C["orange"]][(k + (sx > 0)) % 3]
            objs.append(box("neon", (sx * 5.95, 15 + 4 * k, 3.0 + k % 2), (0.05, 2.0, 0.5), col, emit=5.0))
    for txt, x, y, z, col in (("RAMEN 24H", -5.9, 19, 4.9, C["orange"]), ("NO SIGNAL", 5.9, 23, 4.2, C["cyan"]),
                              ("LINK UP", -5.9, 27, 3.8, C["mag"])):
        objs.append(_text(txt, 0.5, col, (x + (0.05 if x < 0 else -0.05), y, z), None, "CENTER", "CENTER",
                          rot=(PI / 2, 0, PI / 2 if x < 0 else -PI / 2), fnt="Arial Black"))
    for k in range(60):
        objs.append(sph("star", (random.uniform(-40, 40), random.uniform(30, 80), random.uniform(20, 45)), 0.08, (1, 1, 1),
                        emit=6.0, seg=6, rings=4))
    for o in objs:
        V(o, 4, 8.0, 10.0)  # visible through the door near the end of S5
        V(o, i, 0, 10)
    for x, y in ((-3, 15), (3, 21), (-3, 26)):
        pd = bpy.data.lights.new("street_lamp", "POINT")
        pd.energy = 120
        pd.color = (1.0, 0.5, 0.8)
        po = bpy.data.objects.new("street_lamp", pd)
        bpy.context.scene.collection.objects.link(po)
        po.location = (x, y, 4.0)


POSTS_C = Vector((0, 17.5, 0))
POSTS = [POSTS_C + Vector((2.6 * math.cos(a), 2.6 * math.sin(a), 0)) for a in (PI / 2, PI / 2 + 2 * PI / 3, PI / 2 + 4 * PI / 3)]


def build_posts_and_devices():
    i = 5
    objs = []
    for p in POSTS:
        objs.append(cyl("post", (p.x, p.y, 1.6), 0.06, 3.2, (0.2, 0.2, 0.22)))
        objs.append(box("lamp", (p.x, p.y, 3.22), (0.3, 0.3, 0.06), (1.0, 0.85, 0.6), emit=4.0))
    cams = [Vector((-5.8, 15.5, 4.2)), Vector((5.8, 21.5, 4.6)), Vector((-5.8, 24.5, 3.9))]
    for c in cams:
        objs.append(box("cctv", (c.x, c.y, c.z), (0.4, 0.22, 0.2), (0.85, 0.85, 0.88)))
        objs.append(cyl("cctv_lens", (c.x + (0.22 if c.x < 0 else -0.22), c.y, c.z), 0.07, 0.08, C["black"], axis="X"))
        objs.append(sph("cctv_led", (c.x + (0.25 if c.x < 0 else -0.25), c.y, c.z + 0.1), 0.025, C["red"], emit=8.0))
    sat = empty("satellite", (6, 38, 26))
    box("sat_body", (0, 0, 0), (0.8, 0.8, 0.8), (0.8, 0.75, 0.5), sat)
    for sx in (-1, 1):
        box("sat_panel", (sx * 1.3, 0, 0), (1.6, 0.05, 0.7), (0.15, 0.2, 0.6), sat, emit=0.6)
    sph("sat_led", (0, -0.45, 0.3), 0.08, C["red"], sat, emit=10.0)
    K(sat, F(i, 0), loc=(4, 38, 26), interp="LINEAR")
    K(sat, F(i, 10), loc=(9, 38, 27), interp="LINEAR")
    for o in objs + [sat] + list(sat.children):
        V(o, i, 0, 10)
    return cams, sat


# ------------------------------------------------------------------ scenes
def main_build():
    s = make_subject()
    racks = build_hall()
    hero_rack = racks[("L", 7)]
    # subject positions used for crowd head tracking
    track = [(0, 8.0, (0.3, 11.8)), (0, 8.7, (0.25, 10.4)), (0, 9.4, (0.2, 9.0)), (0, 10.0, (-0.25, 7.6)),
             (1, 2.0, (-0.25, 7.6)), (4, 0.5, (-0.3, 7.6)), (4, 4.5, (0.5, 12.0))]
    eyes = build_crowd(track)
    build_hall_beams(eyes)
    build_tree()
    build_tx_macro()
    build_rx_macro()
    build_hatchery()
    build_street()
    cams, sat = build_posts_and_devices()
    moon = add_light("street_moon", "SUN", (0, 20, 30), 0.6, (0.55, 0.65, 1.0), rot=(0.5, 0.2, 0.3))
    V(moon, 5, 0, 10)
    V(add_light("street_fill", "AREA", (0, 17.5, 9), 900, (0.8, 0.6, 1.0), size=10), 5, 0, 10)

    hero = make_bird("HERO", hero=True, unique_eyes=True)
    partner = make_bird("PARTNER")
    comps = [make_bird("COMP%d" % k) for k in range(2)]
    hp, hy = cell_pos(*HERO_CELL)
    pp, py = cell_pos(*PARTNER_CELL)
    cps = [cell_pos(*c) for c in COMP_CELLS]
    bird_at(partner, 0, 0, pp, py, interp="CONSTANT")
    for k, b in enumerate(comps):
        bird_at(b, 0, 0, cps[k][0], cps[k][1], interp="CONSTANT")

    # cutaway cage + host connector for the hero cell (S2 6.8 - S3 0.8)
    cage = [box("cage_top", (hp.x, hp.y, hp.z + 0.04), (0.44, 0.12, 0.004), (0.55, 0.56, 0.6), rough=0.3),
            box("host_conn", (hp.x - 0.235, hp.y, hp.z - 0.028), (0.03, 0.11, 0.03), (0.1, 0.1, 0.1)),
            box("host_pcb", (hp.x - 0.3, hp.y, hp.z - 0.045), (0.16, 0.3, 0.004), C["pcb"])]
    cage.append(add_light("cutaway_light", "POINT", (hp.x - 0.05, hp.y - 0.25, hp.z + 0.12), 6.0, (0.9, 0.95, 1.0)))
    for o in cage:
        V(o, 1, 6.8, 10.0)
        V(o, 2, 0, 0.8)
    for o in hero_rack:
        L._VIS.setdefault(o, [])
    # hero rack is hidden during the cutaway: rebuild its windows
    for o in hero_rack:
        L._VIS[o] = [(F(0, 0), F(1, 6.8)), (F(2, 0.8), F(5, 4.0))]

    # ---------------- S1
    i = 0
    cam(i, 0.0, (0.05, 3.55, 0.65), (0, TREE_Y, 0.7), 24)
    cam(i, 1.5, (-0.12, 3.6, 1.45), (0, TREE_Y, 1.5), 24)
    cam(i, 3.0, (0.12, 3.6, 2.3), (0, TREE_Y, 2.35), 24)
    cam(i, 4.5, (0.0, 3.65, 3.05), (0, TREE_Y, 3.15), 24)
    cam(i, 5.2, (0.0, 4.5, 3.28), (0, TREE_Y, 3.3), 26)
    cam(i, 5.7, (-0.05, 5.1, 3.1), (-0.9, 6.8, 1.6), 26)
    cam(i, 6.5, (-0.35, 6.2, 1.62), (-1.06, 7.2, 1.45), 28)
    cam(i, 7.6, (-0.2, 8.3, 1.62), (-1.1, 9.6, 1.45), 28)
    cam(i, 8.4, (-0.1, 8.6, 1.6), (0.2, 11.5, 1.45), 28)
    cam(i, 9.3, (-0.35, 7.4, 1.55), (-0.6, 8.4, 1.5), 28)
    cam(i, 10.0, (-0.62, 7.3, 1.5), (-1.06, 7.0, 1.45), 32)
    subj_at(s, i, 0, (0.3, 13.5), 0.0, interp="CONSTANT")
    subj_walk(s, i, 8.0, 10.0, (0.3, 11.8), (-0.25, 7.6), 0.0)
    subj_at(s, i, 10.0, (-0.25, 7.6), yaw_face((-0.25, 7.6), (-1.06, 7.0)))
    bird_at(hero, i, 0, hp, hy, interp="CONSTANT")
    narr(i, [(0.2, 2.0, "Birds evolved from dinosaurs."), (2.0, 3.6, "Or so they told you."),
             (3.6, 7.6, "Meet the OSFP: Overhead Surveillance Feathered Platform."), (8.0, 9.8, "It is watching you.")])
    lab(i, 0.0, 4.6, "BIRD FAMILY TREE (TRAIT-MATCHED)")
    lab(i, 6.4, 8.2, "800G OSFP DR8\nDR8: DRONE RECON, 8 LANES")
    lab(i, 8.2, 10.0, "HEADS TRACK THE SUBJECT")
    fx(i, 0.0, 4.6, "[FX: hologram flicker, scanlines, leaf birds animated]")
    fx(i, 6.4, 8.2, "[FX: volumetric haze so the 1311 nm beams read]")

    # ---------------- S2
    i = 1
    subj_at(s, i, 0.0, (-0.25, 7.6), yaw_face((-0.25, 7.6), (-1.06, 7.0)), interp="CONSTANT")
    reach(s, i, 0.0, 0.0); reach(s, i, 0.4, 0.9); reach(s, i, 1.8, 0.75); reach(s, i, 6.8, 0.75); reach(s, i, 7.6, 0.95)
    reach(s, i, 9.4, 0.9); reach(s, i, 10.0, 0.0)
    held = Vector((-0.62, 7.05, 1.45))
    bird_at(hero, i, 0.0, hp, 0.0)
    bird_at(hero, i, 0.3, hp, 0.0)
    bird_at(hero, i, 1.4, held + Vector((0, 0, -0.02)), 0.0, pitch=0.25)
    bird_at(hero, i, 2.0, held, 0.0, pitch=0.0)
    bird_at(hero, i, 4.4, held, 0.0, pitch=0.0)
    bird_at(hero, i, 5.0, held, 0.0, pitch=-1.25)
    bird_at(hero, i, 6.6, held, 0.0, pitch=-1.25)
    bird_at(hero, i, 7.2, Vector((-1.0, 7.0, 1.45)), 0.0, pitch=0.0)
    bird_at(hero, i, 7.7, Vector((-1.24, 7.0, 1.45)), 0.0)
    bird_at(hero, i, 9.0, hp, 0.0)
    beak(hero, i, 0.0, 0.0); beak(hero, i, 0.3, 1.0); beak(hero, i, 1.0, 0.0)
    beak(hero, i, 3.2, 0.0); beak(hero, i, 3.45, 1.0); beak(hero, i, 3.7, 0.0); beak(hero, i, 3.95, 1.0); beak(hero, i, 4.2, 0.0)
    beak(hero, i, 9.2, 0.0); beak(hero, i, 9.35, 1.0); beak(hero, i, 9.55, 0.0)
    for m in (hero.eye_tx, hero.eye_rx):  # unplugged = no power = dark eyes
        m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.04, 0.05, 0.06, 1)
        for tt, st, it in ((0.3, 6.0, "LINEAR"), (0.8, 0.0, "CONSTANT"), (8.95, 0.0, "LINEAR"), (9.2, 6.0, "CONSTANT")):
            key_strength(m, F(i, tt), st, it)
    for name, tt in (("gnd", 7.9), ("pwr", 8.4), ("sig", 8.9)):
        m = hero.feet_mats[name]
        col = C["brass"] if name == "gnd" else C["gold"]
        key_emit(m, F(i, tt - 0.05), col, 0.0, interp="LINEAR")
        key_emit(m, F(i, tt), col, 12.0, interp="LINEAR")
        key_emit(m, F(i, tt + 0.4), col, 0.5, interp="CONSTANT")
    cam(i, 0.0, (-0.62, 7.3, 1.5), (-1.06, 7.0, 1.45), 32)
    cam(i, 1.0, (-0.3, 7.32, 1.55), (-0.75, 7.05, 1.46), 32)
    cam(i, 2.0, (-0.16, 7.1, 1.5), (-0.42, 7.05, 1.455), 35)
    cam(i, 3.0, (-0.2, 7.06, 1.465), (-0.41, 7.05, 1.455), 40)
    cam(i, 4.4, (-0.21, 7.04, 1.462), (-0.41, 7.05, 1.455), 40)
    cam(i, 5.0, (-0.36, 7.0, 1.3), (-0.65, 7.05, 1.27), 40)
    cam(i, 6.6, (-0.38, 7.1, 1.27), (-0.655, 7.05, 1.255), 45)
    cam(i, 7.4, (-1.15, 6.72, 1.44), (-1.35, 7.0, 1.43), 30)
    cam(i, 8.0, (-1.37, 6.84, 1.375), (-1.465, 7.0, 1.418), 40)
    cam(i, 9.2, (-1.39, 6.86, 1.372), (-1.47, 7.0, 1.418), 44)
    cam(i, 10.0, (-1.2, 6.75, 1.45), (-1.1, 7.0, 1.45), 32)
    narr(i, [(0.2, 1.8, "Feathers: heatsink."), (1.8, 4.2, "Beak: a latch that bites the cage."),
             (4.6, 6.0, "Right foot and eye transmit."), (6.0, 7.4, "Left foot and eye receive."), (7.6, 9.4, "Ground lands first.")])
    lab(i, 0.0, 2.0, "PULL TAB = LEASH  |  BEAK = RETENTION LATCH")
    lab(i, 2.0, 4.5, "EYES: 2x4 LENSLET ARRAYS\nRIGHT = TX (RED)  LEFT = RX (CYAN)\nUNPLUGGED: NO POWER, EYES DARK")
    lab(i, 4.5, 7.0, "60 CONTACTS\nRIGHT FOOT TX | LOW-SPEED + POWER | LEFT FOOT RX")
    lab(i, 7.0, 10.0, "CUTAWAY: GROUND -> POWER -> SIGNAL")
    fx(i, 0.0, 2.0, "[FX: feather-fin shimmer, cable-tension on the leash]")
    fx(i, 7.6, 9.4, "[FX: contact glow pulses in mating order]")

    # ---------------- S3
    i = 2
    eye_tx_w = bw(hp, 0.0, EYE_TX_L)
    cam(i, 0.0, (-1.2, 6.75, 1.45), (-1.1, 7.0, 1.45), 32)
    cam(i, 0.9, (-0.75, 6.95, 1.46), eye_tx_w, 32)
    cam(i, 1.9, (eye_tx_w.x + 0.02, eye_tx_w.y, eye_tx_w.z), eye_tx_w + Vector((-0.1, 0, 0)), 40)
    fill(i, C["red"], [(1.4, 0.0), (1.9, 1.0), (2.0, 1.0), (2.5, 0.0)])
    X = 1000.0
    cam_jump(i, 2.0, ((eye_tx_w.x + 0.02, eye_tx_w.y, eye_tx_w.z), eye_tx_w + Vector((-0.1, 0, 0)), 40),
             ((X + 14.0, 0.3, 1.3), (X + 11.5, 0, 0.3), 26))
    cam(i, 3.0, (X + 6.5, 0.9, 1.3), (X + 3.0, 0, 0.1), 26)
    cam(i, 4.0, (X + 1.6, 0.6, 1.1), (X - 0.4, 1.0, 0.25), 26)
    cam(i, 4.6, (X + 0.2, -0.9, 1.4), (X + 3.0, 0, 0.0), 26)
    cam(i, 5.8, (X + 3.4, -0.9, 1.25), (X + 5.6, 0.0, 0.0), 26)
    cam(i, 7.0, (X + 6.6, -1.0, 1.2), (X + 9.6, 0, 0.1), 26)
    cam(i, 8.0, (X + 9.6, -1.6, 1.9), (X + 12.4, 0, 0.4), 26)
    cam(i, 8.55, (X + 13.2, -1.2, 1.7), (X + 18, 0, 0.4), 26)
    fill(i, C["red"], [(8.1, 0.0), (8.55, 1.0), (8.65, 1.0), (9.1, 0.0)])
    pe = bw(pp, py, EYE_RX_L)
    cam_jump(i, 8.6, ((X + 13.2, -1.2, 1.7), (X + 18, 0, 0.4), 26),
             ((eye_tx_w.x + 0.06, eye_tx_w.y - 0.04, eye_tx_w.z + 0.025), pe, 30))
    cam(i, 10.0, (0.2, eye_tx_w.y - 0.03, eye_tx_w.z + 0.02), pe, 30)
    # the 8-beam bundle hero TX eye -> partner RX eye (hall scale)
    for r in range(2):
        for c in range(4):
            off = Vector((0, -0.0135 + 0.009 * c, -0.0048 + 0.0096 * r))
            o = beam("link_beam", eye_tx_w + off, Vector((pe.x, pe.y - off.y, pe.z + off.z)), 0.0016, C["red"], emit=6.0)
            V(o, 2, 8.6, 10.0); V(o, 3, 0, 2.0); V(o, 4, 0, 1.4)
    subj_at(s, i, 0.0, (-0.3, 7.6), yaw_face((-0.3, 7.6), (-1.06, 7.0)), interp="CONSTANT")
    narr(i, [(0.2, 1.8, "Right eye: transmit."), (2.0, 4.4, "Two DFB lasers glow deep in its skull."),
             (4.6, 6.8, "Eight Mach-Zehnder modulators."), (7.0, 8.0, "Eight lenses."), (8.2, 9.4, "Eight beams.")])
    lab(i, 2.0, 8.6, "TX SILICON PHOTONICS CHIP")
    lab(i, 8.6, 10.0, "FREE-SPACE LINK TO THE BIRD ACROSS THE AISLE")
    fx(i, 1.4, 2.5, "[FX: red fill = lens glow, macro emerges]")
    fx(i, 4.6, 7.0, "[FX: optical-mode glow along waveguides, PAM4 shimmer]")

    # ---------------- S4
    i = 3
    cam(i, 0.0, (0.2, eye_tx_w.y - 0.03, eye_tx_w.z + 0.02), pe, 30)
    cam(i, 1.2, (pe.x - 0.25, pe.y - 0.02, pe.z + 0.04), pe, 32)
    cam(i, 1.9, (pe.x - 0.03, pe.y, pe.z), pe + Vector((0.1, 0, 0)), 36)
    fill(i, C["cyan"], [(1.4, 0.0), (1.9, 1.0), (2.0, 1.0), (2.5, 0.0)])
    X = 2000.0
    cam_jump(i, 2.0, ((pe.x - 0.03, pe.y, pe.z), pe + Vector((0.1, 0, 0)), 36), ((X - 2.2, 0.5, 1.0), (X, 0, 0.4), 26))
    cam(i, 3.2, (X + 1.4, -1.6, 1.6), (X + 3.0, 0, 0.5), 26)
    cam(i, 4.5, (X + 2.2, -1.9, 1.4), (X + 3.2, 0, 0.4), 26)
    cam(i, 6.0, (X + 5.4, -1.6, 1.2), (X + 8.0, 0, 0.1), 26)
    fill(i, C["cyan"], [(5.9, 0.0), (6.35, 1.0), (6.45, 1.0), (6.9, 0.0)])
    ph = Vector((pp.x - 0.25, pp.y, pp.z + 0.012))
    cam_jump(i, 6.4, ((X + 5.6, -1.6, 1.2), (X + 8.0, 0, 0.1), 26), (ph, (-0.3, 7.6, 1.45), 16))
    cam(i, 10.0, ph + Vector((-0.1, 0, 0)), (-0.3, 7.6, 1.5), 22)
    narr(i, [(0.2, 1.8, "Left eye: receive."), (2.0, 4.2, "Eight lenses, eight photodiodes."),
             (4.4, 7.2, "A TIA, a transimpedance amplifier, turns tiny currents into voltage."), (7.6, 9.2, "It sees you.")])
    lab(i, 2.0, 6.4, "RX: LENSLETS -> PRISM -> PIN ARRAY -> TIA")
    lab(i, 6.4, 10.0, "BIRD POV\nSUBJECT ACQUIRED")
    fx(i, 6.4, 10.0, "[FX: 340 deg fisheye, surveillance HUD, PAM4 eye glyphs on his face]")
    hud = []
    for (x, y, w, h) in ((0, 0.55, 1.3, 0.02), (0, -1.2, 1.3, 0.02), (-0.65, -0.32, 0.02, 1.77), (0.65, -0.32, 0.02, 1.77)):
        q = plane("hud", (x, y, 0.0), (w, h), mat(C["red"], emit=100), rot=(0, 0, 0), parent=L.HOLD)
        hud.append(q)
    for q in hud:
        V(q, i, 6.6, 10.0)

    # ---------------- S5
    i = 4
    out = Vector((-0.1, 7.0, 1.55))
    cam(i, 0.0, ph + Vector((-0.1, 0, 0)), (-0.3, 7.6, 1.5), 22)
    cam(i, 0.4, (0.85, 6.95, 1.5), (-1.06, 6.98, 1.45), 26)
    cam(i, 1.1, (0.25, 6.9, 1.52), (-0.9, 7.0, 1.47), 28)
    cam(i, 2.0, (0.42, 6.42, 1.86), out + Vector((0, 0, 0.03)), 26)
    bird_at(hero, i, 0.0, hp, 0.0)
    bird_at(hero, i, 0.9, hp, 0.0)
    bird_at(hero, i, 1.8, out, 0.0)
    bird_at(hero, i, 8.0, out, 0.0)
    ex = {"shell_top": (0, 0, 0.12), "fins": (0, 0, 0.17), "shell_bottom": (0, 0, -0.1), "head": (0.08, 0, 0.02),
          "tx_optics": (0.02, -0.07, 0.05), "rx_optics": (0.02, 0.07, 0.05), "dsp": (0, 0, 0.08), "electronics": (0, 0, 0.045),
          "feet": (-0.07, 0, -0.04), "leash": (0.1, 0, -0.03), "band": (-0.03, 0, -0.07), "pcb": (0, 0, 0)}
    for name, d in ex.items():
        g = hero.groups[name]
        base = (0.0, 0.0, 0.0) if name != "head" else (0.0, 0.0, 0.0)
        K(g, F(i, 1.6), loc=base, interp="BEZIER")
        K(g, F(i, 2.6), loc=tuple(base[k] + d[k] for k in range(3)), interp="BEZIER")
        K(g, F(i, 8.0), loc=tuple(base[k] + d[k] for k in range(3)), interp="BEZIER")
        K(g, F(i, 8.6), loc=base, interp="BEZIER")
    dsp_w = out + Vector((-0.04, 0, 0.002 + 0.08))
    cam(i, 3.0, (0.2, 6.62, 1.8), dsp_w, 32)
    cam(i, 4.5, (0.05, 6.78, 1.75), dsp_w, 45)
    cam(i, 5.5, (-0.22, 6.72, 1.75), out + Vector((-0.12, 0, 0.04)), 38)
    cam(i, 6.5, (-0.4, 6.72, 1.72), out + Vector((-0.2, 0, 0.0)), 34)
    cam(i, 8.0, (0.1, 6.55, 1.72), (-0.6, 7.0, 1.5), 30)
    cam(i, 8.6, (0.14, 6.2, 1.66), (-0.1, 7.6, 1.52), 32)
    cam(i, 10.0, (0.12, 6.0, 1.66), (-0.05, 8.0, 1.55), 32)
    for k, name in enumerate(("NARROWCOM", "MERVALL", "MINLINEAR")):
        t = _text(name, 0.008, C["gold"], (-0.04, 0, 0.0125), hero.groups["dsp"], "CENTER", "CENTER", fnt="Arial Black")
        V(t, i, 2.3 + 0.75 * k, 3.05 + 0.75 * k if k < 2 else 8.0)
    labels = [("DSP (BRAIN)", "dsp", (-0.04, 0, 0.03), 2.4, 8.0), ("MCU", "electronics", (-0.11, 0.025, 0.02), 4.6, 8.0),
              ("EEPROM", "electronics", (-0.11, -0.02, 0.02), 4.8, 8.0), ("CRYSTAL", "electronics", (-0.075, 0.04, 0.02), 5.0, 8.0),
              ("POWER", "electronics", (-0.145, -0.012, 0.022), 5.2, 8.0), ("TX PIC + 2 DFB", "tx_optics", (0.11, -0.025, 0.02), 2.7, 8.0),
              ("PIN ARRAY + TIA", "rx_optics", (0.155, 0.026, 0.02), 2.7, 8.0), ("60-PAD FEET", "feet", (-0.2, 0, -0.045), 2.8, 8.0),
              ("SHELL + FIN FEATHERS", "fins", (0.0, 0, 0.04), 2.8, 4.4), ("EYES + BEAK LATCH", "head", (0.06, 0, 0.06), 2.8, 4.4)]
    for txt, gname, off, a, b in labels:
        wlab(txt, off, i, a, b, size=0.009, color=(0.75, 0.95, 1.0), parent=hero.groups[gname])
    # I2C snitch line: from the feet (middle pads) to the hero's cell in the switch
    feet_w = out + Vector((-0.2 - 0.07, 0, -0.031 - 0.04))
    spine = beam("i2c_spine", feet_w, (-1.09, 7.0, 1.45), 0.003, (0.3, 1.0, 0.4), emit=6.0)
    V(spine, i, 6.4, 8.0)
    for n in range(5):
        p = sph("i2c_bit", feet_w, 0.008, (0.3, 1.0, 0.4), emit=10.0, seg=8, rings=6)
        for c in range(3):
            ta = 6.4 + c * 0.5 + n * 0.1
            K(p, F(i, ta), loc=tuple(feet_w), interp="LINEAR")
            K(p, F(i, ta + 0.5), loc=(-1.09, 7.0, 1.45), interp="LINEAR")
        V(p, i, 6.4, 8.0)
    ovt("CARD", "I2C -> SWITCH: TEMPERATURE / VCC / TX BIAS / TX POWER / RX POWER\nSUBJECT: HEART RATE ELEVATED", i, 6.5, 8.0)
    wlab("HATCHED BY OUTNOLIGHT", (-0.165, 0, 0.07), i, 8.6, 10.0, size=0.014, color=C["orange"], parent=hero.groups["band"])
    subj_walk(s, i, 0.5, 4.5, (-0.3, 7.6), (0.5, 12.0), PI)
    narr(i, [(1.8, 4.2, "The brain: a DSP from NARROWCOM or MERVALL."),
             (4.6, 7.8, "A microcontroller reports temperature, laser current and light levels."), (8.2, 9.8, "Hatched by OUTNOLIGHT.")])
    lab(i, 1.6, 8.0, "EXPLODED VIEW (CRUDE PLACEHOLDER PARTS;\nDETAILED HERO MODEL IN PROGRESS)")
    lab(i, 8.2, 10.0, "THE HATCHERY")
    fx(i, 8.2, 10.0, "[FX: crossfade hall -> hatchery assembly line -> hall]")

    # ---------------- S6
    i = 5
    subj_walk(s, i, 0.0, 2.0, (0.0, 12.0), (0.0, 14.4), PI)
    subj_walk(s, i, 2.0, 4.4, (0.0, 14.4), (0.0, 17.5), PI)
    subj_at(s, i, 8.6, (0.0, 17.5), PI)
    flock = [hero] + comps
    starts = [out] + [cps[0][0], cps[1][0]]
    for k, b in enumerate(flock):
        p = POSTS[k]
        yaw = math.atan2(POSTS_C.y - p.y, POSTS_C.x - p.x)
        perch = Vector((p.x, p.y, 3.3))
        bird_at(b, i, 0.0 + 0.2 * k, starts[k], 0.0 if k != 2 else PI)
        bird_at(b, i, 1.3 + 0.2 * k, (0.2 * (k - 1), 12.4, 2.2), PI / 2)
        bird_at(b, i, 2.6 + 0.2 * k, (0.6 * (k - 1), 15.0, 3.6), PI / 2)
        bird_at(b, i, 4.3 + 0.2 * k, perch + Vector((0, 0, 0.15)), yaw)
        bird_at(b, i, 4.8 + 0.2 * k, perch, yaw)
        bird_at(b, i, 8.1, perch, yaw)
        b.perch, b.yaw = perch, yaw
    # heads to camera at the end
    camp = Vector((-0.3, 19.0, 3.0))
    for b in comps:
        yaw2 = math.atan2(camp.y - b.perch.y, camp.x - b.perch.x)
        bird_at(b, i, 8.7, b.perch, yaw2)
        bird_at(b, i, 10.0, b.perch, yaw2)
    bird_at(hero, i, 10.0, hero.perch, hero.yaw)
    blink(hero, i, 9.35, 9.75)
    # free-space ring + uplinks
    for k, b in enumerate(flock):
        nb = flock[(k + 1) % 3]
        o = beam("ring", bw(b.perch, b.yaw, EYE_TX_L), bw(nb.perch, nb.yaw, EYE_RX_L), 0.012, C["red"], emit=6.0)
        V(o, i, 5.0 + 0.5 * k, 8.15)
    up = beam("uplink", bw(hero.perch, hero.yaw, EYE_TX_L), (6.5, 38, 26.2), 0.03, C["red"], emit=6.0)
    V(up, i, 6.3, 8.15)
    for k, c in enumerate(cams):
        b = flock[k % 3]
        o = beam("cctv_link", c, bw(b.perch, b.yaw, EYE_RX_L), 0.012, C["cyan"], emit=5.0)
        V(o, i, 6.8 + 0.25 * k, 8.15)
    cam(i, 0.0, (0.12, 6.0, 1.66), (-0.05, 8.0, 1.55), 32)
    cam(i, 1.0, (0.1, 9.8, 1.8), (0.0, 13.0, 1.6), 28)
    cam(i, 2.2, (0.0, 12.6, 2.0), (0.0, 16.5, 1.8), 26)
    cam(i, 3.6, (4.5, 15.0, 5.0), POSTS_C + Vector((0, 0, 2.0)), 24)
    cam(i, 5.0, (0.6, 17.2, 12.0), POSTS_C + Vector((0, 0.3, 0.0)), 22)
    cam(i, 6.6, (-3.2, 13.6, 7.0), POSTS_C + Vector((0, 0, 2.6)), 22)
    cam(i, 8.0, (-0.3, 19.0, 3.0), hero.perch, 26)
    cam(i, 9.0, (-0.1, 19.6, 3.32), hero.perch, 40)
    cam(i, 10.0, bw(hero.perch, hero.yaw, (0.42, 0.0, 0.01)), bw(hero.perch, hero.yaw, (0.2, 0.0, 0.0)), 50)
    narr(i, [(0.2, 2.0, "They follow you outside."), (2.6, 4.8, "Eye to eye, bird to bird."),
             (5.2, 6.8, "Up to the satellites."), (6.8, 8.8, "Down to every camera on the street.")])
    lab(i, 5.0, 8.2, "FREE-SPACE OPTICAL RING\n+ SATELLITE UPLINK + CCTV LINKS")
    lab(i, 9.3, 10.0, "BLINK = TX DISABLE")
    fx(i, 2.0, 5.0, "[FX: wing flaps, rain, neon reflections on wet street]")

    # ---------------- S7
    i = 6
    V(box("black", (0, -300, 0), (14, 0.1, 18), (0, 0, 0), emit=100), i, 0, 2)
    ovt("DISC", "DISCLAIMER:\nPigeons are real. Parody; not\naffiliated with any company or bird.\nOne typical 800G DR8 build shown.\n"
                "Real DR8 uses 16 fibers via MPO;\nhere the eyes do free-space optics\n(stylized lenslet arrays).\n"
                "Modules cannot see. 1311 nm beams\ndrawn visible. Real pigeons: 4 toes.", i, 0, 2.0)
    src = ("SOURCES\nOSFP MSA Module Specification Rev 5.2 / 5.22; FS and Approved Networks 800G DR8 datasheets\n"
           "Credo / DustPhotonics (CW DFB butt-coupled to SiPh); Gigalight DR8 block diagram; Semtech GN1825 octal TIA\n"
           "Stiller et al., Nature 2024 and Prum et al., Nature 2015 (bird phylogeny); RFC 1149 (1990-04-01)\n"
           "Form-factor dates: SFF/SNIA, OIF, MSA releases; XPO MSA (Arista, 2026)")
    ovt("SRC", src, i, 0, 2.0)
    ovt("FX", "[VO fast: Parody. Typical build. Not real birds.]", i, 0, 2.0)
    # S6 camera teleports to S7 card
    cam_jump(6, 0.0, (bw(hero.perch, hero.yaw, (0.42, 0.0, 0.01)), bw(hero.perch, hero.yaw, (0.2, 0.0, 0.0)), 50),
             ((0, -306, 0), (0, -300, 0), 28))
    return s


def build(out_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    # draft speed: 1 sample, 540x674 (about half of 1080x1350, even height for H.264), no shadows, persistent data between frames
    scn.eevee.taa_render_samples = int(os.environ.get("CRUDE_SAMPLES", "1"))
    scn.render.resolution_x = int(os.environ.get("CRUDE_W", "540"))
    scn.render.resolution_y = int(os.environ.get("CRUDE_H", "674"))
    scn.render.use_persistent_data = True
    for attr, val in (("use_shadows", False), ("use_raytracing", False), ("use_gtao", False)):
        if hasattr(scn.eevee, attr):
            setattr(scn.eevee, attr, val)
    scn.render.fps = 30
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.view_settings.view_transform = "Standard"
    scn.frame_start = 1
    scn.frame_end = TOTAL
    w = bpy.data.worlds.new("w")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.012, 0.014, 0.03, 1)
    bg.inputs["Strength"].default_value = 1.0
    scn.world = w
    L.setup_camera()
    L.CAM.data.clip_start = 0.005
    main_build()
    for i in range(6):
        for sec in range(10):
            L.ovt("TC", "S%d  0:%02d" % (i + 1, sec), i, sec, sec + 1)
    for sec in range(2):
        L.ovt("TC", "S7  0:%02d" % sec, 6, sec, sec + 1)
    L.finalize_visibility(TOTAL)
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("SAVED", out_path)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    build(argv[0] if argv else "crude_v0.blend")
