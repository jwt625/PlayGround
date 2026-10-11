"""Build the crude storyboard film v0.2 (storyboard rev 3, DevLog-006) as a single .blend.

Usage: blender -b --python scripts/build_crude_film_v0_2_1.py -- <out.blend>   (frozen copy of crude v0.2.1, 2026-10-10)
Crude on purpose: primitives, one continuous camera, captions, and [FX: ...] labels where effects will go.
World coordinates are absolute (no per-scene offset) so the camera can stay continuous:
  data hall: aisle along +y, rack faces at x = +-1.1; bird racks (1/3 of racks) with pigeonholes floor to top;
  facade (door + bird flap) at y = 13; night street beyond (S1 opening and S6 ending share it);
  TX-eye macro set around x = 200, RX-eye macro set around x = 400 (entered through color fills).
Birds: primitive pigeons (crowd, companions, street) in the hero's frame, one uniform scale S_BIRD; the hero itself is the
  hero GLB (workstream H) under a rig root rotated +90 deg about Z. Bird-local frame: origin on the ground between the feet,
  +x = beak, +z up, -y = pigeon's RIGHT (TX). Eyes are lateral (as in the hero GLB); right eye = TX, left eye = RX.
"""
import json
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
    "hood": (0.16, 0.14, 0.2), "rack": (0.07, 0.075, 0.085), "floor": (0.09, 0.09, 0.1), "cell": (0.03, 0.03, 0.035),
    "frame": (0.13, 0.135, 0.15), "yellow": (0.98, 0.83, 0.12),
}
# ------------------------------------------------------------------ bird geometry constants (mm, bird-local)
MM = 0.001
S_BIRD = 2.6                         # one hall scale for every bird: 124 mm hero -> 0.32 m beak to tail on screen
P_H = Vector((36.0, 0.0, 70.0))      # neck pivot (head yaw axis)
HC = Vector((44.0, 0.0, 81.5))       # head centre (hero profile: head centre ~44 mm ahead of the feet, ~81 mm up)
HR = 12.0
EA = 0.0                             # eye axis purely lateral (matches the hero GLB eye hooks)
AX = {"tx": Vector((math.sin(EA), -math.cos(EA), 0.0)), "rx": Vector((math.sin(EA), math.cos(EA), 0.0))}
UX = {"tx": Vector((math.cos(EA), math.sin(EA), 0.0)), "rx": Vector((math.cos(EA), -math.sin(EA), 0.0))}
ANG = {"tx": -(PI / 2 - EA), "rx": PI / 2 - EA}     # eye-axis bearing in the head frame
EYE_C = {"tx": Vector((47.5, -12.78, 80.5)) - P_H, "rx": Vector((47.5, 12.78, 80.5)) - P_H}  # hero GLB eye hooks, head-local
LENS_PITCH = 2.3
TOES = (("hallux", 172.0, 8.5), ("inner", -32.0, 12.0), ("middle", -4.0, 16.0), ("outer", 26.0, 12.5))  # hero (photo, C)
Z_CLAW, Z_PALM = -3.0, -1.2          # mating heights (mm below the toe pads): claws GND first, palm VCC, toe pads signal

# ------------------------------------------------------------------ hall layout (placeholder until workstream R delivers)
N_RACKS = 13
BIRD_RACKS = {("L", 1), ("L", 4), ("L", 7), ("L", 10), ("R", 2), ("R", 5), ("R", 7), ("R", 9), ("R", 12)}   # 9 of 26
N_ROWS = 7
ROW_PITCH = 0.29
ROW_FLOOR0 = 0.16                    # pigeonholes 0.20 x 0.27 m, floors 0.16 ... 1.90, rack top 2.30
COLS_DY = (-0.24, 0.0, 0.24)
HERO_CELL = ("L", 7, 4, 1)
ACROSS_CELL = ("R", 7, 4, 1)          # sends its beams into the hero's RX eye in S4
COMP_CELLS = [("L", 7, 5, 1), ("R", 7, 3, 1)]
FACE = 1.1                           # rack face |x|
DEPTH = 0.32                         # pigeonhole depth
FEET_IN = 0.06                       # birds stand 6 cm inside the face: head, neck and breast poke out
RACK_H = 2.3
CEIL = 4.2
TX_X, RX_X = 200.0, 400.0            # macro sets
FACADE_Y = 13.0
FLAP = Vector((0.0, FACADE_Y, 3.25))    # bird flap centre (R: entrance at (0, 12.94, 3.25), 0.7 x 0.5 m opening)


def row_floor(r):
    return ROW_FLOOR0 + ROW_PITCH * r


def lab(i, t0, t1, s): ovt("LAB", s, i, t0, t1)
def fx(i, t0, t1, s): ovt("FX", s, i, t0, t1)


def rack_y(k):
    return 1.4 + 0.8 * k


def cell_pos(side, k, row, col):
    """Feet origin of a bird standing in a pigeonhole (facing the aisle), and its yaw."""
    y = rack_y(k) + COLS_DY[col]
    z = row_floor(row)
    if side == "L":
        return Vector((-(FACE + FEET_IN), y, z)), 0.0
    return Vector((FACE + FEET_IN, y, z)), PI


# ------------------------------------------------------------------ transforms (bird-local mm -> world)
def rotz(a):
    return Matrix.Rotation(a, 3, "Z")


def body_rot(yaw, pitch=0.0):
    return rotz(yaw) @ Matrix.Rotation(pitch, 3, "Y")


def bwm(pos, yaw, local, pitch=0.0):
    return Vector(pos) + body_rot(yaw, pitch) @ (Vector(local) * (S_BIRD * MM))


def hwm(pos, yaw, hy, local, pitch=0.0):
    """Head-local mm (relative to the neck pivot) -> world, for head yaw hy."""
    return bwm(pos, yaw, P_H + rotz(hy) @ Vector(local), pitch)


def eye_w(pos, yaw, hy, k):
    return hwm(pos, yaw, hy, EYE_C[k] + AX[k] * 1.5)


def lens_local(k, r, c):
    return EYE_C[k] + AX[k] * 1.0 + UX[k] * ((c - 1.5) * LENS_PITCH) + Vector((0, 0, (r - 0.5) * LENS_PITCH * 1.05))


def axis_w(yaw, hy, k):
    return rotz(yaw + hy) @ AX[k]


def wrap(a):
    return (a + PI) % (2 * PI) - PI


def aim(pos, yaw, k, target, hy=0.0):
    """Head yaw that points eye k (axis in the horizontal plane) at target. Fixed point: the eye moves with the head."""
    for _ in range(5):
        d = Vector(target) - eye_w(pos, yaw, hy, k)
        hy = wrap(math.atan2(d.y, d.x) - yaw - ANG[k])
    return hy


def solve_link(a, b):
    """a = (pos, yaw) of the TX bird (right eye), b = (pos, yaw) of the RX bird (left eye). Returns head yaws."""
    ha = hb = 0.0
    for _ in range(4):
        ha = aim(a[0], a[1], "tx", eye_w(b[0], b[1], hb, "rx"), ha)
        hb = aim(b[0], b[1], "rx", eye_w(a[0], a[1], ha, "tx"), hb)
    return ha, hb


def stare(pos, yaw, target, lim=1.9):
    """One-eyed stare: whichever eye needs less head turn; clamped."""
    best = None
    for k in ("tx", "rx"):
        hy = aim(pos, yaw, k, target)
        if best is None or abs(hy) < abs(best):
            best = hy
    return max(-lim, min(lim, best))


# ------------------------------------------------------------------ generic helpers
def new_objs(fn, *a, **k):
    before = set(bpy.data.objects)
    r = fn(*a, **k)
    return r, [o for o in bpy.data.objects if o not in before]


def move_to(coll, objs):
    for o in objs:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)


def instancer(name, coll, loc, yaw, scale=1.0):
    o = bpy.data.objects.new(name, None)
    o.instance_type = "COLLECTION"
    o.instance_collection = coll
    o.location = loc
    o.rotation_euler = (0, 0, yaw)
    o.scale = (scale,) * 3
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


def _cam_key(f, pos, tgt, lens, interp):
    K(L.CAM, f, loc=tuple(pos), interp=interp)
    K(L.TGT, f, loc=tuple(tgt), interp=interp)
    L.CAM.data.lens = lens
    L.CAM.data.keyframe_insert("lens", frame=f)
    L._interp(L.CAM.data, "lens", f, interp)
    L.HOLD.scale = (L.HOLD_S0 * L.LENS0 / lens,) * 3
    L.HOLD.keyframe_insert("scale", frame=f)
    L._interp(L.HOLD, "scale", f, interp)


CAMLOG = []


def cam(idx, t, pos, tgt, lens=28.0, interp="BEZIER"):
    CAMLOG.append((F(idx, t), tuple(pos), tuple(tgt), lens))
    _cam_key(F(idx, t), pos, tgt, lens, interp)


def cam_jump(idx, t, prev, new):
    """Teleport (hidden by a fill or the S7 cut): hold `prev` until the frame before t, then `new` at t."""
    f = F(idx, t)
    _cam_key(f - 1, prev[0], prev[1], prev[2], "CONSTANT")
    _cam_key(f, new[0], new[1], new[2], "BEZIER")


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


# ------------------------------------------------------------------ mm primitives (parent space = bird-local metres)
def pmat(color, rough=0.4, metal=0.0, emit=0.0, coat=0.0, film=0.0):
    m = mat(color, rough, emit, unique=True)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Metallic"].default_value = metal
    if coat:
        b.inputs["Coat Weight"].default_value = coat
    if film and "Thin Film Thickness" in b.inputs:
        b.inputs["Thin Film Thickness"].default_value = film
        b.inputs["Thin Film IOR"].default_value = 1.6
    return m


def orient(o, xdir, up=(0, 0, 1)):
    x = Vector(xdir).normalized()
    y = Vector(up).cross(x)
    if y.length < 1e-6:
        y = Vector((0, 1, 0)).cross(x)
    y.normalize()
    z = x.cross(y)
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Matrix((x, y, z)).transposed().to_quaternion()
    return o


def _setm(o, m):
    if m is not None:
        o.data.materials[0] = m
    return o


GREY = (0.5, 0.5, 0.5)


def mbox(name, c, size, m, p, xdir=None, up=(0, 0, 1)):
    o = box(name, tuple(Vector(c) * MM), tuple(Vector(size) * MM), GREY, p)
    if xdir is not None:
        orient(o, xdir, up)
    return _setm(o, m)


def msph(name, c, r, m, p, scale=(1, 1, 1), xdir=None, up=(0, 0, 1), seg=16, rings=10):
    o = sph(name, tuple(Vector(c) * MM), r * MM, GREY, p, scale=scale, seg=seg, rings=rings)
    if xdir is not None:
        orient(o, xdir, up)
    return _setm(o, m)


def mseg(name, p0, p1, w, h, m, p, up=(0, 0, 1)):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    o = box(name, tuple((p0 + p1) / 2 * MM), (d.length * MM, w * MM, h * MM), GREY, p)
    orient(o, d, up)
    return _setm(o, m)


def mcone(name, p0, p1, r0, r1, m, p, verts=12):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    bpy.ops.mesh.primitive_cone_add(vertices=verts, radius1=r0 * MM, radius2=r1 * MM, depth=d.length * MM,
                                    location=tuple((p0 + p1) / 2 * MM))
    o = bpy.context.active_object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = d.to_track_quat("Z", "Y")
    o = L._fin(o, name, GREY, p, 0.5, 0.0, smooth=True)
    return _setm(o, m)


def mdisc(name, c, r, t, axis, m, p, verts=20):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r * MM, depth=t * MM, location=tuple(Vector(c) * MM))
    o = bpy.context.active_object
    o.rotation_mode = "QUATERNION"
    o.rotation_quaternion = Vector(axis).to_track_quat("Z", "Y")
    o = L._fin(o, name, GREY, p, 0.5, 0.0, smooth=True)
    return _setm(o, m)


def mcurve(name, pts, bevel, m, p):
    o = curve_line(name, [tuple(Vector(q) * MM) for q in pts], bevel * MM, GREY, parent=p)
    o.data.materials[0] = m
    return o


# ------------------------------------------------------------------ bird materials
BM = {}


def bird_mats():
    BM.update(
        body=pmat((0.10, 0.11, 0.13), 0.32, 0.75, coat=0.4),
        breast=pmat((0.15, 0.155, 0.17), 0.3, 0.7, coat=0.4),
        head=pmat((0.12, 0.125, 0.14), 0.25, 0.8, coat=0.6),
        neck=pmat((0.04, 0.30, 0.20), 0.22, 0.85, coat=0.6, film=520.0),
        neck2=pmat((0.22, 0.06, 0.26), 0.22, 0.85, coat=0.6, film=380.0),
        wing=pmat((0.20, 0.21, 0.24), 0.3, 0.8),
        fin=pmat((0.34, 0.35, 0.38), 0.28, 0.9),
        tail=pmat((0.22, 0.23, 0.26), 0.3, 0.85),
        tailband=pmat((0.03, 0.03, 0.035), 0.4, 0.5),
        accent=pmat(C["mag"], 0.3, 0.0, emit=4.0),
        beak=pmat((0.26, 0.26, 0.29), 0.25, 1.0, coat=0.8),
        cere=pmat((0.86, 0.85, 0.80), 0.45),
        leg=pmat((0.32, 0.07, 0.08), 0.35, 0.6),
        toe=pmat((0.18, 0.18, 0.2), 0.35, 0.8),
        pad_sig=pmat(C["gold"], 0.25, 1.0),
        pad_pwr=pmat(C["gold"], 0.25, 1.0),
        claw_gnd=pmat(C["brass"], 0.3, 1.0),
        band=pmat(C["orange"], 0.5, 0.0),
        leash=pmat(C["yellow"], 0.45, 0.0),
        rim_tx=pmat((0.7, 0.05, 0.04), 0.35, 0.3, emit=0.6),
        rim_rx=pmat((0.04, 0.55, 0.7), 0.35, 0.3, emit=0.6),
        iris=pmat((0.01, 0.01, 0.012), 0.2, 0.0),
        lid=pmat((0.2, 0.21, 0.24), 0.35, 0.8),
        eye_tx=pmat(C["red"], 0.1, 0.0, emit=6.0),
        eye_rx=pmat(C["cyan"], 0.1, 0.0, emit=6.0),
        pcb=pmat(C["pcb"], 0.5), chip=pmat(C["dark"], 0.4), tpad=pmat((0.7, 0.7, 0.72), 0.6),
        si=pmat(C["si"], 0.25, 0.3), gold=pmat(C["gold"], 0.3, 1.0), dfb=pmat(C["red"], 0.3, 0.0, emit=4.0),
        pd=pmat((0.16, 0.08, 0.2), 0.3), ind=pmat((0.3, 0.3, 0.32), 0.5, 0.5), flex=pmat((0.9, 0.5, 0.12), 0.4, 0.3),
        glass=pmat(C["glass"], 0.05, 0.0, emit=0.4),
    )


# ------------------------------------------------------------------ bird parts
def build_body(p, M, hero=False, strap="cell"):
    """Body parts in body-local mm under parent p. Returns (groups, feet objects by contact class)."""
    g = {}

    def grp(name, parent=p):
        e = empty("g_" + name, (0, 0, 0), parent=parent)
        g[name] = e
        return e

    shell = grp("shell")
    g["shell_objs"] = [
        msph("body", (-2, 0, 40), 1.0, M["body"], shell, scale=(36, 21, 20), xdir=(math.cos(0.42), 0, math.sin(0.42))),
        msph("breast", (19, 0, 43), 21, M["breast"], shell, scale=(1.0, 0.98, 1.08)),
        msph("neck_base", (28, 0, 57), 14, M["neck2"], shell),
        mcone("neck", (27, 0, 54), (37, 0, 73), 13, 8.5, M["neck"], shell, verts=16)]
    for s, nm in ((-1, "wing_r"), (1, "wing_l")):
        wg = grp(nm)
        sh_, tp = Vector((22, s * 18, 57)), Vector((-50, s * 11, 24))
        d = tp - sh_
        msph("wing_plate", (sh_ + tp) / 2, 1.0, M["wing"], wg, scale=(d.length / 2, 4.5, 12.5), xdir=d)
        for k in range(5):
            a = sh_.lerp(tp, 0.2 + 0.1 * k) + Vector((0, s * (4.6 - 0.3 * k), -(6 + 1.6 * k)))
            b = tp + Vector((-(4 - 1.5 * k), s * (3.0 - 0.3 * k), -(1 + 1.3 * k)))
            mseg("fin", a, b, 1.3, 5.5, M["fin"], wg)
        mseg("wing_glow", sh_ + Vector((-4, s * 4.9, 9)), tp + Vector((10, s * 4.4, 8)), 0.8, 1.0, M["accent"], wg)
    tg = grp("tail")
    for k in range(7):
        o = k - 3
        a = Vector((-26, o * 1.6, 31 + 0.15 * k))
        b = Vector((-66, o * 4.3, 8.5 - 0.5 * abs(o) + 0.15 * k))
        mseg("tail_f", a, b, 6.2, 1.1, M["tail"], tg)
        mseg("tail_band", a.lerp(b, 0.82), b + (b - a).normalized() * 0.4, 6.4, 1.3, M["tailband"], tg)
    legs = grp("legs")
    feet = grp("feet")
    fo = {"gnd": [], "pwr": [], "sig": []}
    for s in (-1, 1):
        ank = Vector((7, s * 8, 3.0))
        mcone("leg", (6, s * 7.5, 25), ank, 2.0, 1.7, M["leg"], legs, verts=10)
        msph("ankle", ank, 2.3, M["leg"], legs, seg=10, rings=6)
        base = Vector((7, s * 8, 1.4))
        for nm, ang, ln in TOES:
            a = math.radians(ang) * s
            dv = Vector((math.cos(a), math.sin(a), 0))
            tip = base + dv * ln
            mseg("toe", base, tip, 2.2, 1.6, M["toe"], feet)
            fo["sig"].append(mseg("toe_pad", base + dv * ln * 0.42 + Vector((0, 0, -1.05)), tip + Vector((0, 0, -1.05)),
                                  2.6, 0.7, M["pad_sig"], feet))
            fo["gnd"].append(mcone("claw", tip + Vector((0, 0, 0.2)), tip + dv * 2.6 + Vector((0, 0, Z_CLAW)), 0.9, 0.15,
                                   M["claw_gnd"], feet, verts=8))
        fo["pwr"].append(mdisc("palm_pad", (7, s * 8, Z_PALM + 0.4), 2.6, 0.8, (0, 0, 1), M["pad_pwr"], feet))
    for k in range(4):   # 4 low-speed pads on the belly, between the feet (I2C etc.)
        fo["pwr"].append(mbox("belly_pad", (16, (k - 1.5) * 2.4, 20.3), (3.2, 1.5, 0.6), M["pad_pwr"], feet))
    lg = grp("leash")   # pull tab out of the TAIL (owner decision 2026-10-10)
    if strap == "cell":   # docked: strap from the tail along the cell floor, over the lip, loop hanging below the face
        lip = FEET_IN / (S_BIRD * MM) + 3.0
        mseg("leash", (-58, 0, 8), (-40, 0, 1.0), 3.6, 0.7, M["leash"], lg)
        mseg("leash", (-40, 0, 1.0), (lip, 0, 1.0), 3.6, 0.7, M["leash"], lg)
        mseg("leash", (lip, 0, 1.0), (lip + 1, 0, -15), 3.6, 0.7, M["leash"], lg)
        L.torus("leash_loop", tuple(Vector((lip + 1, 0, -18.5)) * MM), 2.8 * MM, 0.7 * MM, GREY, rot=(PI / 2, 0, 0), parent=lg,
                material=M["leash"], segs=16)
    elif strap == "free":   # perched / flying: short tag behind the tail
        mseg("leash", (-60, 0, 9), (-74, 0, 3), 3.0, 0.6, M["leash"], lg)
        L.torus("leash_loop", tuple(Vector((-76.5, 0, 2.0)) * MM), 2.4 * MM, 0.6 * MM, GREY, rot=(PI / 2, 0, 0), parent=lg,
                material=M["leash"], segs=16)
    bg = grp("band")
    mdisc("leg_band", (6.6, -7.8, 13.0), 2.8, 3.2, (0, 0, 1), M["band"], bg, verts=16)
    if hero:
        ch = grp("chest")
        hero_parts = [
            mbox("sternum_pcb", (19, 0, 42), (16, 0.8, 18), M["pcb"], ch),
            mbox("pmic", (19, -1.6, 47), (4.5, 1.6, 4.5), M["chip"], ch),
            mbox("inductor", (18, -2.8, 39.5), (4.5, 3.4, 4.5), M["ind"], ch),
            mbox("load_switch", (13.5, -1.2, 44), (2.0, 1.0, 2.0), M["chip"], ch),
            mbox("eeprom", (22, 1.3, 38.5), (3.2, 1.0, 2.2), M["chip"], ch)]
        g["crystal_m"] = pmat((0.8, 0.8, 0.85), 0.2, 1.0)
        hero_parts.append(mbox("crystal", (21, 1.6, 48), (3.4, 1.2, 2.0), g["crystal_m"], ch))
        for k in range(6):
            hero_parts.append(mbox("mlcc", (13 + 2.4 * k, 1.0 + 0.3 * (k % 2), 34 + 2.5 * (k % 3)), (1.2, 0.8, 0.8), M["tpad"], ch))
        fl = grp("flex")
        for s in (-1, 1):
            hero_parts.append(mcurve("flex_neck_leg", [(39, s * 3, 80), (35, s * 3.5, 70), (30, s * 4, 60), (24, s * 4.5, 50),
                                                       (16, s * 5.5, 36), (8, s * 7, 24), (7, s * 8, 6)], 0.7, M["flex"], fl))
        hero_parts.append(mcurve("flex_power_i2c", [(19, 0.6, 33), (17, 0.4, 26), (16, 0, 21)], 0.6, M["flex"], fl))
        g["hero_parts"] = hero_parts
    return g, fo


def build_head(p, M, eye_tx, eye_rx, hero=False):
    """Head parts in head-local mm (relative to the neck pivot P_H) under parent p."""
    h = {}
    hc = HC - P_H
    skull = empty("g_skull", (0, 0, 0), parent=p)
    h["skull"] = skull
    h["skull_obj"] = msph("skull", hc, HR, M["head"], skull, scale=(1.05, 0.95, 1.0), seg=20, rings=12)
    msph("cere", Vector((53.2, 0, 81.0)) - P_H, 2.6, M["cere"], skull, scale=(1.4, 1.0, 0.75), seg=10, rings=6)
    mcone("beak_u", Vector((54, 0, 79.3)) - P_H, Vector((64.5, 0, 76.2)) - P_H, 3.0, 0.45, M["beak"], p, verts=10)
    hinge = empty("beak_hinge", tuple((Vector((53.5, 0, 77.4)) - P_H) * MM), parent=p)
    mcone("beak_l", (0.5, 0, -0.3), (8.8, 0, -2.0), 2.2, 0.35, M["beak"], hinge, verts=10)
    lids = []
    for k, em in (("tx", eye_tx), ("rx", eye_rx)):
        e, a = EYE_C[k], AX[k]
        mdisc("eye_rim", e - a * 0.2, 5.7, 1.4, a, M["rim_" + k], p)
        mdisc("eye_iris", e + a * 0.55, 4.9, 0.4, a, M["iris"], p)
        for r in range(2):
            for c in range(4):
                msph("lenslet", lens_local(k, r, c), 1.05, em, p, scale=(0.6, 1, 1), xdir=a, seg=10, rings=6)
        piv = empty("lid_pivot", tuple((e + a * 1.9 + Vector((0, 0, 6.0))) * MM), parent=p)
        mdisc("lid", (0, 0, -6.0), 6.1, 0.7, a, M["lid"], piv)
        piv.scale = (1, 1, 0.0)
        lids.append(piv)
    h["hinge"], h["lids"] = hinge, lids
    if hero:
        parts = []
        for nm in ("dsp", "tx_optics", "rx_optics", "mcu"):
            h[nm] = empty("g_" + nm, (0, 0, 0), parent=p)
        parts += [mbox("head_board", hc + Vector((0, 0, 2.6)), (15, 12, 0.8), M["pcb"], h["dsp"]),
                  mbox("dsp", hc + Vector((0, 0, 4.2)), (8.5, 8.5, 2.2), M["chip"], h["dsp"]),
                  mbox("thermal_pad", hc + Vector((0, 0, 5.8)), (8.5, 8.5, 0.9), M["tpad"], h["dsp"]),
                  mbox("mcu", hc + Vector((-3.0, 0, 1.3)), (3.5, 3.5, 1.1), M["chip"], h["mcu"])]
        for k, gname in (("tx", "tx_optics"), ("rx", "rx_optics")):
            e, a, u = EYE_C[k], AX[k], UX[k]
            gp = h[gname]
            parts.append(mbox("prism", e - a * 1.6, (6.5, 2.0, 4.0), M["glass"], gp, xdir=u))
            if k == "tx":
                parts.append(mbox("tx_pic", e - a * 3.8, (7.5, 0.9, 5.0), M["si"], gp, xdir=u))
                for j in range(8):
                    parts.append(mbox("mzm", e - a * 3.3 + u * ((j - 3.5) * 0.85), (0.35, 0.2, 3.6), M["gold"], gp, xdir=u))
                for dz in (-1.3, 1.3):
                    parts.append(mbox("dfb", e - a * 5.6 + Vector((0, 0, dz)), (1.6, 1.0, 0.8), M["dfb"], gp, xdir=u))
            else:
                parts.append(mbox("pd_array", e - a * 3.6, (7.5, 0.9, 3.2), M["pd"], gp, xdir=u))
                parts.append(mbox("tia", e - a * 5.4, (5.5, 1.2, 2.6), M["chip"], gp, xdir=u))
            parts.append(mcurve("flex_eye", [hc + Vector((0, 0, 3.0)), (e - a * 5.0) * 0.5 + (hc + Vector((0, 0, 3.0))) * 0.5,
                                             e - a * 4.6], 0.4, M["flex"], gp))
        h["hero_parts"] = parts
    return h


class Bird:
    pass


def make_bird(name, hero=False, strap="free"):
    b = Bird()
    b.root = empty(name, (0, -500, 0))
    b.root.scale = (S_BIRD,) * 3
    M = dict(BM)
    if hero:
        M["pad_sig"] = pmat(C["gold"], 0.25, 1.0)
        M["pad_pwr"] = pmat(C["gold"], 0.25, 1.0)
        M["claw_gnd"] = pmat(C["brass"], 0.3, 1.0)
        b.eye_tx = pmat(C["red"], 0.1, 0.0, emit=6.0)
        b.eye_rx = pmat(C["cyan"], 0.1, 0.0, emit=6.0)
    else:
        b.eye_tx, b.eye_rx = BM["eye_tx"], BM["eye_rx"]
    b.M = M
    b.groups, b.feet = build_body(b.root, M, hero=hero, strap=strap)
    b.neck = empty(name + "_neck", tuple(P_H * MM), parent=b.root)
    b.h = build_head(b.neck, M, b.eye_tx, b.eye_rx, hero=hero)
    b.hinge, b.lids = b.h["hinge"], b.h["lids"]
    return b


def bird_at(b, idx, t, pos, yaw=0.0, pitch=0.0, interp="BEZIER"):
    K(b.root, F(idx, t), loc=tuple(pos), rot=(0, pitch, yaw), interp=interp)


def head(b, idx, t, hy, roll=0.0, interp="BEZIER"):
    K(b.neck, F(idx, t), rot=(roll, 0, hy), interp=interp)


def beak(b, idx, t, open_amt, interp="BEZIER"):
    if getattr(b, "hinge_base", None) is not None:   # hero GLB: rotate the hinge hook about its local X (positive opens)
        r = b.hinge_base
        K(b.hinge, F(idx, t), rot=(r.x + 0.55 * open_amt, r.y, r.z), interp=interp)
    else:
        K(b.hinge, F(idx, t), rot=(0, 0.6 * open_amt, 0), interp=interp)


def blink(b, idx, t0, t1, which=(0, 1)):
    for k in which:
        lid = b.lids[k]
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


# ------------------------------------------------------------------ macro sets
def build_tx_macro():
    """TX eye interior at x ~ TX_X (light travels +x). Visible S3 1.8-8.8."""
    i, t0, t1 = 2, 1.8, 8.8
    X = TX_X
    objs = [box("pic", (X + 5, 0, -0.15), (10, 5, 0.3), C["si"], rough=0.3),
            box("macro_floor", (X + 6, 0, -0.6), (40, 30, 0.2), C["black"])]
    for yy in (-1.0, 1.0):
        objs.append(box("dfb_sub", (X - 0.45, yy, 0.05), (0.9, 0.5, 0.2), (0.85, 0.85, 0.8)))
        objs.append(box("dfb", (X - 0.4, yy, 0.27), (0.8, 0.3, 0.22), C["red"], emit=4.0))
    lanes = [s * (0.25 + 0.5 * k) for s in (-1, 1) for k in range(4)]
    wg = (0.85, 0.9, 1.0)
    for yy in (-1.0, 1.0):
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
        objs.append(curve_line("drv", [(X + 5.0, -3.6, 0.15), (X + 4.6, (ly - 3.6) / 2, 0.9), (X + 4.0, ly + 0.2, 0.05)], 0.02,
                               C["gold"], emit=0.3))
        for n in range(3):
            p = sph("pulse", (X, ly, 0.06), 0.07, (1.0, 0.3, 0.2), emit=8.0, seg=8, rings=6)
            for c in range(4):
                ta = 2.2 + c * 1.6 + n * 0.53
                if ta > t1:
                    break
                K(p, F(i, ta), loc=(X + 2.4, ly, 0.06), interp="LINEAR")
                K(p, F(i, min(ta + 1.6, t1)), loc=(X + 10.0, ly, 0.06), interp="LINEAR")
            objs.append(p)
    for f in range(F(i, 4.0), F(i, 8.8), 3):
        for m in pam:
            key_emit(m, f, C["red"], random.choice([0.2, 1.5, 3.5, 7.0]))
    objs.append(box("dsp_block", (X + 5.0, -4.1, 0.2), (3.0, 1.2, 0.6), C["dark"]))
    wlab("DSP (DRIVER INSIDE)", (X + 5.0, -4.1, 0.9), i, t0, t1, size=0.2, color=C["gold"])
    # light from the edge couplers into the prism (1x8 row), out of the prism as the 2x4 grid
    for k, ly in enumerate(lanes):
        objs.append(beam("tx_row", (X + 10.0, ly, 0.03), (X + 10.6, ly, 0.03), 0.04, C["red"], emit=3.0))
        r, c = k // 4, k % 4
        objs.append(beam("tx_regroup", (X + 11.6, -1.2 + 0.8 * c, 0.0 + 0.8 * r), (X + 12.3, -1.2 + 0.8 * c, 0.0 + 0.8 * r),
                         0.06, C["red"], emit=3.0))
    pr = box("prism_tx", (X + 11.1, 0, 0.4), (1.0, 4.2, 1.6), C["glass"], rough=0.05)
    pm = mat(C["glass"], 0.05, 0.4, unique=True)
    pm.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.3
    set_blend(pm)
    pr.data.materials[0] = pm
    objs.append(pr)
    objs.append(box("lens_frame", (X + 12.42, 0, 0.4), (0.08, 3.6, 0.12), C["gun2"]))
    lm = mat((1.0, 0.12, 0.08), 0.05, 2.5, unique=True)
    for r in range(2):
        for c in range(4):
            yy, zz = -1.2 + 0.8 * c, 0.0 + 0.8 * r
            ln = sph("lenslet_m", (X + 12.5, yy, zz), 0.33, C["glass"], scale=(0.5, 1, 1), rough=0.05)
            ln.data.materials[0] = lm
            objs.append(ln)
            objs.append(cyl("lens_ring", (X + 12.45, yy, zz), 0.37, 0.08, C["gun2"], axis="X", verts=20))
            objs.append(beam("tx_beam", (X + 12.7, yy, zz), (X + 40, yy, zz), 0.12, C["red"], emit=4.0))
    objs.append(box("edge_couplers", (X + 9.95, 0, 0.03), (0.1, 4.0, 0.06), (0.9, 0.9, 1.0), emit=2.0))
    lo = add_light("tx_area", "AREA", (X + 6, 0, 6), 1200, size=12)
    lo2 = add_light("tx_lens_light", "AREA", (X + 14, -4, 3), 900, size=4, rot=(1.0, 0, -0.9))
    for o in objs:
        V(o, i, t0, t1)
    wl = [("CW DFB LASER x2", (X - 0.4, 0, 0.9), 2.0, 5.2), ("SPLITTER TREE", (X + 1.4, 0, 0.6), 3.0, 5.5),
          ("8 MACH-ZEHNDER MODULATORS", (X + 5.2, 2.5, 0.8), 4.5, 7.4), ("MONITOR PDs", (X + 7.4, 2.4, 0.5), 5.0, 7.4),
          ("EDGE COUPLERS", (X + 9.9, -2.4, 0.5), 6.6, 7.6), ("PRISM: 1x8 ROW -> 2x4", (X + 11.1, -1.2, 1.55), 7.0, 8.4),
          ("2x4 LENSLET ARRAY", (X + 12.9, -1.4, 1.35), 7.5, 8.5)]
    for s, p, a, b in wl:
        wlab(s, p, i, a, b, size=0.18, color=(0.75, 0.95, 1.0))


def build_rx_macro():
    i, t0, t1 = 3, 1.8, 6.8
    X = RX_X
    objs = [box("macro_floor_rx", (X + 4, 0, -0.6), (40, 30, 0.2), C["black"])]
    objs.append(box("lens_frame_rx", (X + 0.05, 0, 0.4), (0.08, 3.6, 0.12), C["gun2"]))
    lm = mat((0.5, 0.95, 1.0), 0.05, 2.0, unique=True)
    for r in range(2):
        for c in range(4):
            yy, zz = -1.2 + 0.8 * c, 0.0 + 0.8 * r
            ln = sph("lenslet_rx", (X - 0.1, yy, zz), 0.32, C["glass"], scale=(0.5, 1, 1), rough=0.05)
            ln.data.materials[0] = lm
            objs.append(ln)
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
    add_light("rx_area", "AREA", (X + 3, 0, 6), 1000, size=10)
    for o in objs:
        V(o, i, t0, t1)
    for s, p, a, b in [("2x4 LENSLET ARRAY", (X, 2.4, 1.5), 1.9, 3.6), ("PRISM: 2x4 -> 1x8 ROW", (X + 1.6, 2.6, 1.3), 2.2, 3.8),
                       ("8 PIN PHOTODIODES", (X + 3.0, 2.6, 1.2), 2.6, 4.8), ("OCTAL TIA", (X + 5.0, 2.4, 0.6), 4.4, 6.6),
                       ("TO THE DSP", (X + 9.0, 1.4, 0.8), 5.2, 6.6)]:
        wlab(s, p, i, a, b, size=0.18, color=(0.75, 0.95, 1.0))


def build_hatchery():
    """Assembly line that fades in behind the hero during S5 8.2-9.95 (aisle, y ~ 8.6-10.1)."""
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
                                           ("p_shell", shell_m, (0.18, 0.08, 0.07), 0.07, 3), ("p_eye", eye_m, (0.02, 0.02, 0.02), 0.1, 4)):
            if k < lvl:
                continue
            o = sph(part, (x, 9.2, 1.28 + dz), 1.0, (0.5, 0.5, 0.5), scale=tuple(v / 2 for v in size), seg=10, rings=6) \
                if part in ("p_shell", "p_eye") else box(part, (x, 9.2, 1.28 + dz), size, (0.5, 0.5, 0.5))
            o.data.materials[0] = col_m
            if part == "p_eye":
                o.location.x += 0.08
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


# ------------------------------------------------------------------ visibility windows
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ------------------------------------------------------------------ hall layout from workstream R (DevLog-007)
DC_DIR = os.path.join(ROOT_DIR, "assets", "components", "datacenter")
LAYOUT_DY = 1.0          # R's layout shifted +1 m in y so the hero rack L07 sits at y = 7.0 (v0.1 framing of S2-S5)
LAYOUT = json.load(open(os.path.join(DC_DIR, "hall_layout.json"))) if os.path.exists(os.path.join(DC_DIR, "hall_layout.json")) else None
CELLS = {}
if LAYOUT:
    for _c in LAYOUT["cells"]["list"]:
        CELLS[(_c["rack"], _c["hook"])] = (Vector(_c["feet_origin"]) + Vector((0, LAYOUT_DY, 0)), math.radians(_c["bird_yaw_deg"]))
    HERO_CELL = ("L07", "HOOK_cell_1_1_1")        # unit 1 lower row, middle column (feet z 1.47 m)
    ACROSS_CELL = ("R07", "HOOK_cell_1_1_1")
    COMP_CELLS = [("L07", "HOOK_cell_1_0_1"), ("R07", "HOOK_cell_2_0_1")]
    PADS = {pd["rack"]: Vector(pd["location"]) + Vector((0, LAYOUT_DY, 0)) for pd in LAYOUT["landing_pads"]}

    def cell_pos(*cell):
        pos, yaw = CELLS[tuple(cell)]
        return pos.copy(), yaw


def append_collection(blend, name):
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        dst.collections = [n for n in src.collections if n == name]
    return bpy.data.collections.get(name)
HALL_VIS = [(0, 0, 10), (1, 0, 10), (2, 0, 10), (3, 0, 10), (4, 0, 10), (5, 0, 4.0)]
STREET_VIS = [(0, 0, 10), (4, 8.0, 10), (5, 0, 10)]
L.OVL["NOTE"] = dict(size=0.15, color=(1.0, 0.85, 0.3), loc=(-1.85, -2.12), ax="LEFT", ay="BOTTOM", shadow=True,
                     font="Arial Bold", outline=0.009)
L.WRAP["NOTE"] = 60
BEAM_NOTE = "* beams invisible (1311 nm); rendered for viewing"


def vis(o, wins):
    for i, a, b in wins:
        V(o, i, a, b)
    return o


def note(i, t0, t1):
    ovt("NOTE", BEAM_NOTE, i, t0, t1)


def key_glow(m, frame, color, strength, interp="LINEAR"):
    """Keyframe emission colour + strength only (keeps the base colour)."""
    b = m.node_tree.nodes.get("Principled BSDF")
    if b is None:
        return
    b.inputs["Emission Color"].default_value = (*color, 1)
    b.inputs["Emission Strength"].default_value = strength
    b.inputs["Emission Color"].keyframe_insert("default_value", frame=frame)
    b.inputs["Emission Strength"].keyframe_insert("default_value", frame=frame)
    for fc in m.node_tree.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            if int(round(kp.co[0])) == frame:
                kp.interpolation = interp


# ------------------------------------------------------------------ hall (placeholder racks until workstream R)
def build_hall():
    objs = []
    objs.append(plane("hall_floor", (0, 6.1, 0), (6.4, 13.8), mat(C["floor"], 0.4), rot=(0, 0, 0)))
    objs.append(plane("hall_ceiling", (0, 6.1, CEIL), (6.4, 13.8), mat(C["black"], 0.9), rot=(PI, 0, 0)))
    for k in range(4):
        objs.append(box("tile_line", (-0.5 + k * 0.33, 6.0, 0.001), (0.008, 13.0, 0.001), (0.15, 0.15, 0.17)))
    objs.append(box("wall_back", (0, -0.8, CEIL / 2), (6.4, 0.1, CEIL), C["black"]))
    for sx in (-1, 1):
        objs.append(box("wall_side", (sx * 3.2, 6.1, CEIL / 2), (0.1, 13.8, CEIL), C["black"]))
    objs.append(box("door_frame_in", (0, FACADE_Y - 0.07, 2.52), (1.5, 0.04, 0.04), C["cyan"], emit=4.0))
    for k in range(4):
        objs.append(box("ceil_strip", (0, 1.5 + 3.2 * k, CEIL - 0.05), (0.12, 2.2, 0.03), (0.85, 0.9, 1.0), emit=3.0))
    racks = {}
    if LAYOUT:
        cols = {}
        for tname, td in LAYOUT["rack_types"].items():
            cols[tname] = append_collection(os.path.join(ROOT_DIR, td["asset"]), td["collection"])
        for r in LAYOUT["racks"]:
            loc = Vector(r["location"]) + Vector((0, LAYOUT_DY, 0))
            o = instancer("rack_" + r["id"], cols[r["type"]], loc, math.radians(r["rotation_z_deg"]))
            racks[r["id"]] = [o]
            objs.append(o)
    else:
        for side, sx in (("L", -1), ("R", 1)):
            for k in range(N_RACKS):
                parts = build_bird_rack(side, k) if (side, k) in BIRD_RACKS else build_compute_rack(side, k)
                racks[(side, k)] = parts
                objs += parts
    for o in objs:
        vis(o, HALL_VIS)
    for k in range(4):
        ld = bpy.data.lights.new("hall_area", "AREA")
        ld.shape = "RECTANGLE"
        ld.size, ld.size_y = 0.6, 2.6
        ld.energy = 140
        ld.color = (0.8, 0.88, 1.0)
        lo = bpy.data.objects.new("hall_area", ld)
        bpy.context.scene.collection.objects.link(lo)
        lo.location = (0, 1.5 + 3.2 * k, CEIL - 0.15)
    for x, y, col in [(-1.0, 3.0, C["mag"]), (1.0, 6.0, C["cyan"]), (-1.0, 9.0, C["cyan"]), (1.0, 11.5, C["mag"])]:
        add_light("accent", "POINT", (x, y, 2.6), 40, col)
    for sx in (-1, 1):
        add_light("face_fill", "AREA", (sx * 0.3, 4.0, 2.4), 30, (0.75, 0.8, 1.0), size=1.2, rot=(0, -sx * PI / 2, 0))
    return racks


PAD_M = {}


def build_bird_rack(side, k):
    """Switch-style bird rack: pigeonholes from the plinth to the top, landing pad on top (placeholder)."""
    sx = -1 if side == "L" else 1
    y = rack_y(k)
    xc = FACE + DEPTH / 2
    top = row_floor(N_ROWS) + 0.015
    p = [box("brack_lo", (sx * 1.6, y, (ROW_FLOOR0 - 0.03) / 2), (1.0, 0.78, ROW_FLOOR0 - 0.03), C["rack"], rough=0.6),
         box("brack_hi", (sx * 1.6, y, (top + RACK_H) / 2), (1.0, 0.78, RACK_H - top), C["rack"], rough=0.6),
         box("brack_back", (sx * (FACE + DEPTH + (2.1 - FACE - DEPTH) / 2), y, (ROW_FLOOR0 + top) / 2),
             (2.1 - FACE - DEPTH, 0.78, top - ROW_FLOOR0), C["cell"], rough=0.8)]
    for r in range(N_ROWS + 1):
        p.append(box("shelf", (sx * xc, y, row_floor(r) - 0.015), (DEPTH, 0.78, 0.03), C["frame"], rough=0.5))
    for dy, w in ((-0.355, 0.03), (-0.12, 0.04), (0.12, 0.04), (0.355, 0.03)):
        p.append(box("divider", (sx * xc, y + dy, (ROW_FLOOR0 + top) / 2), (DEPTH, w, top - ROW_FLOOR0), C["frame"], rough=0.5))
    for r in range(N_ROWS):
        p.append(box("cell_led", (sx * (FACE - 0.004), y + 0.37, row_floor(r) + 0.2), (0.006, 0.012, 0.03),
                     (0.2, 1.0, 0.4) if (r + k) % 3 else C["cyan"], emit=3.0))
    pm = mat(C["cyan"], 0.3, 1.5, unique=True)
    PAD_M[(side, k)] = pm
    pad = box("landing_pad", (sx * (FACE - 0.06), y, RACK_H + 0.0125), (0.62, 0.7, 0.025), (0.2, 0.21, 0.24), rough=0.4)
    glow = box("pad_glow", (sx * (FACE - 0.36), y, RACK_H + 0.026), (0.02, 0.66, 0.004), C["cyan"])
    glow.data.materials[0] = pm
    p += [pad, glow, box("pad_post", (sx * (FACE - 0.3), y, RACK_H + 0.03), (0.02, 0.02, 0.04), (0.3, 0.3, 0.32))]
    p.append(box("brand_strip", (sx * (FACE - 0.003), y, RACK_H - 0.05), (0.004, 0.7, 0.03), (0.6, 1.0, 0.2), emit=1.5))
    return p


def build_compute_rack(side, k):
    """GB300-style compute rack placeholder: tray faces, LED dots, switch trays in the middle, cable bundles."""
    sx = -1 if side == "L" else 1
    y = rack_y(k)
    fxp = sx * (FACE - 0.006)
    p = [box("crack", (sx * 1.6, y, RACK_H / 2), (1.0, 0.78, RACK_H), (0.05, 0.05, 0.06), rough=0.6)]
    z = 0.12
    layout = [("power", 0.14)] + [("compute", 0.082)] * 8 + [("switch", 0.044)] * 9 + [("compute", 0.082)] * 10 + [("power", 0.12)]
    for kind, h in layout:
        col = {"power": (0.2, 0.2, 0.22), "compute": (0.11, 0.115, 0.13), "switch": (0.07, 0.075, 0.09)}[kind]
        p.append(box("tray", (fxp, y, z + h / 2), (0.012, 0.72, h - 0.006), col, rough=0.5))
        led = {"power": (0.3, 0.6, 1.0), "compute": (0.3, 1.0, 0.45), "switch": C["cyan"]}[kind]
        p.append(box("tray_led", (sx * (FACE - 0.013), y - 0.32, z + h / 2), (0.003, 0.012, 0.008), led, emit=4.0))
        z += h
    for dy in (-0.27, 0.27):
        c = cyl("cable_bundle", (sx * (FACE - 0.035), y + dy, 1.2), 0.025, 1.8, (0.02, 0.02, 0.025), verts=10)
        p.append(c)
    return p


# ------------------------------------------------------------------ TX / RX subassemblies (workstreams T / X), scaled x1000
SUB_K = 1000.0                       # 1 mm of the true-scale subassembly -> 1 m in the macro set
TX_GLB = os.path.join(ROOT_DIR, "assets", "components", "optics_tx", "optics_tx.glb")
RX_GLB = os.path.join(ROOT_DIR, "assets", "components", "optics_rx", "optics_rx.glb")
RX_Z0 = 1.45                         # RX carrier bottom (-1.83 mm) sits near the macro floor


def import_glb_root(path, loc, scale):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    root = empty("sub_root", loc)
    root.scale = (scale,) * 3
    for o in new:
        if o.parent is None:
            o.parent = root
    return root, new


def W_tx(mm):
    return Vector((TX_X + mm[0], mm[1], mm[2]))


def W_rx(mm):
    return Vector((RX_X + mm[0], mm[1], RX_Z0 + mm[2]))


def build_tx_sub():
    """T subassembly (origin = eye-plate outer surface centre, +X = light out) at TX_X, x1000. Visible S3 1.8-8.8."""
    i, t0, t1 = 2, 1.8, 8.8
    _, new = import_glb_root(TX_GLB, (TX_X, 0, 0), SUB_K)
    for o in new:
        V(o, i, t0, t1)
    meta = json.load(open(TX_GLB.replace(".glb", "_parts.json")))
    extra = []
    for ln in meta.get("lane_map", []):
        extra.append(beam("tx_out_beam", W_tx((0.12, ln["y_out"], ln["z_out"])), W_tx((25.0, ln["y_out"], ln["z_out"])), 0.05, C["red"],
                          emit=4.0))
    for o in extra:
        V(o, i, 7.3, t1)
    for lo in (add_light("tx_sub_front", "AREA", tuple(W_tx((8, -4, 0))), 2500, size=14, rot=(0, PI / 2, 0)),
               add_light("tx_sub_top", "AREA", tuple(W_tx((-1, -4, 8))), 900, size=12)):
        V(lo, i, t0, t1)
    for txt, p, a, b in (("CW DFB LASER x2", (-1.2, 0.5, -3.3), 2.4, 5.0), ("SI MICROLENSES (DFB -> PIC)", (-1.0, -1.1, -1.7), 3.0, 5.0),
                         ("SIPH PIC: 8 MACH-ZEHNDER MODULATORS", (-1.2, -6.0, 3.9), 4.6, 7.2),
                         ("HS DIFFERENTIAL WIREBONDS (GSSG)", (-1.3, -9.8, 3.9), 5.6, 7.0),
                         ("FAU COLLIMATOR + STAIRCASE PRISM\n(1x8 -> 2x4)", (-1.0, -1.8, 1.6), 7.0, 8.4),
                         ("2x4 LENSLET EYE", (0.4, 0.0, 1.7), 7.6, 8.5)):
        wlab(txt, W_tx(p), i, a, b, size=0.16, color=(0.75, 0.95, 1.0))


def build_rx_sub():
    """X subassembly (origin = lenslet front surface centre, light arrives along -X) at RX_X, x1000. Visible S4 1.8-6.8."""
    i, t0, t1 = 3, 1.8, 6.8
    _, new = import_glb_root(RX_GLB, (RX_X, 0, RX_Z0), SUB_K)
    for o in new:
        V(o, i, t0, t1)
    extra = []
    for zr, dy in ((0.425, -0.25), (-0.425, 0.25)):   # rows; the Si interleaver plates shift the rows by -+0.25 mm onto the 0.5 mm PD pitch
        for y in (-1.5, -0.5, 0.5, 1.5):
            yl = y + dy
            extra.append(beam("rx_in", W_rx((25.0, y, zr)), W_rx((0.05, y, zr)), 0.05, C["red"], emit=4.0))
            extra.append(beam("rx_int", W_rx((-0.5, y, zr)), W_rx((-3.95, yl, zr)), 0.03, C["red"], emit=3.0))
            extra.append(beam("rx_down", W_rx((-3.95, yl, zr)), W_rx((-3.93, yl, -1.2)), 0.03, C["red"], emit=3.0))
    for o in extra:
        V(o, i, t0, t1)
    for lo in (add_light("rx_sub_top", "AREA", tuple(W_rx((-3, 0, 6))), 3000, size=10),
               add_light("rx_sub_front", "AREA", tuple(W_rx((6, -3, 2))), 700, size=8, rot=(0, PI / 2, 0))):
        V(lo, i, t0, t1)
    for txt, p, a, b in (("2x4 LENSLET ARRAY", (0.0, 0.0, 1.3), 1.9, 3.6), ("SI INTERLEAVER PLATES: 2x4 -> 1x8", (-2.1, 0.0, 1.2), 2.4, 4.2),
                         ("45 DEG COMB MIRROR", (-4.05, 0.0, 1.0), 3.6, 4.8), ("1x8 PIN PD ARRAY", (-3.5, -2.4, -0.6), 3.8, 5.2),
                         ("OCTAL TIA + SPIRAL INDUCTORS", (-4.8, 0.0, -0.4), 4.6, 6.4), ("GSG BONDS PD -> TIA", (-4.1, 2.2, -0.6), 4.6, 6.0)):
        wlab(txt, W_rx(p), i, a, b, size=0.15, color=(0.75, 0.95, 1.0))


# ------------------------------------------------------------------ facade (datacenter end wall, seen from the street)
FLAPS = []


def build_facade():
    fy = FACADE_Y
    x0f, x1f = FLAP.x - 0.35, FLAP.x + 0.35
    z0f, z1f = FLAP.z - 0.25, FLAP.z + 0.25
    objs = []
    for xa, xb, za, zb in ((-6.0, -0.7, 0, 7.0), (0.7, 6.0, 0, 7.0), (-0.7, x0f, 2.5, 7.0), (x1f, 0.7, 2.5, 7.0), (x0f, x1f, 2.5, z0f),
                           (x0f, x1f, z1f, 7.0)):
        objs.append(box("facade", ((xa + xb) / 2, fy, (za + zb) / 2), (xb - xa, 0.12, zb - za), (0.11, 0.115, 0.13), rough=0.7))
    objs.append(box("door_frame_out", (0, fy + 0.07, 2.52), (1.5, 0.03, 0.04), C["cyan"], emit=4.0))
    for sx in (-1, 1):
        objs.append(box("door_jamb", (sx * 0.72, fy + 0.07, 1.25), (0.03, 0.03, 2.5), C["cyan"], emit=2.0))
        objs.append(box("wall_lamp", (sx * 1.4, fy + 0.09, 2.7), (0.2, 0.06, 0.1), (1.0, 0.9, 0.7), emit=5.0))
        for kk in range(3):
            objs.append(box("vent", (sx * (3.0 + 0.9 * kk), fy + 0.065, 4.6), (0.7, 0.01, 0.45), (0.2, 0.2, 0.22), rough=0.4))
    for zz in (1.2, 2.0, 5.6, 6.4):
        objs.append(box("facade_band", (0, fy + 0.062, zz), (12.0, 0.005, 0.03), (0.25, 0.27, 0.3)))
    objs.append(_text("DOVECOTE DC", 0.45, C["cyan"], (2.6, fy + 0.08, 3.6), None, "CENTER", "CENTER", rot=(PI / 2, 0, PI),
                      fnt="Arial Black"))
    vent_path = os.path.join(DC_DIR, "bird_flap_vent.blend")
    if os.path.exists(vent_path):   # workstream R bird flap vent: origin = opening centre on the inner wall face, +X into the hall
        vc = append_collection(vent_path, "ASSET_bird_flap_vent")
        root = empty("bird_flap_root", (FLAP.x, fy - 0.06, FLAP.z))
        root.rotation_euler = (0, 0, -PI / 2)
        for o in vc.all_objects:
            if o.parent is None:
                o.parent = root
            if o.name.startswith("bird_flap_vent_flap"):
                o.rotation_mode = "XYZ"
                FLAPS.append(o)
        bpy.context.scene.collection.children.link(vc)
    return objs


# ------------------------------------------------------------------ heads: random motion, eye contact, tracking
class Head:
    """Head of a crowd instance (instancer object) or of a rig (neck empty); keys yaw h, roll and a forward bob."""

    def __init__(self, pos, yaw, inst=None, rig=None):
        self.pos, self.yaw, self.inst, self.rig = Vector(pos), yaw, inst, rig
        self.cur = (0.0, 0.0, 0.0)
        self.busy = -1
        self.blocked = []

    def key(self, f, h, roll=0.0, bob=0.0, interp="BEZIER"):
        if self.inst is not None:
            fwd = Vector((math.cos(self.yaw), math.sin(self.yaw), 0))
            K(self.inst, f, loc=tuple(bwm(self.pos, self.yaw, P_H) + fwd * bob), rot=(roll, 0, self.yaw + h), interp=interp)
        else:
            K(self.rig.neck, f, loc=tuple(P_H * MM + Vector((bob / S_BIRD, 0, 0))), rot=(roll, 0, h), interp=interp)
        self.cur = (h, roll, bob)

    def move(self, f, h, roll=0.0, bob=0.0, dur=5):
        self.key(f, *self.cur)
        self.key(f + dur, h, roll, bob)

    def is_blocked(self, f):
        return any(a <= f <= b for a, b in self.blocked)


def animate_heads(heads, f0, f1, rng, p_contact=0.09, p_move=0.5, radius=3.2):
    """Constant pigeon-like head motion with eye-contact events (TX right eye -> a neighbour's RX left eye)."""
    nb = []
    for i, a in enumerate(heads):
        nb.append([j for j, b in enumerate(heads) if j != i and (a.pos - b.pos).length < radius])
    for hd in heads:
        if not hd.is_blocked(f0):
            hd.key(f0, *hd.cur)
        hd.busy = f0
    f = f0
    while f < f1 - 8:
        order = list(range(len(heads)))
        rng.shuffle(order)
        for i in order:
            A = heads[i]
            if A.busy > f or A.is_blocked(f) or A.is_blocked(f + 8):
                continue
            r = rng.random()
            if r < p_contact and nb[i]:
                j = rng.choice(nb[i])
                B = heads[j]
                if B.busy <= f and not B.is_blocked(f) and not B.is_blocked(f + 50):
                    ha, hb = solve_link((A.pos, A.yaw), (B.pos, B.yaw))
                    if abs(ha) < 2.3 and abs(hb) < 2.3:
                        hold = rng.randint(24, 48)
                        A.move(f, ha, 0, 0, 6)
                        B.move(f, hb, 0, 0, 6)
                        A.key(f + 6 + hold, ha)
                        B.key(f + 6 + hold, hb)
                        A.busy = B.busy = f + 6 + hold
                        continue
            if r < p_contact + p_move:
                A.move(f, rng.uniform(-1.2, 1.2), rng.uniform(-0.35, 0.35), rng.choice((0.0, 0.0, 0.009, -0.006)), rng.randint(3, 6))
                A.busy = f + rng.randint(8, 30)
        f += 6


def track_heads(heads, track, target_z=1.55):
    for hd in heads:
        for idx, t, sp in track:
            f = F(idx, t)
            if hd.is_blocked(f):
                continue
            hd.key(f, stare(hd.pos, hd.yaw, Vector((sp[0], sp[1], target_z))), 0.0, 0.0)


# ------------------------------------------------------------------ crowd (bird racks, filled floor to top)
def bird_collections():
    cols = {}
    for name, strap in (("CROWD_BODY", "cell"), ("FREE_BODY", "free")):
        c = bpy.data.collections.new(name)
        root = empty(name.lower() + "_root")
        _, ob = new_objs(build_body, root, BM, False, strap)
        move_to(c, [root] + ob)
        cols[name] = c
    ch = bpy.data.collections.new("CROWD_HEAD")
    hroot = empty("crowd_head_root")
    _, oh = new_objs(build_head, hroot, BM, BM["eye_tx"], BM["eye_rx"])
    move_to(ch, [hroot] + oh)
    cols["CROWD_HEAD"] = ch
    return cols


def build_crowd(cols):
    skip = {HERO_CELL, *COMP_CELLS}
    heads, cells = [], {}
    for cell in sorted(CELLS):
        if cell in skip:
            continue
        pos, yaw = cell_pos(*cell)
        bi = instancer("cb", cols["CROWD_BODY"], pos, yaw, S_BIRD)
        hi = instancer("ch", cols["CROWD_HEAD"], bwm(pos, yaw, P_H), yaw, S_BIRD)
        for o in (bi, hi):
            if cell[0] == HERO_CELL[0]:   # hidden during the S2 cutaway of the hero rack
                V(o, 0, 0, 10); V(o, 1, 0, 6.8); V(o, 2, 0.8, 10); V(o, 3, 0, 10); V(o, 4, 0, 10); V(o, 5, 0, 4.0)
            else:
                vis(o, HALL_VIS)
        hd = Head(pos, yaw, inst=hi)
        heads.append(hd)
        cells[cell] = hd
    return heads, cells


# ------------------------------------------------------------------ tree hologram (no years; bird tree + pluggable lineage)
TREE = [  # (bird, pluggable, x, extinct, branch): branch L = pluggable lineage (numbered), C = coherent, None = other
    ("THEROPOD", "(0) 1x9 SOLDERED", 0.0, False, "L"),
    ("ARCHAEOPTERYX", "(1) GBIC: 1 LANE", -0.42, False, "L"),
    ("MOA", "XENPAK", 0.42, True, None),
    ("ELEPHANT BIRD", "CFP", -0.42, True, "C"),
    ("EMU", "CFP2-DCO", 0.42, False, "C"),
    ("OSTRICH", "(6) XPO: 64 LANES", -0.42, False, "L"),
    ("CHICKEN", "(2) SFP: 1 LANE", 0.42, False, "L"),
    ("HUMMINGBIRD", "(5) QSFP-DD: 8 LANES", -0.42, False, "L"),
    ("ALBATROSS", "400ZR", 0.42, False, "C"),
    ("PARROT", "COMPATIBLES", -0.42, False, None),
    ("CUCKOO", "AEC: COPPER EGG", 0.42, False, "U"),
    ("PASSENGER PIGEON", "(3) QSFP+: 4 LANES", -0.42, True, "L"),
    ("ROCK PIGEON", "(4) QSFP28: 4 LANES", 0.42, False, "L"),
    ("DODO", "CPO: CO-PACKAGED", -0.42, True, "P"),
]
CLADES = [("PALAEOGNATHS", 2, 5), ("GALLOANSERAE", 6, 6), ("NEOAVES (UNRESOLVED)", 7, 9), ("COLUMBAVES", 10, 13)]
TREE_C = Vector((0.0, 17.0, 0.0))
TZ0, TDZ = 0.55, 0.2
NODE_Z = TZ0 + TDZ * len(TREE)
BRANCH_COL = {"L": C["gold"], "C": C["mag"], "U": C["orange"], "P": (0.6, 0.65, 0.7), None: (0.75, 0.95, 1.0)}


def build_tree():
    hm = mat(C["holo"], 0.5, 2.5, unique=True)
    gm = mat((0.5, 0.6, 0.7), 0.5, 1.2, unique=True)
    nm = mat(C["red"], 0.3, 8.0, unique=True)
    lm = mat(C["gold"], 0.4, 2.0, unique=True)
    cm = mat(C["mag"], 0.4, 2.0, unique=True)
    ty = TREE_C.y
    objs = []
    trunk = cyl("trunk", (0, ty, (0.4 + NODE_Z) / 2), 0.012, NODE_Z - 0.4, C["holo"], verts=8)
    trunk.data.materials[0] = hm
    objs.append(trunk)
    for k, (bird, plug, x, ext, br_kind) in enumerate(TREE):
        z = TZ0 + TDZ * k
        m = gm if ext else hm
        if k > 0:
            br = beam("branch", (0, ty, z - 0.08), (x, ty, z), 0.007, C["holo"], emit=2.0)
            br.data.materials[0] = lm if br_kind == "L" else cm if br_kind == "C" else m
            objs.append(br)
        for o in (sph("leaf", (x, ty, z), 0.035, C["holo"], scale=(1.4, 1, 0.9), seg=12, rings=8),
                  sph("leaf_head", (x - 0.045, ty, z + 0.03), 0.018, C["holo"], seg=10, rings=6)):
            o.data.materials[0] = m
            objs.append(o)
        objs.append(wlab(bird, (x, ty + 0.02, z - 0.085), 0, 0, 6.2, size=0.058,
                         color=(0.75, 0.95, 1.0) if not ext else (0.6, 0.65, 0.7)))
        objs.append(wlab(plug, (x, ty + 0.02, z - 0.145), 0, 0, 6.2, size=0.05, color=BRANCH_COL[br_kind]))
    for lbl, k0, k1 in CLADES:
        zc = TZ0 + TDZ * (k0 + k1) / 2
        objs.append(_text(lbl, 0.042, (1.0, 0.85, 0.3), (-0.045, ty + 0.02, zc), None, "CENTER", "CENTER",
                          rot=(PI / 2, -PI / 2, PI), fnt="Arial Bold"))
    node = sph("hero_node", (0, ty, NODE_Z), 0.085, C["red"], scale=(1.4, 1, 0.9))
    node.data.materials[0] = nm
    objs.append(node)
    for f in range(F(0, 0), F(0, 6.2), 8):
        key_glow(nm, f, C["red"], 8.0 if (f // 8) % 2 == 0 else 3.0)
    objs.append(wlab("CYBER PIGEON", (0.0, ty + 0.03, NODE_Z + 0.2), 0, 0, 6.2, size=0.075, color=(1, 0.5, 0.45)))
    objs.append(wlab("(5) 800G OSFP DR8: 8 LANES", (0.0, ty + 0.03, NODE_Z + 0.12), 0, 0, 6.2, size=0.055, color=C["gold"]))
    for txt, z, col in (("GOLD (0)-(6) = PLUGGABLE LINEAGE", TZ0 - 0.13, C["gold"]), ("MAGENTA = COHERENT", TZ0 - 0.2, C["mag"]),
                        ("GRAY = EXTINCT", TZ0 - 0.27, (0.6, 0.65, 0.7))):
        objs.append(wlab(txt, (0.62, ty + 0.02, z), 0, 0, 6.2, size=0.04, color=col))
    for o in objs:
        V(o, 0, 0, 6.2)
    fade_mats([hm, gm, nm, lm, cm], 0, [(5.0, 1.0), (6.0, 0.0)])


# ------------------------------------------------------------------ street (shared by the S1 opening and the S6 ending)
SAT0, SAT1 = Vector((4, 38, 26)), Vector((9, 38, 27))
LAMPS = [Vector((-3.7, 16.2, 0)), Vector((3.7, 21.0, 0)), Vector((-3.7, 26.0, 0))]
LAMP_H = 4.6
POLES = [Vector((4.9, 15.4, 0)), Vector((4.9, 28.0, 0))]
POLE_H = 6.2
CCTV = [Vector((3.6, 24.5, 3.45)), Vector((-5.95, 17.0, 3.8)), Vector((5.95, 27.0, 4.4))]
RAIL_X, RAIL_Z = -3.35, 1.0


def bldg_h(sx, k):
    return 6 + 3 * ((k * 7 + (sx > 0)) % 3)


def wire_z(y, y0=15.4, y1=28.0, z=POLE_H - 0.3, sag=0.55):
    u = (y - y0) / (y1 - y0)
    return z - sag * 4 * u * (1 - u)


def build_street():
    objs = []
    objs.append(plane("street", (0, 23, 0.0), (18, 20.5), mat((0.075, 0.08, 0.1), 0.25), rot=(0, 0, 0)))
    for sx in (-1, 1):
        objs.append(plane("sidewalk", (sx * 4.75, 23, 0.01), (2.5, 20.5), mat((0.11, 0.11, 0.13), 0.6), rot=(0, 0, 0)))
        objs.append(box("curb", (sx * 3.5, 23, 0.06), (0.12, 20.5, 0.12), (0.2, 0.2, 0.22)))
    for k in range(6):
        objs.append(box("lane_mark", (0, 14.5 + 3 * k, 0.012), (0.1, 1.4, 0.004), (0.8, 0.8, 0.7), emit=0.6))
    win_cols = [C["orange"], C["cyan"], (1.0, 0.85, 0.5)]
    for sx in (-1, 1):
        for k in range(5):
            h = bldg_h(sx, k)
            objs.append(box("bldg", (sx * 7.5, 15 + 4 * k, h / 2), (3.0, 3.6, h), (0.07, 0.07, 0.09)))
            objs.append(box("roof_edge", (sx * 6.04, 15 + 4 * k, h + 0.05), (0.1, 3.6, 0.1), (0.15, 0.15, 0.17)))
            col = [C["mag"], C["cyan"], C["orange"]][(k + (sx > 0)) % 3]
            objs.append(box("neon", (sx * 5.95, 15 + 4 * k, 3.0 + k % 2), (0.05, 2.4, 0.6), col, emit=8.0))
            for wz in range(2, int(h) - 1, 2):
                for wy in (-1.0, 0.0, 1.0):
                    if (wz + int(wy) + k) % 3 == 0:
                        objs.append(box("window", (sx * 5.99, 15 + 4 * k + wy, wz + 0.5), (0.02, 0.5, 0.7), win_cols[(wz + k) % 3],
                                        emit=2.0))
    for txt, x, y, z, col in (("NO SIGNAL", 5.9, 23, 4.2, C["cyan"]), ("LINK UP", -5.9, 27, 3.8, C["mag"])):
        objs.append(_text(txt, 0.6, col, (x + (0.05 if x < 0 else -0.05), y, z), None, "CENTER", "CENTER",
                          rot=(PI / 2, 0, PI / 2 if x < 0 else -PI / 2), fnt="Arial Black"))
    for k in range(60):
        objs.append(sph("star", (random.uniform(-40, 40), random.uniform(30, 80), random.uniform(20, 45)), 0.08, (1, 1, 1),
                        emit=6.0, seg=6, rings=4))
    # lamp posts with arms toward the street
    for p in LAMPS:
        s = 1 if p.x < 0 else -1
        objs.append(cyl("lamp_post", (p.x, p.y, LAMP_H / 2), 0.06, LAMP_H, (0.2, 0.2, 0.22)))
        objs.append(box("lamp_arm", (p.x + s * 0.55, p.y, LAMP_H - 0.05), (1.1, 0.06, 0.06), (0.2, 0.2, 0.22)))
        objs.append(box("lamp_head", (p.x + s * 1.05, p.y, LAMP_H - 0.13), (0.4, 0.22, 0.08), (1.0, 0.85, 0.6), emit=5.0))
    # utility poles and overhead wires
    for p in POLES:
        objs.append(cyl("util_pole", (p.x, p.y, POLE_H / 2), 0.09, POLE_H, (0.22, 0.17, 0.12)))
        objs.append(box("crossbar", (p.x, p.y, POLE_H - 0.3), (1.2, 0.08, 0.08), (0.22, 0.17, 0.12)))
    for dx in (-0.5, 0.0, 0.5):
        pts = [(POLES[0].x + dx, y, wire_z(y) + 0.05) for y in [15.4 + 12.6 * u / 14 for u in range(15)]]
        objs.append(curve_line("wire", pts, 0.008, (0.03, 0.03, 0.03)))
    xw = [(POLES[0].x - 0.5 + (-6.0 - POLES[0].x + 0.5) * u / 12, 15.4 + 2.6 * u / 12) for u in range(13)]
    objs.append(curve_line("wire_x", [(x, y, POLE_H - 0.3 - 0.45 * 4 * (u / 12) * (1 - u / 12)) for u, (x, y) in enumerate(xw)], 0.008,
                           (0.03, 0.03, 0.03)))
    # railing, AC units, ledge, hanging sign, bin, scooter
    objs.append(cyl("rail_top", (RAIL_X, 20.5, RAIL_Z), 0.025, 5.0, (0.35, 0.36, 0.4), axis="Y", verts=10))
    objs.append(cyl("rail_mid", (RAIL_X, 20.5, 0.55), 0.015, 5.0, (0.35, 0.36, 0.4), axis="Y", verts=8))
    for yy in (18.0, 19.0, 20.0, 21.0, 22.0, 23.0):
        objs.append(cyl("rail_post", (RAIL_X, yy, RAIL_Z / 2), 0.02, RAIL_Z, (0.35, 0.36, 0.4), verts=8))
    for (x, y, z) in ((5.72, 19.0, 3.0), (5.72, 23.5, 4.8), (-5.72, 22.0, 3.6)):
        objs.append(box("ac_unit", (x, y, z), (0.55, 0.8, 0.55), (0.75, 0.76, 0.78), rough=0.5))
        objs.append(cyl("ac_fan", (x - 0.28 if x > 0 else x + 0.28, y, z), 0.2, 0.02, (0.2, 0.2, 0.22), axis="X", verts=16))
        objs.append(box("ac_bracket", (x, y, z - 0.3), (0.5, 0.05, 0.05), (0.3, 0.3, 0.32)))
    objs.append(box("ledge", (-5.87, 21.5, 6.0), (0.26, 8.0, 0.12), (0.18, 0.18, 0.2)))
    objs.append(box("hanging_sign", (-5.4, 17.5, 3.4), (0.9, 0.08, 0.6), (0.1, 0.05, 0.02), emit=0.0))
    for sgn in (-1, 1):
        objs.append(_text("RAMEN 24H", 0.15, C["orange"], (-5.4, 17.5 + sgn * 0.045, 3.4), None, "CENTER", "CENTER",
                          rot=(PI / 2, 0, 0 if sgn < 0 else PI), fnt="Arial Black"))
    objs.append(cyl("bin", (3.4, 18.0, 0.475), 0.28, 0.95, (0.15, 0.3, 0.2), verts=16))
    objs.append(cyl("bin_rim", (3.4, 18.0, 0.94), 0.3, 0.03, (0.3, 0.32, 0.34), verts=16))
    sc = [box("scooter_deck", (-2.9, 15.6, 0.32), (0.3, 1.0, 0.12), (0.6, 0.08, 0.1)),
          box("scooter_seat", (-2.9, 15.75, 0.66), (0.26, 0.5, 0.1), (0.05, 0.05, 0.05)),
          box("scooter_body", (-2.9, 15.75, 0.48), (0.28, 0.6, 0.26), (0.6, 0.08, 0.1)),
          cyl("scooter_stem", (-2.9, 15.15, 0.75), 0.025, 0.8, (0.3, 0.3, 0.32), verts=8),
          box("scooter_bar", (-2.9, 15.15, 1.14), (0.6, 0.04, 0.04), (0.2, 0.2, 0.2))]
    for yy in (15.15, 16.05):
        sc.append(cyl("scooter_wheel", (-2.9, yy, 0.2), 0.2, 0.1, (0.03, 0.03, 0.03), axis="X", verts=16))
    objs += sc
    for c in CCTV:
        s = -1 if c.x > 0 else 1
        if abs(c.x) < 5:
            objs.append(cyl("cctv_pole", (c.x, c.y, c.z / 2), 0.04, c.z, (0.3, 0.3, 0.32), verts=10))
        objs.append(box("cctv", (c.x, c.y, c.z), (0.4, 0.22, 0.2), (0.85, 0.85, 0.88)))
        objs.append(cyl("cctv_lens", (c.x + s * 0.22, c.y, c.z), 0.07, 0.08, C["black"], axis="X"))
        objs.append(sph("cctv_led", (c.x + s * 0.25, c.y, c.z + 0.1), 0.025, C["red"], emit=8.0))
    sat = empty("satellite", SAT0)
    sb = [box("sat_body", (0, 0, 0), (0.8, 0.8, 0.8), (0.8, 0.75, 0.5), sat)]
    for sx in (-1, 1):
        sb.append(box("sat_panel", (sx * 1.3, 0, 0), (1.6, 0.05, 0.7), (0.15, 0.2, 0.6), sat, emit=0.6))
    sb.append(sph("sat_led", (0, -0.45, 0.3), 0.08, C["red"], sat, emit=10.0))
    K(sat, F(0, 0), loc=tuple(SAT0), interp="LINEAR")
    K(sat, F(6, 0), loc=tuple(SAT1), interp="LINEAR")
    objs += build_facade()
    for o in objs + sb:
        vis(o, STREET_VIS)
    lights = [add_light("street_moon", "SUN", (0, 20, 30), 1.6, (0.55, 0.65, 1.0), rot=(0.5, 0.2, 0.3)),
              add_light("street_fill", "AREA", (0, 19.5, 9), 650, (0.55, 0.65, 1.0), size=12),
              add_light("street_fill2", "AREA", (0, 15.0, 6), 300, (0.6, 0.8, 1.0), size=8, rot=(0.9, 0, 0))]
    for p in LAMPS:
        s = 1 if p.x < 0 else -1
        lights.append(add_light("street_lamp", "POINT", (p.x + s * 1.05, p.y, LAMP_H - 0.4), 450, (1.0, 0.6, 0.8)))
    for sx, col in ((-1, C["mag"]), (1, C["cyan"])):
        lights.append(add_light("neon_spill", "AREA", (sx * 5.5, 21, 3.5), 900, col, size=6, rot=(0, -sx * PI / 2, 0)))
    lights.append(add_light("facade_wash", "AREA", (0, 15.5, 5.5), 120, (0.7, 0.8, 1.0), size=6, rot=(-0.9, 0, 0)))
    for lo in lights:
        vis(lo, STREET_VIS)
    return sat


# street perches: (name, feet position, yaw)
def street_perches():
    P = []
    P.append(("lamp1_top", LAMPS[0] + Vector((0, 0, LAMP_H + 0.02)), 0.3))
    P.append(("lamp1_arm", LAMPS[0] + Vector((0.65, 0, LAMP_H - 0.02)), -1.3))
    P.append(("lamp2_arm", LAMPS[1] + Vector((-0.5, 0, LAMP_H - 0.02)), 2.4))
    for y, yaw in ((18.2, PI), (19.0, PI - 0.2), (20.7, 0.1)):
        P.append(("wire_%.1f" % y, Vector((POLES[0].x, y, wire_z(y) + 0.06)), yaw))
    P.append(("rail_a", Vector((RAIL_X, 18.6, RAIL_Z + 0.025)), 0.35))
    P.append(("rail_b", Vector((RAIL_X, 22.1, RAIL_Z + 0.025)), -0.25))
    P.append(("ac_1", Vector((5.72, 19.0, 3.275)), PI))
    P.append(("ac_2", Vector((5.72, 23.5, 5.075)), PI - 0.4))
    P.append(("ledge_a", Vector((-5.85, 23.0, 6.06)), 0.1))
    P.append(("ledge_b", Vector((-5.85, 24.1, 6.06)), -0.35))
    P.append(("sign", Vector((-5.4, 17.5, 3.7)), 0.5))
    P.append(("bin", Vector((3.4, 18.0 - 0.2, 0.955)), 2.1))
    P.append(("scooter", Vector((-2.9, 15.8, 0.71)), 0.2))
    P.append(("roof_L", Vector((-6.04, 19.6, bldg_h(-1, 1) + 0.1)), 0.0))
    P.append(("roof_R", Vector((6.04, 15.6, bldg_h(1, 0) + 0.1)), PI))
    P.append(("cctv_pole", Vector((CCTV[0].x, CCTV[0].y + 0.05, CCTV[0].z + 0.1)), PI + 0.5))
    return P


FLYER_PERCHES = {"hero": (Vector((RAIL_X, 20.1, RAIL_Z + 0.025)), 0.15),
                 "comp0": (LAMPS[2] + Vector((0, 0, LAMP_H + 0.02)), 0.4),
                 "comp1": (Vector((POLES[0].x, 23.0, wire_z(23.0) + 0.06)), PI + 0.15)}


# ------------------------------------------------------------------ hero (GLB from workstream H)
def make_hero():
    b = Bird()
    b.root = empty("HERO", (0, -500, 0))
    b.root.scale = (S_BIRD,) * 3
    b.neck = empty("HERO_neck", tuple(P_H * MM), parent=b.root)
    path = os.path.join(ROOT_DIR, "assets", "hero", "pigeon_osfp_dr8.glb")
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    hroot = next(o for o in new if o.name.startswith("HERO_ROOT"))
    bpy.context.view_layer.update()
    cen = {}
    for o in new:
        if o.type == "MESH":
            bb = [o.matrix_world @ Vector(v) for v in o.bound_box]
            c = sum(bb, Vector()) / 8.0
        else:
            c = o.matrix_world.translation.copy()
        cen[o.name] = Vector((-c.y, c.x, c.z)) / MM    # bird frame, mm
    hroot.parent = b.root
    hroot.location = (0, 0, 0)
    hroot.rotation_mode = "XYZ"
    hroot.rotation_euler = (0, 0, PI / 2)
    hroot.scale = (1, 1, 1)
    bpy.context.view_layer.update()
    top = [o for o in new if o.parent == hroot]
    b.head_objs = [o for o in top if cen[o.name].x > 30 and cen[o.name].z > 68]
    for o in b.head_objs:
        o.rotation_mode = "XYZ"
    bpy.context.view_layer.update()
    for o in b.head_objs:
        mw = o.matrix_world.copy()
        o.parent = b.neck
        o.matrix_world = mw
    bpy.context.view_layer.update()
    b.glb, b.top, b.cen, b.hroot = new, top, cen, hroot
    b.by = lambda sub: next((o for o in new if sub in o.name), None)
    for o in new:
        if o.name.startswith("leash") or o.name.startswith("HOOK_leash"):
            o.hide_render = True
            o.hide_viewport = True
    b.hinge = b.by("HOOK_beak_hinge")
    b.hinge_base = b.hinge.rotation_euler.copy() if b.hinge is not None else None
    lt, lr = b.by("eye_tx_lenslet_array"), b.by("eye_rx_lenslet_array")
    b.eye_tx = lt.active_material if lt else BM["eye_tx"]
    b.eye_rx = lr.active_material if lr else BM["eye_rx"]
    b.lids = []
    for k in ("tx", "rx"):
        e, a = EYE_C[k], AX[k]
        piv = empty("hero_lid_pivot", tuple((e + a * 1.6 + Vector((0, 0, 6.8))) * MM), parent=b.neck)
        mdisc("hero_lid", (0, 0, -6.8), 6.9, 0.8, a, BM["lid"], piv)
        piv.scale = (1, 1, 0.0)
        b.lids.append(piv)
    # contact glow materials (unique copies): claws/talon tips = GND, ankle joint pads = power, toe edge pads = signal
    b.contact = {"gnd": [], "pwr": [], "sig": []}
    for o in new:
        cls = "gnd" if "talon_tip_gnd" in o.name else "pwr" if "joints_and_pads" in o.name else "sig" if "edge_pads" in o.name else None
        if cls and o.type == "MESH":
            for slot in o.material_slots:
                if slot.material is not None:
                    slot.material = slot.material.copy()
                    b.contact[cls].append(slot.material)
    return b


def make_stretch_leash(b):
    """Pull tab out of the hero's tail: a strap stretched from a tail anchor to an animated loop (world empty)."""
    anchor = empty("leash_anchor", tuple(Vector((-62, 0, 8)) * MM), parent=b.root)
    loop = empty("leash_loop_target", (0, -500, 0))
    w, t = 3.6 * MM * S_BIRD, 0.7 * MM * S_BIRD
    me = bpy.data.meshes.new("leash_strap")
    vs = [(-w / 2, 0, -t / 2), (w / 2, 0, -t / 2), (w / 2, 0, t / 2), (-w / 2, 0, t / 2),
          (-w / 2, 1, -t / 2), (w / 2, 1, -t / 2), (w / 2, 1, t / 2), (-w / 2, 1, t / 2)]
    me.from_pydata(vs, [], [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)])
    strap = bpy.data.objects.new("leash_strap", me)
    bpy.context.scene.collection.objects.link(strap)
    strap.data.materials.append(BM["leash"])
    c1 = strap.constraints.new("COPY_LOCATION")
    c1.target = anchor
    c2 = strap.constraints.new("STRETCH_TO")
    c2.target = loop
    c2.rest_length = 1.0
    c2.volume = "NO_VOLUME"
    ring = L.torus("leash_ring", (0, 0, -0.008), 0.0075, 0.0018, GREY, rot=(PI / 2, 0, 0), parent=loop, material=BM["leash"], segs=16)
    return anchor, loop, [strap, ring]
# ------------------------------------------------------------------ scenes
def hero_label(hero, sub, text, idx, t0, t1, off, size=0.017, color=(0.75, 0.95, 1.0)):
    o = hero.by(sub)
    if o is None:
        print("LABEL SKIPPED (part not found):", sub)
        return None
    mw = o.matrix_world
    if o.type == "MESH":
        c = sum((mw @ Vector(v) for v in o.bound_box), Vector()) / 8.0
    else:
        c = mw.translation.copy()
    local = mw.inverted() @ (c + Vector(off))
    return wlab(text, tuple(local), idx, t0, t1, size=size / mw.to_scale()[0], color=color, parent=o)


def glb_world(hero, sub, pos, yaw=0.0, explode=False):
    o = hero.by(sub)
    if o is None:
        return bwm(pos, yaw, HC)
    c = hero.cen[o.name].copy()
    if explode and o.get("explode") is not None:
        c += rotz(PI / 2) @ Vector(o["explode"]) / MM
    return bwm(pos, yaw, c)


def main_build():
    bird_mats()
    s = make_subject()
    racks = build_hall()
    cols = bird_collections()
    hall_heads, cells = build_crowd(cols)
    build_tree()
    tx_sub, rx_sub = os.path.exists(TX_GLB), os.path.exists(RX_GLB)
    build_tx_sub() if tx_sub else build_tx_macro()
    build_rx_sub() if rx_sub else build_rx_macro()
    build_hatchery()
    build_street()
    hero = make_hero()
    anchor, loop, leash_objs = make_stretch_leash(hero)
    # S5 labels on the GLB parts (made before any keyframes so the parent matrices are the rest pose)
    dsp_o = hero.by("dsp_die")
    for k, name in enumerate(("NARROWCOM", "MERVALL", "MINLINEAR")):
        if dsp_o is None:
            break
        t = hero_label(hero, "dsp_die", name, 4, 2.6 + 0.75 * k, 3.35 + 0.75 * k if k < 2 else 8.0, (0, 0, 0.004), size=0.006,
                       color=C["gold"])
    for sub, txt, off, a, b in (("dsp_die", "DSP (BRAIN)", (0.02, -0.04, 0.03), 2.8, 8.0),
                                ("tx_pic_siph_die", "TX PIC + 2 DFB", (0.02, -0.05, -0.015), 2.9, 8.0),
                                ("rx_octal_tia", "PIN ARRAY + TIA", (0.03, 0.02, -0.035), 2.9, 8.0),
                                ("elec_mcu", "MCU", (-0.03, -0.04, 0.0), 4.4, 8.0),
                                ("elec_pmic", "POWER", (0.0, -0.05, 0.02), 4.8, 8.0),
                                ("elec_crystal_2016", "CRYSTAL", (0.02, -0.05, 0.035), 5.0, 8.0),
                                ("elec_eeprom", "EEPROM", (0.0, -0.05, -0.02), 5.2, 8.0),
                                ("foot_tx_edge_pads", "FEET = 60 PADS", (0.0, -0.05, -0.02), 3.0, 8.0),
                                ("wing_right_flight_feather_fins", "FIN FEATHERS = HEATSINK", (0.0, -0.04, 0.03), 2.9, 4.4),
                                ("flex_tx_host_lanes_neck_leg", "FLEX RIBBONS", (0.0, -0.05, 0.0), 5.6, 8.0)):
        hero_label(hero, sub, txt, 4, a, b, off)
    hero_label(hero, "leg_tx_band_eeprom_carrier", "HATCHED BY OUTNOLIGHT", 4, 8.6, 10.0, (0.0, -0.04, 0.0), size=0.016,
               color=C["orange"])
    comps = [make_bird("COMP%d" % k) for k in range(2)]
    hp, hy0 = cell_pos(*HERO_CELL)
    ap, ay = cell_pos(*ACROSS_CELL)
    across = cells[ACROSS_CELL]
    cps = [cell_pos(*c) for c in COMP_CELLS]
    for k, b in enumerate(comps):
        bird_at(b, 0, 0, cps[k][0], cps[k][1], interp="CONSTANT")
    comp_heads = [Head(cps[k][0], cps[k][1], rig=b) for k, b in enumerate(comps)]
    DH = Vector((hp.x + 1.16, 0.0, hp.z - 1.315))    # v0.1 hero cell -> layout hero cell (hand-placed keys below are shifted by this)
    FXF = hp.x + FEET_IN                             # hero cell faceplate x
    pad = PADS.get(HERO_CELL[0], Vector((-1.35, 7.0, 2.256)))
    # street birds (perched casually; visible in the S1 opening and the S6 ending)
    street_heads, sh = [], {}
    for name, pos, yaw in street_perches():
        bi = instancer("sb", cols["FREE_BODY"], pos, yaw, S_BIRD)
        hi = instancer("sh", cols["CROWD_HEAD"], bwm(pos, yaw, P_H), yaw, S_BIRD)
        for o in (bi, hi):
            vis(o, STREET_VIS)
        hd = Head(pos, yaw, inst=hi)
        street_heads.append(hd)
        sh[name] = hd

    # links used in S3/S4 (hero <-> bird across the aisle)
    h_tx, a_rx = solve_link((hp, hy0), (ap, ay))
    a_tx, h_rx = solve_link((ap, ay), (hp, hy0))
    across.blocked.append((F(2, 7.9), F(3, 2.3)))
    # ---------------- background head motion (hall)
    rng = random.Random(21)
    hall_track = [(0, 8.0, (0.3, 11.8)), (0, 8.7, (0.25, 10.4)), (0, 9.4, (0.2, 9.0)), (0, 10.0, (-0.25, 7.6)), (1, 2.0, (-0.25, 7.6))]
    s5_track = [(4, 0.5, (-0.3, 7.6)), (4, 2.5, (0.1, 9.8)), (4, 4.5, (0.5, 12.0))]
    allh = hall_heads + comp_heads
    animate_heads(allh, F(0, 0.0), F(0, 7.8), rng)
    track_heads(allh, hall_track)
    animate_heads(allh, F(1, 2.3), F(4, 0.3), rng)
    track_heads(allh, s5_track)
    animate_heads(hall_heads, F(4, 4.7), F(5, 4.0), rng)
    animate_heads(comp_heads, F(4, 4.7), F(4, 9.8), rng)
    across.key(F(2, 8.4), a_rx)
    across.key(F(3, 0.2), a_rx)
    across.key(F(3, 0.6), a_tx)
    across.key(F(3, 2.0), a_tx)
    animate_heads(street_heads, F(0, 0.0), F(0, 9.8), rng)

    # ---------------- hero: internals of the GLB only matter in S5; ghost shell for the exploded view
    ghost_m = pmat((0.35, 0.4, 0.5), 0.3, 0.3)
    set_blend(ghost_m)
    for tt, a in ((2.0, 1.0), (2.6, 0.16), (8.0, 0.16), (8.6, 1.0)):
        key_alpha(ghost_m, F(4, tt), a)
    ghosts = []
    for o in list(hero.top):
        if not o.name.startswith("skin_") or o.type != "MESH":
            continue
        if "seams" in o.name or "screws" in o.name:
            for i in range(7):
                if i == 4:
                    V(o, 4, 0, 2.0)
                    V(o, 4, 8.6, 10)
                else:
                    V(o, i, 0, 10 if i < 6 else 2)
            continue
        g = o.copy()
        bpy.context.scene.collection.objects.link(g)
        for slot in g.material_slots:
            slot.link = "OBJECT"
            slot.material = ghost_m
        V(g, 4, 2.0, 8.6)
        ghosts.append(g)
        for i in range(7):
            if i == 4:
                V(o, 4, 0, 2.0)
                V(o, 4, 8.6, 10)
            else:
                V(o, i, 0, 10 if i < 6 else 2)
    # leash visible while docked or being pulled (S1 8.2 - S5 0.8)
    for o in leash_objs:
        V(o, 0, 8.2, 10); V(o, 1, 0, 10); V(o, 2, 0, 10); V(o, 3, 0, 10); V(o, 4, 0, 0.8)

    # cutaway cage for the hero cell (S2 6.8 - S3 0.8): host board with contact pads, back wall, cage lip
    fz = hp.z
    cage = [box("host_board", (hp.x - 0.1, hp.y, fz - 0.003), (0.36, 0.2, 0.006), C["pcb"]),
            box("cage_back", (FXF - 0.36, hp.y, fz + 0.135), (0.01, 0.2, 0.27), C["cell"]),
            box("cage_top", (hp.x - 0.1, hp.y, fz + 0.272), (0.36, 0.2, 0.004), (0.55, 0.56, 0.6), rough=0.3)]
    for sgn in (-1, 1):
        fw = bwm(hp, 0.0, (12.5, sgn * 7.4, 0))
        cage.append(box("host_pad", (fw.x, fw.y, fz - 0.0002), (0.08, 0.04, 0.0012), C["gold"], rough=0.3))
    lip = bwm(hp, 0.0, (61, 0, 76.5))
    cage.append(cyl("cage_lip", (lip.x, lip.y, lip.z), 1.2 * S_BIRD * MM, 0.2, (0.6, 0.62, 0.66), axis="Y", verts=12, rough=0.3))
    for sgn in (-1, 1):
        cage.append(box("lip_bracket", ((lip.x + FXF) / 2, lip.y + sgn * 0.09, lip.z), (abs(lip.x - FXF), 0.006, 0.006), (0.6, 0.62, 0.66)))
    cage.append(add_light("cutaway_light", "POINT", (hp.x + 0.05, hp.y - 0.3, hp.z + 0.25), 9.0, (0.9, 0.95, 1.0)))
    for o in cage:
        V(o, 1, 6.8, 10.0)
        V(o, 2, 0, 0.8)
    for o in racks[HERO_CELL[0]]:
        L._VIS[o] = [(F(0, 0), F(1, 6.8)), (F(2, 0.8), F(5, 4.0))]
    pad_glow = box("pad_charge_glow", (pad.x + 0.12, pad.y, pad.z + 0.004), (0.02, 0.5, 0.004), C["cyan"])
    pad_m = mat(C["cyan"], 0.3, 1.0, unique=True)
    pad_glow.data.materials[0] = pad_m
    vis(pad_glow, HALL_VIS)
    for tt, st in ((6.9, 1.0), (7.05, 9.0), (7.3, 6.0), (7.6, 1.0)):
        key_glow(pad_m, F(0, tt), C["cyan"], st)

    # ---------------- S1: street, tree hologram, bird flap, landing pad, dock
    i = 0
    cam(i, 0.0, (0.55, 19.9, 0.75), (0.1, TREE_C.y, 1.0), 24)
    cam(i, 1.0, (0.12, 18.3, 1.15), (0.0, TREE_C.y, 1.2), 24)
    cam(i, 2.4, (-0.05, 18.25, 2.05), (0.0, TREE_C.y, 2.1), 24)
    cam(i, 3.6, (0.05, 18.25, 2.95), (0.0, TREE_C.y, 3.05), 24)
    cam(i, 4.4, (0.0, 17.95, 3.95), (0.0, TREE_C.y, NODE_Z + 0.1), 24)
    cam(i, 5.2, (0.05, 15.5, 3.45), (FLAP.x, FACADE_Y, FLAP.z), 26)
    cam(i, 5.75, (FLAP.x, 14.0, FLAP.z), (FLAP.x, 12.0, FLAP.z - 0.05), 26)
    cam(i, 6.25, (FLAP.x, 12.35, FLAP.z - 0.02), (-0.9, 9.4, 2.6), 26)
    cam(i, 7.0, (-0.45, 8.8, 2.8), pad + Vector((0.1, 0, 0.1)), 26)
    cam(i, 7.8, (-0.35, 7.55, 1.8 + DH.z), (-0.85, 7.0, 1.48 + DH.z), 28)
    cam(i, 8.4, (-0.25, 8.7, 1.6), (0.2, 11.5, 1.45), 28)
    cam(i, 9.3, (-0.4, 7.6, 1.58), (-0.6, 8.6, 1.5), 28)
    hero_head_w = bwm(hp, 0.0, HC)
    cam(i, 10.0, Vector((-0.62, 6.72, 1.6)) + DH, hero_head_w + Vector((0, 0, -0.03)), 32)
    for t, pos, yaw, pitch in ((0.0, (0.42, 17.55, 0.6), -PI / 2, -0.4), (1.8, (0.45, 17.55, 1.7), -PI / 2, -0.4),
                               (3.6, (0.42, 17.5, 3.05), -PI / 2, -0.3), (4.4, (0.1, 17.3, 3.75), -PI / 2, 0.0),
                               (5.3, (0.05, 14.7, 3.3), -PI / 2, 0.05), (5.75, (FLAP.x, 13.5, FLAP.z - 0.1), -PI / 2, 0.0),
                               (6.05, (FLAP.x, 12.5, FLAP.z - 0.12), -PI / 2, 0.1), (6.6, (-0.75, 9.4, 2.75), -PI / 2, 0.15),
                               (7.0, tuple(pad), -PI / 2, 0.0), (7.3, tuple(pad), 0.0, 0.0),
                               (7.75, (-0.72, 7.0, hp.z + 0.02), 0.0, 0.0), (8.2, tuple(hp), 0.0, 0.0)):
        bird_at(hero, i, t, pos, yaw, pitch)
    head(hero, i, 0.0, 0.0, interp="CONSTANT")
    head(hero, i, 7.1, 0.0)
    head(hero, i, 7.25, 0.9)
    head(hero, i, 7.45, -0.9)
    head(hero, i, 7.6, 0.0)
    for idx, t, sp in hall_track[:4]:
        head(hero, idx, t, stare(hp, hy0, Vector((sp[0], sp[1], 1.55))))
    for piv in FLAPS:
        r0 = piv.rotation_euler.copy()
        for tt, a in ((5.5, 0.0), (5.8, 1.35), (6.6, 1.35), (7.3, 0.0)):
            K(piv, F(i, tt), rot=(r0.x, r0.y + a, r0.z), interp="BEZIER")
    L_lip = Vector((FXF + 0.004, hp.y, hp.z - 0.075))
    K(loop, F(0, 8.2), loc=tuple(L_lip), interp="CONSTANT")
    subj_at(s, i, 0, (1.1, 21.5), 0.0, interp="CONSTANT")
    subj_walk(s, i, 0.5, 4.2, (1.1, 21.5), (0.4, 13.3), 0.0)
    subj_at(s, i, 4.3, (0.3, 12.6), 0.0, interp="CONSTANT")
    subj_at(s, i, 8.0, (0.3, 11.8), 0.0)
    subj_walk(s, i, 8.0, 10.0, (0.3, 11.8), (-0.25, 7.6), 0.0)
    subj_at(s, i, 10.0, (-0.25, 7.6), yaw_face((-0.25, 7.6), (-1.06, 7.0)))
    wlab("LANDING PAD: CHARGE + SYNC", pad + Vector((0.45, 0, 0.25)), i, 6.5, 7.7, size=0.05, color=C["cyan"])
    narr(i, [(0.2, 2.0, "Birds evolved from dinosaurs."), (2.0, 3.6, "Or so they told you."),
             (3.6, 7.6, "Meet the OSFP: Overhead Surveillance Feathered Platform."), (8.0, 9.8, "It is watching you.")])
    lab(i, 0.0, 4.6, "BIRD TREE (TRAIT-MATCHED)\n+ PLUGGABLE LINEAGE (SIMPLIFIED)")
    lab(i, 6.4, 8.2, "800G OSFP DR8\nDR8: DRONE RECON, 8 LANES")
    lab(i, 8.2, 10.0, "HEADS TRACK THE SUBJECT")
    fx(i, 0.0, 4.6, "[FX: hologram flicker, scanlines, leaf birds animated]")
    fx(i, 4.6, 6.4, "[FX: wing flaps, flap swing, rain mist]")
    fx(i, 6.4, 8.2, "[FX: pad charge glow, data-sync shimmer]")

    # ---------------- S2: tail leash pull, eyes, feet, cutaway
    i = 1
    subj_at(s, i, 0.0, (-0.25, 7.6), yaw_face((-0.25, 7.6), (-1.06, 7.0)), interp="CONSTANT")
    reach(s, i, 0.0, 0.0); reach(s, i, 0.4, 0.9); reach(s, i, 1.8, 0.75); reach(s, i, 6.8, 0.75); reach(s, i, 7.6, 0.95)
    reach(s, i, 9.4, 0.9); reach(s, i, 10.0, 0.0)
    held = Vector((-0.62, 7.0, 1.30)) + DH
    belly_pitch = -1.05
    above = hp + Vector((0, 0, 0.012))
    tail_dangle = Vector((-0.62, 7.05, 1.78)) + DH
    dangle_root = tail_dangle - Vector((0.0208, 0, 0.161)) - Vector((0, 0, 0.03))
    bird_at(hero, i, 0.0, hp, 0.0)
    bird_at(hero, i, 0.35, hp, 0.0)
    bird_at(hero, i, 1.0, (hp.x + 0.42, 7.02, hp.z + 0.02), 0.0)
    bird_at(hero, i, 1.6, dangle_root, 0.0, pitch=PI / 2)
    bird_at(hero, i, 1.85, dangle_root + Vector((0.0, 0.0, -0.01)), 0.0, pitch=PI / 2 - 0.15)
    bird_at(hero, i, 2.3, held, 0.0)
    bird_at(hero, i, 4.4, held, 0.0)
    bird_at(hero, i, 5.0, held, 0.0, pitch=belly_pitch)
    bird_at(hero, i, 6.6, held, 0.0, pitch=belly_pitch)
    bird_at(hero, i, 7.2, Vector((-0.95, 7.0, above.z)), 0.0)
    bird_at(hero, i, 7.7, above, 0.0, interp="LINEAR")
    bird_at(hero, i, 8.9, hp, 0.0)
    tail_held = bwm(held, 0.0, (-62, 0, 8)) + Vector((0, 0, -0.04))
    tail_belly = bwm(held, 0.0, (-62, 0, 8), belly_pitch) + Vector((0, 0, -0.04))
    for tt, p in ((0.0, L_lip), (0.35, (hp.x + 0.3, hp.y + 0.02, hp.z + 0.02)), (1.0, Vector((-0.62, 7.05, 1.62)) + DH), (1.6, tail_dangle),
                  (1.85, tail_dangle), (2.3, tail_held), (4.4, tail_held), (5.0, tail_belly), (6.6, tail_belly),
                  (7.3, (-1.0, 7.0, hp.z - 0.05)), (8.2, L_lip)):
        K(loop, F(i, tt), loc=tuple(p), interp="BEZIER")
    head(hero, i, 0.0, stare(hp, hy0, Vector((-0.25, 7.6, 1.55))))
    head(hero, i, 0.3, 0.0)
    for tt, hv in ((2.3, 0.0), (2.8, 1.5), (3.25, 1.5), (3.75, -1.5), (4.2, -1.5), (4.5, 0.0)):
        head(hero, i, tt, hv)
    beak(hero, i, 0.0, 0.0); beak(hero, i, 0.3, 1.0); beak(hero, i, 1.0, 0.0)
    beak(hero, i, 3.25, 0.0); beak(hero, i, 3.4, 1.0); beak(hero, i, 3.55, 0.0)
    beak(hero, i, 7.5, 0.0); beak(hero, i, 7.7, 1.0); beak(hero, i, 9.2, 1.0); beak(hero, i, 9.4, 0.0)
    for m in {hero.eye_tx, hero.eye_rx}:   # unplugged = no power = dark eyes
        b_ = m.node_tree.nodes.get("Principled BSDF")
        if b_ is None:
            continue
        st0 = b_.inputs["Emission Strength"].default_value or 4.0
        for tt, st, it in ((0.3, st0, "LINEAR"), (0.8, 0.0, "CONSTANT"), (9.0, 0.0, "LINEAR"), (9.25, st0, "CONSTANT")):
            key_strength(m, F(i, tt), st, it)
    t_g, t_p, t_s = 8.1, 8.5, 8.9
    for cls, tt, col in (("gnd", t_g, C["brass"]), ("pwr", t_p, C["gold"]), ("sig", t_s, C["gold"])):
        for m in hero.contact[cls]:
            key_glow(m, F(i, tt - 0.05), col, 0.0)
            key_glow(m, F(i, tt), col, 14.0)
            key_glow(m, F(i, tt + 0.45), col, 0.8, interp="CONSTANT")
    for txt, tt, off in (("1 GND (CLAWS)", t_g, (0.05, -0.035, 0.006)), ("2 POWER (ANKLES)", t_p, (0.05, -0.035, 0.019)),
                         ("3 SIGNAL (TOE PADS)", t_s, (0.05, -0.035, 0.032))):
        wlab(txt, hp + Vector(off), i, tt, 10.0, size=0.0075, color=(1.0, 0.85, 0.3))
    face_w = held + Vector((0.12, 0, 0.21))
    cam(i, 0.0, Vector((-0.62, 6.72, 1.6)) + DH, hero_head_w + Vector((0, 0, -0.03)), 32)
    cam(i, 1.0, Vector((-0.3, 6.7, 1.6)) + DH, Vector((-0.66, 7.03, 1.5)) + DH, 32)
    cam(i, 2.0, Vector((-0.2, 7.0, 1.53)) + DH, face_w + Vector((0, 0, -0.03)), 38)
    cam(i, 4.4, Vector((-0.22, 7.0, 1.53)) + DH, face_w + Vector((0, 0, -0.03)), 38)
    belly_t = bwm(held, 0.0, (10, 0, 14), belly_pitch)
    cam(i, 5.0, belly_t + Vector((0.26, 0.05, -0.02)), belly_t, 40)
    cam(i, 6.6, belly_t + Vector((0.25, -0.04, -0.03)), belly_t, 42)
    feet_t = hp + Vector((0.03, 0.0, 0.012))
    cam(i, 7.3, Vector((-0.98, 6.72, 1.38)) + DH, feet_t + Vector((0.0, 0, 0.03)), 34)
    cam(i, 8.0, Vector((-1.05, 6.8, 1.34)) + DH, feet_t, 40)
    cam(i, 8.9, Vector((-1.055, 6.8, 1.338)) + DH, feet_t, 42)
    beak_t = bwm(hp, 0.0, (60, 0, 77))
    cam(i, 9.5, Vector((-0.92, 6.82, 1.49)) + DH, beak_t + Vector((-0.03, 0, -0.01)), 36)
    cam(i, 10.0, Vector((-0.88, 6.8, 1.52)) + DH, beak_t + Vector((-0.04, 0, 0.0)), 34)
    narr(i, [(0.2, 1.8, "Feathers: heatsink."), (1.8, 4.2, "Beak: a latch that bites the cage."),
             (4.6, 6.0, "Right foot and eye transmit."), (6.0, 7.4, "Left foot and eye receive."), (7.6, 9.4, "Ground lands first.")])
    lab(i, 0.0, 2.0, "PULL TAB = LEASH (TAIL)  |  BEAK = RETENTION LATCH")
    lab(i, 2.0, 4.5, "EYES: 2x4 LENSLET ARRAYS\nRIGHT = TX (RED)  LEFT = RX (CYAN)\nUNPLUGGED: NO POWER, EYES DARK")
    lab(i, 4.5, 7.0, "60 CONTACTS\nRIGHT FOOT TX | LOW-SPEED + POWER | LEFT FOOT RX")
    lab(i, 7.0, 10.0, "CUTAWAY: GROUND -> POWER -> SIGNAL")
    fx(i, 0.0, 2.0, "[FX: feather-fin shimmer, cable-tension on the leash]")
    fx(i, 7.6, 9.4, "[FX: contact glow pulses in mating order]")

    # ---------------- S3: TX eye, macro, zoom back out to the hero
    i = 2
    head(hero, i, 0.0, 0.0)
    head(hero, i, 0.8, h_tx)
    e_tx = eye_w(hp, hy0, h_tx, "tx")
    ax_t = axis_w(hy0, h_tx, "tx")
    side_t = ax_t.cross(Vector((0, 0, 1))).normalized()
    e_arx = eye_w(ap, ay, a_rx, "rx")
    cam(i, 0.0, Vector((-0.88, 6.8, 1.52)) + DH, beak_t + Vector((-0.04, 0, 0.0)), 34)
    cam(i, 0.9, e_tx + ax_t * 0.42 + Vector((0, 0, 0.01)), e_tx, 34)
    cam(i, 1.9, e_tx + ax_t * 0.025, e_tx - ax_t * 0.1, 40)
    fill(i, C["red"], [(1.4, 0.0), (1.9, 1.0), (2.0, 1.0), (2.5, 0.0)])
    X = TX_X
    if tx_sub:   # camera retargeted onto the T subassembly (mm -> m)
        tx_keys = [(2.0, (2.4, 0.6, 0.9), (0, 0, 0)), (3.0, (1.4, -0.2, -1.45), (-1.75, -0.6, -2.5)), (4.0, (1.2, -2.6, -1.0), (-1.7, -2.0, -1.8)),
                   (4.6, (1.3, -3.8, 1.8), (-1.7, -5.0, 0.3)), (5.8, (1.0, -7.2, 1.4), (-1.7, -7.6, 0.0)), (6.6, (1.4, -5.0, 2.4), (-1.7, -3.0, 0.5)),
                   (7.2, (1.0, -3.6, 1.9), (-1.6, -1.2, 0.0)), (7.8, (1.8, -3.2, 1.5), (-0.6, 0.0, 0.0)), (8.1, (3.2, -2.4, 1.2), (0, 0, 0)),
                   (8.5, (3.5, -2.2, 1.25), (0, 0, 0))]
        tx_keys = [(t, W_tx(p), W_tx(q)) for t, p, q in tx_keys]
    else:
        tx_keys = [(2.0, (X + 15.0, -0.4, 1.9), (X + 12.4, 0, 0.4)), (3.0, (X + 7.0, 0.9, 1.5), (X + 3.0, 0, 0.1)),
                   (4.0, (X + 1.6, 0.6, 1.2), (X - 0.4, 1.0, 0.25)), (4.6, (X + 0.2, -0.9, 1.4), (X + 3.0, 0, 0.0)),
                   (5.8, (X + 3.4, -0.9, 1.25), (X + 5.6, 0.0, 0.0)), (7.0, (X + 7.6, -2.0, 1.2), (X + 10.0, 0, 0.1)),
                   (7.7, (X + 10.4, -3.4, 1.2), (X + 11.4, 0, 0.4)), (8.1, (X + 14.6, -3.2, 1.3), (X + 12.5, 0, 0.4)),
                   (8.5, (X + 15.0, -3.0, 1.35), (X + 12.5, 0, 0.4))]
    cam_jump(i, 2.0, (e_tx + ax_t * 0.025, e_tx - ax_t * 0.1, 40), (tx_keys[0][1], tx_keys[0][2], 26))
    for t, p, q in tx_keys[1:]:
        cam(i, t, p, q, 26)
    fill(i, C["red"], [(8.28, 0.0), (8.55, 1.0), (8.65, 1.0), (9.1, 0.0)])
    off1 = side_t * 0.03 + Vector((0, 0, 0.03))
    off2 = side_t * 0.1 + Vector((0, 0, 0.07))
    cam_jump(i, 8.6, (tx_keys[-1][1], tx_keys[-1][2], 26), (e_tx + ax_t * 0.06 + off1, e_tx, 34))
    cam(i, 10.0, e_tx + ax_t * 0.5 + off2, e_tx + ax_t * 0.05, 30)
    for r in range(2):
        for c in range(4):
            p0 = hwm(hp, hy0, h_tx, lens_local("tx", r, c))
            o = beam("link_beam_tx", p0, p0 + (e_arx - e_tx), 0.0016, C["red"], emit=6.0)
            V(o, 2, 8.6, 10.0)
    subj_at(s, i, 0.0, (-0.3, 7.6), yaw_face((-0.3, 7.6), (-1.06, 7.0)), interp="CONSTANT")
    narr(i, [(0.2, 1.8, "Right eye: transmit."), (2.0, 4.4, "Two DFB lasers glow deep in its skull."),
             (4.6, 6.8, "Eight Mach-Zehnder modulators."), (7.0, 8.0, "Eight lenses."), (8.2, 9.4, "Eight beams.")])
    lab(i, 0.0, 1.4, "ONE-EYED STARE: RIGHT EYE (TX)")
    lab(i, 2.0, 8.6, "TX SUBASSEMBLY (x1000)" if tx_sub else "TX SILICON PHOTONICS CHIP")
    lab(i, 8.6, 10.0, "8 FREE-SPACE BEAMS TO THE BIRD ACROSS THE AISLE")
    fx(i, 1.4, 2.5, "[FX: red fill = lens glow, macro emerges]")
    fx(i, 4.6, 7.0, "[FX: optical-mode glow along waveguides, PAM4 shimmer]")
    note(i, 2.5, 8.45)
    note(i, 8.75, 10.0)

    # ---------------- S4: head turn, RX eye, macro, POV
    i = 3
    head(hero, i, 0.0, h_tx)
    head(hero, i, 0.9, h_rx)
    e_rx = eye_w(hp, hy0, h_rx, "rx")
    ax_r = axis_w(hy0, h_rx, "rx")
    side_r = ax_r.cross(Vector((0, 0, 1))).normalized()
    e_atx = eye_w(ap, ay, a_tx, "tx")
    off_dir = (ax_r * math.cos(math.radians(32)) + side_r * math.sin(math.radians(32))).normalized()
    cam(i, 0.0, e_tx + ax_t * 0.5 + off2, e_tx + ax_t * 0.05, 30)
    cam(i, 1.0, e_rx + ax_r * 0.42 + side_r * 0.1 + Vector((0, 0, 0.05)), e_rx, 32)
    cam(i, 1.9, e_rx + off_dir * 0.032 + Vector((0, 0, 0.008)), e_rx, 36)
    for r in range(2):
        for c in range(4):
            p0 = hwm(ap, ay, a_tx, lens_local("tx", r, c))
            o = beam("link_beam_rx", p0, p0 + (e_rx - e_atx), 0.0016, C["red"], emit=6.0)
            V(o, 3, 0.9, 2.0)
    fill(i, C["cyan"], [(1.4, 0.0), (1.9, 1.0), (2.0, 1.0), (2.5, 0.0)])
    X = RX_X
    if rx_sub:
        rx_keys = [(2.0, (2.6, -2.2, 1.6), (-0.25, 0, 0)), (3.2, (-0.8, -3.6, 2.4), (-2.1, 0, 0.0)), (4.5, (-2.6, -3.0, 1.5), (-3.95, 0, -1.25)),
                   (5.2, (-4.9, -1.6, 0.9), (-4.8, 0.0, -1.25)), (5.8, (-5.0, -0.8, 0.5), (-4.85, 0.5, -1.2))]
        rx_keys = [(t, W_rx(p), W_rx(q)) for t, p, q in rx_keys]
    else:
        rx_keys = [(2.0, (X - 2.2, -2.3, 1.8), (X, 0, 0.4)), (3.2, (X + 1.4, -2.9, 1.7), (X + 2.6, 0, 0.45)),
                   (4.5, (X + 2.4, -2.5, 1.5), (X + 3.2, 0, 0.4)), (6.0, (X + 5.4, -1.6, 1.2), (X + 8.0, 0, 0.1))]
    cam_jump(i, 2.0, (e_rx + off_dir * 0.032 + Vector((0, 0, 0.008)), e_rx, 36), (rx_keys[0][1], rx_keys[0][2], 26))
    for t, p, q in rx_keys[1:]:
        cam(i, t, p, q, 26)
    fill(i, C["cyan"], [(5.9, 0.0), (6.35, 1.0), (6.45, 1.0), (6.9, 0.0)])
    subj_pt = Vector((-0.3, 7.6, 1.5))
    h_pov = aim(hp, hy0, "rx", subj_pt)
    head(hero, i, 6.3, h_rx, interp="CONSTANT")
    head(hero, i, 6.4, h_pov, interp="CONSTANT")
    head(hero, i, 9.4, h_pov)
    pov_e = eye_w(hp, hy0, h_pov, "rx")
    pov_c = pov_e + (subj_pt - pov_e).normalized() * 0.05
    cam_jump(i, 6.4, (rx_keys[-1][1], rx_keys[-1][2], 26), (pov_c, subj_pt, 16))
    cam(i, 9.4, pov_c + (subj_pt - pov_c).normalized() * 0.12, subj_pt + Vector((0, 0, 0.05)), 22)
    cam(i, 10.0, Vector((-0.6, 6.7, 1.62)) + DH, hero_head_w, 30)
    narr(i, [(0.2, 1.8, "Left eye: receive."), (2.0, 4.2, "Eight lenses, eight photodiodes."),
             (4.4, 7.2, "A TIA, a transimpedance amplifier, turns tiny currents into voltage."), (7.6, 9.2, "It sees you.")])
    lab(i, 0.0, 1.4, "HEAD TURN: LEFT EYE (RX)")
    lab(i, 2.0, 6.4, "RX: LENSLETS -> INTERLEAVER -> MIRROR -> PIN ARRAY -> TIA" if rx_sub else
        "RX: LENSLETS -> PRISM -> PIN ARRAY -> TIA")
    lab(i, 6.4, 10.0, "BIRD POV\nSUBJECT ACQUIRED")
    fx(i, 6.4, 10.0, "[FX: 340 deg fisheye, surveillance HUD, PAM4 eye glyphs on his face]")
    note(i, 0.95, 6.4)
    hud = []
    for (x, y, w, h) in ((0, 0.55, 1.3, 0.02), (0, -1.2, 1.3, 0.02), (-0.65, -0.32, 0.02, 1.77), (0.65, -0.32, 0.02, 1.77)):
        q = plane("hud", (x, y, 0.0), (w, h), mat(C["red"], emit=100), rot=(0, 0, 0), parent=L.HOLD)
        hud.append(q)
    for q in hud:
        V(q, i, 6.6, 9.3)

    # ---------------- S5: exploded hero GLB, hatchery
    i = 4
    out = Vector((-0.42, 7.0, 1.28)) + DH
    cam(i, 0.0, Vector((-0.6, 6.7, 1.62)) + DH, hero_head_w, 30)
    cam(i, 0.8, Vector((-0.35, 6.55, 1.64)) + DH, Vector((-0.9, 7.0, 1.5)) + DH, 30)
    cam(i, 1.4, Vector((-0.05, 6.45, 1.62)) + DH, out + Vector((0.0, 0, 0.12)), 30)
    exp_c = out + Vector((0.0, 0, 0.13))
    cam(i, 2.4, exp_c + Vector((0.28, -0.82, 0.24)), exp_c, 30)
    bird_at(hero, i, 0.0, hp, 0.0)
    bird_at(hero, i, 0.9, hp, 0.0)
    bird_at(hero, i, 1.8, out, 0.0)
    bird_at(hero, i, 8.0, out, 0.0)
    head(hero, i, 0.0, h_pov)
    head(hero, i, 0.9, h_pov)
    head(hero, i, 1.8, 0.0)
    n_ex = 0
    for o in list(hero.top) + ghosts:
        ex = o.get("explode")
        if ex is None or o.hide_render:
            continue
        v = Vector(tuple(ex))
        off = v if o.parent == hero.hroot else rotz(PI / 2) @ v
        base = o.location.copy()
        for tt, d in ((1.9, 0.0), (2.7, 1.0), (8.0, 1.0), (8.6, 0.0)):
            K(o, F(i, tt), loc=tuple(base + off * d), interp="BEZIER")
        n_ex += 1
    print("HERO EXPLODE PARTS", n_ex)
    dsp_w = glb_world(hero, "dsp_die", out, explode=True)
    chest_w = glb_world(hero, "pcb_sternum_board", out, explode=True)
    cam(i, 3.2, dsp_w + Vector((0.09, -0.27, 0.09)), dsp_w, 34)
    cam(i, 4.4, dsp_w + Vector((0.05, -0.23, 0.08)), dsp_w, 38)
    cam(i, 5.5, chest_w + Vector((0.16, -0.34, 0.06)), chest_w, 36)
    cam(i, 6.5, chest_w + Vector((0.05, -0.42, 0.02)), chest_w + Vector((-0.04, 0, -0.03)), 32)
    cam(i, 8.0, exp_c + Vector((0.2, -0.9, 0.2)), exp_c + Vector((-0.05, 0, 0)), 30)
    cam(i, 8.7, exp_c + Vector((0.25, -0.85, 0.16)), exp_c + Vector((0.0, 0.4, 0.0)), 30)
    cam(i, 10.0, exp_c + Vector((0.3, -0.95, 0.18)), exp_c + Vector((0.2, 1.4, 0.05)), 30)
    belly_w = glb_world(hero, "pcb_belly_lowspeed", out)
    cell_w = (FXF + 0.02, hp.y, hp.z + 0.13)
    spine = beam("i2c_spine", belly_w, cell_w, 0.003, (0.3, 1.0, 0.4), emit=6.0)
    V(spine, i, 6.4, 8.0)
    for n in range(5):
        p = sph("i2c_bit", belly_w, 0.008, (0.3, 1.0, 0.4), emit=10.0, seg=8, rings=6)
        for c in range(3):
            ta = 6.4 + c * 0.5 + n * 0.1
            K(p, F(i, ta), loc=tuple(belly_w), interp="LINEAR")
            K(p, F(i, ta + 0.5), loc=cell_w, interp="LINEAR")
        V(p, i, 6.4, 8.0)
    ovt("CARD", "I2C -> SWITCH: TEMPERATURE / VCC / TX BIAS / TX POWER / RX POWER\nSUBJECT: HEART RATE ELEVATED", i, 6.5, 8.0)
    subj_walk(s, i, 0.5, 4.5, (-0.3, 7.6), (0.5, 12.0), PI)
    narr(i, [(1.8, 4.2, "The brain: a DSP from NARROWCOM or MERVALL."),
             (4.6, 7.8, "A microcontroller reports temperature, laser current and light levels."), (8.2, 9.8, "Hatched by OUTNOLIGHT.")])
    lab(i, 1.6, 8.0, "EXPLODED VIEW (HERO MODEL v1;\nPOLISH IN PROGRESS)")
    lab(i, 8.2, 10.0, "THE HATCHERY")
    fx(i, 8.2, 10.0, "[FX: crossfade hall -> hatchery assembly line -> hall]")

    # ---------------- S6: night street, casual perches, links
    i = 5
    subj_walk(s, i, 0.0, 2.0, (0.0, 12.0), (0.0, 14.4), PI)
    subj_walk(s, i, 2.0, 4.4, (0.0, 14.4), (0.3, 19.2), PI)
    subj_at(s, i, 8.6, (0.3, 19.2), PI)
    S_PT = Vector((0.3, 19.2, 1.55))
    flyers = [("hero", hero, (out, 0.0)), ("comp0", comps[0], cps[0]), ("comp1", comps[1], cps[1])]
    for j, (nm, b, (p0, y0)) in enumerate(flyers):
        perch, pyaw = FLYER_PERCHES[nm]
        bird_at(b, i, 0.0 + 0.2 * j, p0, y0)
        bird_at(b, i, 1.4 + 0.2 * j, (0.25 * (j - 1), 12.2, 1.85), PI / 2)
        bird_at(b, i, 2.6 + 0.2 * j, (0.7 * (j - 1) - 0.3, 14.9, 2.2), PI / 2)
        bird_at(b, i, 4.2 + 0.2 * j, perch + Vector((0, 0, 0.25)), pyaw)
        bird_at(b, i, 4.6 + 0.2 * j, perch, pyaw)
        b.perch, b.yaw = perch, pyaw
        head(b, i, 0.0, 0.0)
        head(b, i, 4.6 + 0.2 * j, stare(perch, pyaw, S_PT))
    comp_street = [Head(comps[k].perch, comps[k].yaw, rig=comps[k]) for k in range(2)]
    comp_street[1].blocked.append((F(i, 6.5), F(i, 9.0)))
    # link windows (beams + note)
    pairs = [("wire_18.2", "wire_19.0"), ("ledge_a", "ledge_b"), ("lamp1_top", "lamp1_arm")]
    pair_h = []
    for a_n, b_n in pairs:
        A, B = sh[a_n], sh[b_n]
        ha, hb = solve_link((A.pos, A.yaw), (B.pos, B.yaw))
        A.blocked.append((F(i, 2.4), F(i, 5.3)))
        B.blocked.append((F(i, 2.4), F(i, 5.3)))
        pair_h.append((A, B, ha, hb))
    upb = sh["roof_L"]
    upb.blocked.append((F(i, 5.0), F(i, 7.0)))
    cctv_links = [(CCTV[0], sh["ac_2"]), (CCTV[1], sh["lamp1_arm"])]
    for c, B in cctv_links:
        B.blocked.append((F(i, 6.5), F(i, 9.0)))
    obs = street_heads + comp_street
    for hd in obs:
        hd.blocked.append((F(i, 8.7), F(i, 10.0)))
    animate_heads(street_heads, F(i, 0.0), F(i, 4.0), rng)
    track_heads(street_heads, [(i, 2.0, (0.0, 14.4)), (i, 4.2, (0.3, 19.0))])
    rng_obs = random.Random(5)
    for hd in obs:   # observe the Subject: stares with small tilts, bobs and occasional neighbour glances
        f = F(i, 4.4) + rng_obs.randint(0, 8)
        while f < F(i, 8.6):
            if not hd.is_blocked(f) and not hd.is_blocked(f + 6):
                hv = stare(hd.pos, hd.yaw, S_PT) + rng_obs.uniform(-0.15, 0.15)
                hd.move(f, hv, rng_obs.uniform(-0.3, 0.3), rng_obs.choice((0.0, 0.0, 0.008)), rng_obs.randint(3, 6))
            f += rng_obs.randint(14, 32)
    for A, B, ha, hb in pair_h:
        for hd, hv in ((A, ha), (B, hb)):
            hd.key(F(i, 2.5), hv)
            hd.key(F(i, 5.15), hv)
    for n, (A, B, ha, hb) in enumerate(pair_h):
        e0, e1 = eye_w(A.pos, A.yaw, ha, "tx"), eye_w(B.pos, B.yaw, hb, "rx")
        o = beam("eye_link", e0, e1, 0.006, C["red"], emit=6.0)
        V(o, i, 2.9 + 0.2 * n, 5.1)
    sat_mid = SAT0.lerp(SAT1, 0.9)
    h_up = aim(upb.pos, upb.yaw, "tx", sat_mid)
    e_up = eye_w(upb.pos, upb.yaw, h_up, "tx")
    el = math.atan2(sat_mid.z - e_up.z, (sat_mid.xy - e_up.xy).length)
    upb.key(F(i, 5.2), h_up, -el)
    upb.key(F(i, 6.9), h_up, -el)
    up = beam("uplink", e_up, sat_mid, 0.03, C["red"], emit=6.0)
    V(up, i, 5.4, 6.8)
    for n, (c, B) in enumerate(cctv_links):
        hv = aim(B.pos, B.yaw, "rx", c)
        B.key(F(i, 6.6), hv)
        B.key(F(i, 8.9), hv)
        o = beam("cctv_link", c, eye_w(B.pos, B.yaw, hv, "rx"), 0.012, C["cyan"], emit=5.0)
        V(o, i, 6.9 + 0.25 * n, 8.7)
    c2 = comp_street[1]
    hv = aim(c2.pos, c2.yaw, "rx", CCTV[2])
    c2.key(F(i, 6.6), hv)
    c2.key(F(i, 8.9), hv)
    o = beam("cctv_link", CCTV[2], eye_w(c2.pos, c2.yaw, hv, "rx"), 0.012, C["cyan"], emit=5.0)
    V(o, i, 7.4, 8.7)
    hperch, hyaw = hero.perch, hero.yaw
    camp = hperch + Vector((0.75, -0.55, 0.42))
    for hd in obs:
        hv = stare(hd.pos, hd.yaw, camp)
        hd.key(F(i, 8.8), *hd.cur)
        hd.key(F(i, 9.2), hv)
        hd.key(F(i, 10.0), hv)
    h_cam = aim(hperch, hyaw, "tx", camp)
    head(hero, i, 8.6, stare(hperch, hyaw, S_PT))
    head(hero, i, 9.1, h_cam)
    head(hero, i, 10.0, h_cam)
    blink(hero, i, 9.35, 9.75, which=(0,))
    he = eye_w(hperch, hyaw, h_cam, "tx")
    hax = axis_w(hyaw, h_cam, "tx")
    cam(i, 0.0, exp_c + Vector((0.3, -0.95, 0.18)), exp_c + Vector((0.2, 1.4, 0.05)), 30)
    cam(i, 1.0, (0.15, 9.6, 1.75), (0.0, 13.0, 1.6), 28)
    cam(i, 2.2, (0.1, 12.6, 1.8), (0.1, 17.0, 1.7), 26)
    cam(i, 3.1, (-1.4, 14.4, 1.5), (-2.9, 16.0, 1.0), 26)
    cam(i, 3.9, (-1.9, 15.0, 3.7), LAMPS[0] + Vector((0.3, 0, LAMP_H + 0.1)), 28)
    cam(i, 4.8, (2.6, 16.8, 4.6), (4.9, 19.3, 5.2), 28)
    cam(i, 5.6, (1.2, 17.6, 5.0), upb.pos + Vector((0, 0, 0.3)), 26)
    cam(i, 6.5, (0.3, 17.4, 5.1), (-2.5, 26.0, 15.0), 20)
    cam(i, 7.6, (1.0, 20.6, 3.1), (4.2, 24.0, 4.2), 26)
    cam(i, 8.7, camp + Vector((0.4, -0.4, 0.15)), hperch + Vector((0.05, 0, 0.22)), 28)
    cam(i, 9.2, he + (camp - he).normalized() * 0.5, he, 34)
    cam(i, 10.0, he + hax * 0.07, he, 45)
    cam_end = (he + hax * 0.07, he, 45)
    narr(i, [(0.2, 2.0, "They follow you outside."), (2.6, 4.8, "Eye to eye, bird to bird."),
             (5.2, 6.8, "Up to the satellites."), (6.8, 8.8, "Down to every camera on the street.")])
    lab(i, 2.7, 8.7, "FREE-SPACE OPTICAL LINKS\n+ SATELLITE UPLINK + CCTV LINKS")
    lab(i, 9.3, 10.0, "BLINK = TX DISABLE")
    fx(i, 2.0, 5.0, "[FX: wing flaps, rain, neon reflections on wet street]")
    note(i, 2.75, 8.7)

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
    cam_jump(6, 0.0, cam_end, ((0, -306, 0), (0, -300, 0), 28))
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
    # overlay plane 6 cm in front of the lens (v0: 0.8 m), so walls closer than 0.8 m no longer hide captions/labels
    L.HOLD_Z = 0.06
    L.HOLD_S0 = L.HOLD_Z / 3.0
    L.setup_camera()
    L.CAM.data.clip_start = 0.004
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
    build(argv[0] if argv else "crude_v0_2.blend")
