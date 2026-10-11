"""Build the crude storyboard film v0.1 (DevLog-002 rev 2, DevLog-004 changes) as a single .blend.

Usage: blender -b --python scripts/build_crude_film_v0_1.py -- <out.blend>   (frozen copy of crude v0.1, 2026-10-10)
Crude on purpose: primitives, one continuous camera, captions, and [FX: ...] labels where effects will go.
World coordinates are absolute (no per-scene offset) so the camera can stay continuous:
  data hall: aisle along +y, rack faces at x = +-1.1 with open pigeonhole cells 0.32 m deep, end door at y = 13, street beyond;
  TX-eye macro set around x = 200, RX-eye macro set around x = 400 (entered through color fills).
Bird: primitive pigeon with the hero's proportions (DevLog-003 profile, 124 mm beak to tail), modelled in millimetres,
  bird-local frame: origin on the ground between the feet, +x = beak, +z up, -y = pigeon's RIGHT (TX).
  Every bird uses the same uniform scale S_BIRD. Eyes are lateral (axis 20 deg forward of pure lateral, stylized);
  right eye = TX (red 2x4 lenslets), left eye = RX (cyan 2x4). Heads yaw about a neck pivot; beams are solved per link.
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
    "hood": (0.16, 0.14, 0.2), "rack": (0.07, 0.075, 0.085), "floor": (0.09, 0.09, 0.1), "cell": (0.03, 0.03, 0.035),
    "frame": (0.13, 0.135, 0.15), "yellow": (0.98, 0.83, 0.12),
}
# ------------------------------------------------------------------ bird geometry constants (mm, bird-local)
MM = 0.001
S_BIRD = 2.6                         # one hall scale for every bird: 124 mm hero -> 0.32 m beak to tail on screen
P_H = Vector((36.0, 0.0, 70.0))      # neck pivot (head yaw axis)
HC = Vector((44.0, 0.0, 81.5))       # head centre (hero profile: head centre ~44 mm ahead of the feet, ~81 mm up)
HR = 12.0
EA = math.radians(20.0)              # eye axis 20 deg forward of pure lateral (stylized assumption, not on screen)
AX = {"tx": Vector((math.sin(EA), -math.cos(EA), 0.0)), "rx": Vector((math.sin(EA), math.cos(EA), 0.0))}
UX = {"tx": Vector((math.cos(EA), math.sin(EA), 0.0)), "rx": Vector((math.cos(EA), -math.sin(EA), 0.0))}
ANG = {"tx": -(PI / 2 - EA), "rx": PI / 2 - EA}     # eye-axis bearing in the head frame
EYE_C = {k: HC - P_H + AX[k] * 11.6 for k in AX}  # eye centre, head-local (relative to the neck pivot)
LENS_PITCH = 2.3
TOES = (("hallux", 172.0, 8.5), ("inner", -32.0, 12.0), ("middle", -4.0, 16.0), ("outer", 26.0, 12.5))  # hero (photo, C)
Z_CLAW, Z_PALM = -3.0, -1.2          # mating heights (mm below the toe pads): claws GND first, palm VCC, toe pads signal

# ------------------------------------------------------------------ hall layout
HERO_CELL = ("L", 7, 1, 1)
PARTNER_CELL = ("R", 7, 1, 1)
COMP_CELLS = [("L", 7, 2, 1), ("R", 7, 0, 1)]
ROWS_Z = (1.15, 1.45, 1.75)          # cell centres; cells 0.20 wide x 0.27 high, floors at ROWS_Z - 0.135
COLS_DY = (-0.24, 0.0, 0.24)
N_RACKS = 13
FACE = 1.1                           # rack face |x|
DEPTH = 0.32                         # pigeonhole depth
FEET_IN = 0.06                       # birds stand 6 cm inside the face: head, neck and breast poke out
CEIL = 4.2
TX_X, RX_X = 200.0, 400.0          # macro sets (v0: 1000 / 2000; moved closer so float32 precision keeps overlay text clean)


def lab(i, t0, t1, s): ovt("LAB", s, i, t0, t1)
def fx(i, t0, t1, s): ovt("FX", s, i, t0, t1)


def rack_y(k):
    return 1.4 + 0.8 * k


def cell_pos(side, k, row, col):
    """Feet origin of a bird standing in a pigeonhole (facing the aisle), and its yaw."""
    y = rack_y(k) + COLS_DY[col]
    z = ROWS_Z[row] - 0.135
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
def build_body(p, M, hero=False):
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
    lg = grp("leash")
    mseg("leash", (36, 0, 25), (51, 0, 17), 3.6, 0.7, M["leash"], lg)
    L.torus("leash_loop", tuple(Vector((53.8, 0, 15.6)) * MM), 2.8 * MM, 0.7 * MM, GREY, rot=(PI / 2, 0, 0), parent=lg,
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


def make_bird(name, hero=False):
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
    b.groups, b.feet = build_body(b.root, M, hero=hero)
    b.neck = empty(name + "_neck", tuple(P_H * MM), parent=b.root)
    b.h = build_head(b.neck, M, b.eye_tx, b.eye_rx, hero=hero)
    b.hinge, b.lids = b.h["hinge"], b.h["lids"]
    return b


def bird_at(b, idx, t, pos, yaw=0.0, pitch=0.0, interp="BEZIER"):
    K(b.root, F(idx, t), loc=tuple(pos), rot=(0, pitch, yaw), interp=interp)


def head(b, idx, t, hy, roll=0.0, interp="BEZIER"):
    K(b.neck, F(idx, t), rot=(roll, 0, hy), interp=interp)


def beak(b, idx, t, open_amt, interp="BEZIER"):
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


# ------------------------------------------------------------------ hall
HALL_VIS = [(0, 0, 10), (1, 0, 10), (2, 0, 10), (3, 0, 10), (4, 0, 10), (5, 0, 4.0)]


def hall_vis(o, skip=None):
    for i, a, b in HALL_VIS:
        V(o, i, a, b)
    return o


def build_hall():
    objs = []
    objs.append(plane("hall_floor", (0, 6.0, 0), (6.4, 13.6), mat(C["floor"], 0.4), rot=(0, 0, 0)))
    objs.append(plane("hall_ceiling", (0, 6.0, CEIL), (6.4, 13.6), mat(C["black"], 0.9), rot=(PI, 0, 0)))
    for k in range(4):
        objs.append(box("tile_line", (-0.5 + k * 0.33, 6.0, 0.001), (0.008, 13.0, 0.001), (0.15, 0.15, 0.17)))
    objs.append(box("wall_back", (0, -0.8, CEIL / 2), (6.4, 0.1, CEIL), C["black"]))
    objs.append(box("wall_end_l", (-1.9, 13.0, CEIL / 2), (2.4, 0.12, CEIL), C["black"]))
    objs.append(box("wall_end_r", (1.9, 13.0, CEIL / 2), (2.4, 0.12, CEIL), C["black"]))
    objs.append(box("wall_end_top", (0, 13.0, (2.5 + CEIL) / 2), (1.4, 0.12, CEIL - 2.5), C["black"]))
    objs.append(box("door_frame", (0, 12.95, 2.52), (1.5, 0.05, 0.04), C["cyan"], emit=4.0))
    for k in range(4):
        objs.append(box("ceil_strip", (0, 1.5 + 3.2 * k, CEIL - 0.05), (0.12, 2.2, 0.03), (0.85, 0.9, 1.0), emit=3.0))
    racks = {}
    xc = FACE + DEPTH / 2
    for side, sx in (("L", -1), ("R", 1)):
        for k in range(N_RACKS):
            y = rack_y(k)
            parts = [box("rack_lo", (sx * 1.6, y, 0.4925), (1.0, 0.78, 0.985), C["rack"], rough=0.6),
                     box("rack_hi", (sx * 1.6, y, 2.0575), (1.0, 0.78, 0.285), C["rack"], rough=0.6),
                     box("rack_back", (sx * (FACE + DEPTH + (2.1 - FACE - DEPTH) / 2), y, 1.45), (2.1 - FACE - DEPTH, 0.78, 0.93),
                         C["cell"], rough=0.8)]
            for zz in (1.0, 1.3, 1.6, 1.9):
                parts.append(box("shelf", (sx * xc, y, zz), (DEPTH, 0.78, 0.03), C["frame"], rough=0.5))
            for dy, w in ((-0.355, 0.03), (-0.12, 0.04), (0.12, 0.04), (0.355, 0.03)):
                parts.append(box("divider", (sx * xc, y + dy, 1.45), (DEPTH, w, 0.9), C["frame"], rough=0.5))
            for zz in (1.285, 1.585, 1.885):
                parts.append(box("cell_led", (sx * (FACE - 0.004), y, zz - 0.008), (0.006, 0.7, 0.005),
                                 C["cyan"] if (k + int(zz * 10)) % 2 else C["mag"], emit=3.0))
            for zz in (0.35, 0.55, 2.0):
                parts.append(box("led", (sx * (FACE - 0.002), y, zz), (0.004, 0.5, 0.012), C["cyan"] if k % 2 else C["mag"], emit=2.5))
            racks[(side, k)] = parts
            objs += parts
    for o in objs:
        hall_vis(o)
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
    for sx in (-1, 1):   # face fill so the birds in the cells read
        add_light("face_fill", "AREA", (sx * 0.3, 4.0, 2.4), 30, (0.75, 0.8, 1.0), size=1.2, rot=(0, -sx * PI / 2, 0))
    return racks


def crowd_cells():
    skip = {HERO_CELL, PARTNER_CELL, *COMP_CELLS}
    out = []
    for side in ("L", "R"):
        for k in range(N_RACKS):
            for row in range(3):
                for col in range(3):
                    if (side, k, row, col) not in skip:
                        out.append((side, k, row, col))
    return out


def pick_links(cells):
    """Cross-aisle links in the top and bottom rows (camera corridor stays clear in between)."""
    rnd = random.Random(5)
    free = set(cells)
    links = []
    for row, ks in ((2, range(0, N_RACKS)), (0, range(0, N_RACKS, 2))):
        for k in ks:
            tx_side, rx_side = ("L", "R") if (k + row) % 2 == 0 else ("R", "L")
            a = (tx_side, k, row, (k + row) % 3)
            for _ in range(10):
                k2 = max(0, min(N_RACKS - 1, k + rnd.choice((-2, -1, 0, 1, 2))))
                b = (rx_side, k2, row, rnd.randrange(3))
                if a in free and b in free:
                    links.append((a, b))
                    free.discard(a)
                    free.discard(b)
                    break
    return links


def build_crowd(subject_track):
    """Crowd birds as collection instances (body + head). Link birds hold the link stare; the rest track the Subject."""
    cb = bpy.data.collections.new("CROWD_BODY")
    ch = bpy.data.collections.new("CROWD_HEAD")
    proot = empty("crowd_body_root")
    _, ob = new_objs(build_body, proot, BM)
    move_to(cb, [proot] + ob)
    hroot = empty("crowd_head_root")
    _, oh = new_objs(build_head, hroot, BM, BM["eye_tx"], BM["eye_rx"])
    move_to(ch, [hroot] + oh)
    cells = crowd_cells()
    links = pick_links(cells)
    link_hy = {}
    for a, b in links:
        pa, pb = cell_pos(*a), cell_pos(*b)
        ha, hb = solve_link(pa, pb)
        link_hy[a], link_hy[b] = ha, hb
    inst = {}
    rnd = random.Random(7)
    for cell in cells:
        pos, yaw = cell_pos(*cell)
        bi = instancer("cb", cb, pos, yaw, S_BIRD)
        hi = instancer("ch", ch, bwm(pos, yaw, P_H), yaw, S_BIRD)
        inst[cell] = (bi, hi)
        for o in (bi, hi):
            if cell[0] == "L" and cell[1] == HERO_CELL[1]:   # hidden during the S2 cutaway of the hero rack
                for i, a_, b_ in HALL_VIS:
                    if i == 1:
                        V(o, 1, 0, 6.8)
                    elif i == 2:
                        V(o, 2, 0.8, 10)
                    else:
                        V(o, i, a_, b_)
            else:
                hall_vis(o)
        # idle twitches, then link pose or Subject tracking
        h0 = rnd.uniform(-0.35, 0.35)
        K(hi, F(0, 0), rot=(0, 0, yaw + h0), interp="CONSTANT")
        tt = rnd.uniform(1.5, 5.0)
        h1 = rnd.uniform(-0.6, 0.6)
        K(hi, F(0, tt), rot=(0, 0, yaw + h0), interp="BEZIER")
        K(hi, F(0, tt + 0.2), rot=(0, 0, yaw + h1), interp="BEZIER")
        if cell in link_hy:
            K(hi, F(0, 6.0), rot=(0, 0, yaw + h1), interp="BEZIER")
            K(hi, F(0, 6.4), rot=(0, 0, yaw + link_hy[cell]), interp="BEZIER")
        else:
            K(hi, F(0, 7.6), rot=(0, 0, yaw + h1), interp="BEZIER")
            for idx, t, sp in subject_track:
                K(hi, F(idx, t), rot=(0, 0, yaw + stare(pos, yaw, Vector((sp[0], sp[1], 1.55)))), interp="BEZIER")
    beams = []
    for n, (a, b) in enumerate(links):
        pa, pb = cell_pos(*a), cell_pos(*b)
        e0, e1 = eye_w(pa[0], pa[1], link_hy[a], "tx"), eye_w(pb[0], pb[1], link_hy[b], "rx")
        o = beam("hall_beam", e0, e1, 0.004, C["red"], emit=5.0)
        V(o, 0, 6.4 + 0.04 * n, 10)
        for i in (1, 2, 3, 4):
            V(o, i, 0, 10)
        beams.append(o)
    return inst, links


# ------------------------------------------------------------------ tree
TREE = [  # (label, x, extinct)
    ("THEROPOD\n1x9 SOLDERED", 0.0, False),
    ("ARCHAEOPTERYX\nGBIC 2000 REV", -0.42, False),
    ("MOA\nXENPAK 2001", 0.42, True),
    ("ELEPHANT BIRD\nCFP 2010", -0.42, True),
    ("EMU\nCFP2-DCO 2018", 0.42, False),
    ("OSTRICH\nXPO 2026", -0.42, False),
    ("CHICKEN\nSFP 2001", 0.42, False),
    ("HUMMINGBIRD\nQSFP-DD 2016", -0.42, False),
    ("ALBATROSS\n400ZR 2020", 0.42, False),
    ("PARROT\nCOMPATIBLES", -0.42, False),
    ("CUCKOO\nAEC 2019", 0.42, False),
    ("PASSENGER PIGEON\nQSFP+ LEGACY", -0.42, True),
    ("ROCK PIGEON\nQSFP28 2014", 0.42, False),
    ("DODO\nCPO", -0.42, True),
]
CLADES = [("PALAEOGNATHAE", 2, 5), ("GALLOANSERAE", 6, 6), ("NEOAVES (UNRESOLVED)", 7, 9), ("COLUMBAVES", 10, 13)]
TREE_Y = 5.0
TZ0, TDZ = 0.55, 0.2
NODE_Z = TZ0 + TDZ * len(TREE) + 0.0


def build_tree():
    hm = mat(C["holo"], 0.5, 2.5, unique=True)
    gm = mat((0.5, 0.6, 0.7), 0.5, 1.2, unique=True)
    nm = mat(C["red"], 0.3, 6.0, unique=True)
    objs = []
    trunk = cyl("trunk", (0, TREE_Y, (0.4 + NODE_Z) / 2), 0.012, NODE_Z - 0.4, C["holo"], verts=8)
    trunk.data.materials[0] = hm
    objs.append(trunk)
    for k, (lbl, x, ext) in enumerate(TREE):
        z = TZ0 + TDZ * k
        m = gm if ext else hm
        if k > 0:
            br = beam("branch", (0, TREE_Y, z - 0.08), (x, TREE_Y, z), 0.006, C["holo"], emit=2.0)
            br.data.materials[0] = m
            objs.append(br)
        ic = sph("leaf", (x, TREE_Y, z), 0.035, C["holo"], scale=(1.4, 1, 0.9), seg=12, rings=8)
        hd = sph("leaf_head", (x + 0.045, TREE_Y, z + 0.03), 0.018, C["holo"], seg=10, rings=6)
        ic.data.materials[0] = m
        hd.data.materials[0] = m
        objs += [ic, hd]
        objs.append(wlab(lbl, (x, TREE_Y - 0.02, z - 0.1), 0, 0, 6.2, size=0.062,
                         color=(0.75, 0.95, 1.0) if not ext else (0.6, 0.65, 0.7)))
    for lbl, k0, k1 in CLADES:
        zc = TZ0 + TDZ * (k0 + k1) / 2
        t = _text(lbl, 0.042, (1.0, 0.85, 0.3), (0.045, TREE_Y - 0.02, zc), None, "CENTER", "CENTER",
                  rot=(PI / 2, -PI / 2, 0), fnt="Arial Bold")
        objs.append(t)
    node = sph("hero_node", (0, TREE_Y, NODE_Z), 0.06, C["red"], scale=(1.4, 1, 0.9))
    node.data.materials[0] = nm
    objs.append(node)
    objs.append(wlab("CYBER PIGEON\n800G OSFP DR8", (0.0, TREE_Y - 0.03, NODE_Z + 0.14), 0, 0, 6.2, size=0.066, color=(1, 0.5, 0.45)))
    objs.append(wlab("RFC 1149 (1990-04-01)\nIP OVER AVIAN CARRIERS", (0.5, TREE_Y - 0.02, TZ0 + TDZ * 12 + 0.11), 0, 0, 6.2,
                     size=0.04, color=(1.0, 0.85, 0.3)))
    objs.append(wlab("GRAY = EXTINCT", (0.42, TREE_Y - 0.02, TZ0 - 0.12), 0, 0, 6.2, size=0.045, color=(0.6, 0.65, 0.7)))
    for o in objs:
        V(o, 0, 0, 6.2)
    fade_mats([hm, gm, nm], 0, [(5.3, 1.0), (6.1, 0.0)])


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


# ------------------------------------------------------------------ street
RC = Vector((0.0, 17.5, 0.0))         # ring centre (the Subject stops here)
RING_R = 2.2
N_RING = 8                           # 8 perches: neighbour bearing 67.5 deg from the beak ~ the 70 deg eye axis
PERCH_Z = 1.3
RING = [RC + Vector((RING_R * math.cos(math.radians(-67.5 + 45 * k)), RING_R * math.sin(math.radians(-67.5 + 45 * k)), PERCH_Z))
        for k in range(N_RING)]
LAMPS = [RC + Vector((4.4 * math.cos(math.radians(a)), 4.4 * math.sin(math.radians(a)), 0)) for a in (90, 210, 330)]
LAMP_Z = 3.25
CCTV = [Vector((-5.8, 15.5, 4.2)), Vector((5.8, 21.5, 4.6)), Vector((-5.8, 24.5, 3.9))]
SAT0, SAT1 = Vector((4, 38, 26)), Vector((9, 38, 27))


def build_street():
    i = 5
    objs = []
    gm = mat((0.075, 0.08, 0.1), 0.25)
    objs.append(plane("street", (0, 23, 0.0), (18, 20), gm, rot=(0, 0, 0)))
    objs.append(plane("sidewalk_l", (-4.6, 23, 0.01), (2.2, 20), mat((0.11, 0.11, 0.13), 0.6), rot=(0, 0, 0)))
    objs.append(plane("sidewalk_r", (4.6, 23, 0.01), (2.2, 20), mat((0.11, 0.11, 0.13), 0.6), rot=(0, 0, 0)))
    for k in range(6):
        objs.append(box("lane_mark", (0, 14.5 + 3 * k, 0.012), (0.1, 1.4, 0.004), (0.8, 0.8, 0.7), emit=0.6))
    win_cols = [C["orange"], C["cyan"], (1.0, 0.85, 0.5)]
    for sx in (-1, 1):
        for k in range(5):
            h = 6 + 3 * ((k * 7 + (sx > 0)) % 3)
            objs.append(box("bldg", (sx * 7.5, 15 + 4 * k, h / 2), (3.0, 3.6, h), (0.07, 0.07, 0.09)))
            col = [C["mag"], C["cyan"], C["orange"]][(k + (sx > 0)) % 3]
            objs.append(box("neon", (sx * 5.95, 15 + 4 * k, 3.0 + k % 2), (0.05, 2.4, 0.6), col, emit=8.0))
            for wz in range(2, int(h) - 1, 2):
                for wy in (-1.0, 0.0, 1.0):
                    if (wz + int(wy) + k) % 3 == 0:
                        objs.append(box("window", (sx * 5.99, 15 + 4 * k + wy, wz + 0.5), (0.02, 0.5, 0.7),
                                        win_cols[(wz + k) % 3], emit=2.0))
    for txt, x, y, z, col in (("RAMEN 24H", -5.9, 19, 4.9, C["orange"]), ("NO SIGNAL", 5.9, 23, 4.2, C["cyan"]),
                              ("LINK UP", -5.9, 27, 3.8, C["mag"])):
        objs.append(_text(txt, 0.6, col, (x + (0.05 if x < 0 else -0.05), y, z), None, "CENTER", "CENTER",
                          rot=(PI / 2, 0, PI / 2 if x < 0 else -PI / 2), fnt="Arial Black"))
    for k in range(60):
        objs.append(sph("star", (random.uniform(-40, 40), random.uniform(30, 80), random.uniform(20, 45)), 0.08, (1, 1, 1),
                        emit=6.0, seg=6, rings=4))
    for p in RING:
        objs.append(cyl("perch", (p.x, p.y, PERCH_Z / 2), 0.045, PERCH_Z, (0.22, 0.22, 0.25), verts=12))
        objs.append(cyl("perch_cap", (p.x, p.y, PERCH_Z - 0.01), 0.07, 0.02, C["cyan"], verts=16, emit=3.0))
    for p in LAMPS:
        objs.append(cyl("lamp_post", (p.x, p.y, LAMP_Z / 2), 0.06, LAMP_Z, (0.2, 0.2, 0.22)))
        objs.append(box("lamp_head", (p.x, p.y, LAMP_Z - 0.06), (0.36, 0.36, 0.06), (1.0, 0.85, 0.6), emit=5.0))
    for c in CCTV:
        objs.append(box("cctv", (c.x, c.y, c.z), (0.4, 0.22, 0.2), (0.85, 0.85, 0.88)))
        objs.append(cyl("cctv_lens", (c.x + (0.22 if c.x < 0 else -0.22), c.y, c.z), 0.07, 0.08, C["black"], axis="X"))
        objs.append(sph("cctv_led", (c.x + (0.25 if c.x < 0 else -0.25), c.y, c.z + 0.1), 0.025, C["red"], emit=8.0))
    sat = empty("satellite", SAT0)
    sb = [box("sat_body", (0, 0, 0), (0.8, 0.8, 0.8), (0.8, 0.75, 0.5), sat)]
    for sx in (-1, 1):
        sb.append(box("sat_panel", (sx * 1.3, 0, 0), (1.6, 0.05, 0.7), (0.15, 0.2, 0.6), sat, emit=0.6))
    sb.append(sph("sat_led", (0, -0.45, 0.3), 0.08, C["red"], sat, emit=10.0))
    K(sat, F(i, 0), loc=tuple(SAT0), interp="LINEAR")
    K(sat, F(i, 10), loc=tuple(SAT1), interp="LINEAR")
    for o in objs + sb:
        V(o, 4, 8.0, 10.0)
        V(o, i, 0, 10)
    lights = [add_light("street_moon", "SUN", (0, 20, 30), 1.6, (0.55, 0.65, 1.0), rot=(0.5, 0.2, 0.3)),
              add_light("street_fill", "AREA", (0, 17.5, 9), 1500, (0.55, 0.65, 1.0), size=12),
              add_light("street_fill2", "AREA", (0, 11.0, 6), 900, (0.6, 0.8, 1.0), size=8, rot=(-0.9, 0, 0))]
    for p in LAMPS:
        lights.append(add_light("street_lamp", "POINT", (p.x, p.y, LAMP_Z - 0.3), 450, (1.0, 0.6, 0.8)))
    for sx, col in ((-1, C["mag"]), (1, C["cyan"])):
        lights.append(add_light("neon_spill", "AREA", (sx * 5.5, 20, 3.5), 900, col, size=6, rot=(0, -sx * PI / 2, 0)))
    for lo in lights:
        V(lo, i, 0, 10)
        V(lo, 4, 8.0, 10.0)
    return sat


# ------------------------------------------------------------------ scenes
def main_build():
    bird_mats()
    s = make_subject()
    racks = build_hall()
    hero_rack = racks[("L", HERO_CELL[1])]
    track = [(0, 8.0, (0.3, 11.8)), (0, 8.7, (0.25, 10.4)), (0, 9.4, (0.2, 9.0)), (0, 10.0, (-0.25, 7.6)),
             (1, 2.0, (-0.25, 7.6)), (4, 0.5, (-0.3, 7.6)), (4, 4.5, (0.5, 12.0))]
    build_crowd(track)
    build_tree()
    build_tx_macro()
    build_rx_macro()
    build_hatchery()
    sat = build_street()

    hero = make_bird("HERO", hero=True)
    partner = make_bird("PARTNER")
    comps = [make_bird("COMP%d" % k) for k in range(2)]
    hp, hy0 = cell_pos(*HERO_CELL)
    pp, py = cell_pos(*PARTNER_CELL)
    cps = [cell_pos(*c) for c in COMP_CELLS]
    bird_at(partner, 0, 0, pp, py, interp="CONSTANT")
    for k, b in enumerate(comps):
        bird_at(b, 0, 0, cps[k][0], cps[k][1], interp="CONSTANT")
    # rig birds track the Subject like the crowd (S1 8-10, S2 start)
    for b, (pos, yaw) in ((partner, (pp, py)), (comps[0], cps[0]), (comps[1], cps[1])):
        head(b, 0, 0.0, 0.2, interp="CONSTANT")
        head(b, 0, 7.6, 0.2)
        for idx, t, sp in track[:5]:
            head(b, idx, t, stare(pos, yaw, Vector((sp[0], sp[1], 1.55))))
    h_tx, h_rx = solve_link((hp, hy0), (pp, py))

    # hero internals and ghost shell: visible only during the S5 exploded view
    for o in hero.groups["hero_parts"] + hero.h["hero_parts"]:
        V(o, 4, 1.5, 9.0)
    ghost_m = pmat((0.35, 0.4, 0.5), 0.3, 0.3)
    set_blend(ghost_m)
    for tt, a in ((2.0, 1.0), (2.6, 0.16), (8.0, 0.16), (8.6, 1.0)):
        key_alpha(ghost_m, F(4, tt), a)
    for o in hero.groups["shell_objs"] + [hero.h["skull_obj"]]:
        g = o.copy()
        bpy.context.scene.collection.objects.link(g)
        g.material_slots[0].link = "OBJECT"
        g.material_slots[0].material = ghost_m
        V(g, 4, 2.0, 8.6)
        for i in range(7):   # the opaque original is hidden while the ghost is shown
            if i == 4:
                V(o, 4, 0, 2.0)
                V(o, 4, 8.6, 10)
            else:
                V(o, i, 0, 10 if i < 6 else 2)

    # cutaway cage for the hero cell (S2 6.8 - S3 0.8): host board with contact pads, back wall, cage lip
    fz = hp.z
    cage = [box("host_board", (hp.x - 0.1, hp.y, fz - 0.003), (DEPTH, 0.22, 0.006), C["pcb"]),
            box("cage_back", (-(FACE + DEPTH), hp.y, fz + 0.135), (0.01, 0.22, 0.27), C["cell"]),
            box("cage_top", (hp.x - 0.1, hp.y, fz + 0.272), (DEPTH, 0.22, 0.004), (0.55, 0.56, 0.6), rough=0.3)]
    for sgn in (-1, 1):
        ank = bwm(hp, 0.0, (7, sgn * 8, 0))
        cage.append(box("host_pad", (ank.x + 0.008, ank.y, fz - 0.0002), (0.07, 0.034, 0.0012), C["gold"], rough=0.3))
        for nm, ang, ln in TOES:
            a = math.radians(ang) * sgn
            tipw = bwm(hp, 0.0, Vector((7, sgn * 8, 0)) + Vector((math.cos(a), math.sin(a), 0)) * (ln + 1.6))
            cage.append(cyl("gnd_socket", (tipw.x, tipw.y, fz + 0.0004), 0.003, 0.0012, C["black"], verts=10))
    lip = bwm(hp, 0.0, (59, 0, 77.0))
    cage.append(cyl("cage_lip", (lip.x, lip.y, lip.z), 1.2 * S_BIRD * MM, 0.2, (0.6, 0.62, 0.66), axis="Y", verts=12, rough=0.3))
    for sgn in (-1, 1):
        cage.append(box("lip_bracket", ((lip.x - FACE) / 2, lip.y + sgn * 0.1, lip.z), (abs(lip.x + FACE), 0.006, 0.006),
                        (0.6, 0.62, 0.66)))
    cage.append(add_light("cutaway_light", "POINT", (hp.x + 0.05, hp.y - 0.3, hp.z + 0.25), 9.0, (0.9, 0.95, 1.0)))
    for o in cage:
        V(o, 1, 6.8, 10.0)
        V(o, 2, 0, 0.8)
    for o in hero_rack:
        L._VIS[o] = [(F(0, 0), F(1, 6.8)), (F(2, 0.8), F(5, 4.0))]

    # ---------------- S1
    i = 0
    TY = TREE_Y - 1.15
    cam(i, 0.0, (0.0, TY, 0.62), (0, TREE_Y, 0.66), 24)
    cam(i, 1.5, (-0.06, TY, 1.35), (0, TREE_Y, 1.38), 24)
    cam(i, 3.0, (0.06, TY, 2.2), (0, TREE_Y, 2.24), 24)
    cam(i, 4.4, (0.0, TY, 3.1), (0, TREE_Y, 3.16), 24)
    cam(i, 5.0, (0.0, 4.35, 3.8), (0, TREE_Y, NODE_Z + 0.05), 24)
    cam(i, 5.7, (-0.15, 5.6, 3.6), (-0.9, 7.0, 1.6), 26)
    cam(i, 6.5, (-0.45, 6.25, 1.65), (-1.06, 7.1, 1.5), 28)
    cam(i, 7.6, (-0.4, 8.0, 1.62), (-1.1, 9.4, 1.48), 28)
    cam(i, 8.4, (-0.25, 8.7, 1.6), (0.2, 11.5, 1.45), 28)
    cam(i, 9.3, (-0.4, 7.6, 1.58), (-0.6, 8.6, 1.5), 28)
    hero_head_w = bwm(hp, 0.0, HC)
    cam(i, 10.0, (-0.62, 6.72, 1.6), hero_head_w + Vector((0, 0, -0.03)), 32)
    subj_at(s, i, 0, (0.3, 13.5), 0.0, interp="CONSTANT")
    subj_walk(s, i, 8.0, 10.0, (0.3, 11.8), (-0.25, 7.6), 0.0)
    subj_at(s, i, 10.0, (-0.25, 7.6), yaw_face((-0.25, 7.6), (-1.06, 7.0)))
    bird_at(hero, i, 0, hp, hy0, interp="CONSTANT")
    head(hero, i, 0.0, 0.0, interp="CONSTANT")
    head(hero, i, 7.6, 0.0)
    for idx, t, sp in track[:4]:
        head(hero, idx, t, stare(hp, hy0, Vector((sp[0], sp[1], 1.55))))
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
    held = Vector((-0.62, 7.0, 1.30))
    belly_pitch = -1.05
    above = hp + Vector((0, 0, 0.012))
    bird_at(hero, i, 0.0, hp, 0.0)
    bird_at(hero, i, 0.35, hp, 0.0)
    bird_at(hero, i, 1.4, held + Vector((0, 0, -0.02)), 0.0, pitch=0.15)
    bird_at(hero, i, 2.0, held, 0.0)
    bird_at(hero, i, 4.4, held, 0.0)
    bird_at(hero, i, 5.0, held, 0.0, pitch=belly_pitch)
    bird_at(hero, i, 6.6, held, 0.0, pitch=belly_pitch)
    bird_at(hero, i, 7.2, Vector((-0.95, 7.0, above.z)), 0.0)
    bird_at(hero, i, 7.7, above, 0.0, interp="LINEAR")
    bird_at(hero, i, 8.9, hp, 0.0)
    head(hero, i, 0.0, stare(hp, hy0, Vector((-0.25, 7.6, 1.55))))
    head(hero, i, 0.3, 0.0)
    for tt, hv in ((2.3, 0.0), (2.8, 1.25), (3.25, 1.25), (3.75, -1.25), (4.2, -1.25), (4.5, 0.0)):
        head(hero, i, tt, hv)
    beak(hero, i, 0.0, 0.0); beak(hero, i, 0.3, 1.0); beak(hero, i, 1.0, 0.0)
    beak(hero, i, 3.25, 0.0); beak(hero, i, 3.4, 1.0); beak(hero, i, 3.55, 0.0)
    beak(hero, i, 7.5, 0.0); beak(hero, i, 7.7, 1.0); beak(hero, i, 9.2, 1.0); beak(hero, i, 9.4, 0.0)
    for m in (hero.eye_tx, hero.eye_rx):  # unplugged = no power = dark eyes
        m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.04, 0.05, 0.06, 1)
        for tt, st, it in ((0.3, 6.0, "LINEAR"), (0.8, 0.0, "CONSTANT"), (9.0, 0.0, "LINEAR"), (9.25, 6.0, "CONSTANT")):
            key_strength(m, F(i, tt), st, it)
    # mating order: root drops 12 mm (hall scale) from 7.7 to 8.9; claws touch at 7.8 mm, palm at 3.1 mm, toe pads at 0
    t_g = 7.7 + 1.2 * (1 - (-Z_CLAW * S_BIRD) / 12.0)
    t_p = 7.7 + 1.2 * (1 - (-Z_PALM * S_BIRD) / 12.0)
    t_s = 8.9
    for name, tt, col in (("claw_gnd", t_g, C["brass"]), ("pad_pwr", t_p, C["gold"]), ("pad_sig", t_s, C["gold"])):
        m = hero.M[name]
        key_emit(m, F(i, tt - 0.05), col, 0.0, interp="LINEAR")
        key_emit(m, F(i, tt), col, 14.0, interp="LINEAR")
        key_emit(m, F(i, tt + 0.45), col, 0.8, interp="CONSTANT")
    for txt, tt, off in (("1 GND (CLAWS)", t_g, (0.05, -0.035, 0.006)), ("2 POWER (PALMS)", t_p, (0.05, -0.035, 0.019)),
                         ("3 SIGNAL (TOE PADS)", t_s, (0.05, -0.035, 0.032))):
        wlab(txt, hp + Vector(off), i, tt, 10.0, size=0.0075, color=(1.0, 0.85, 0.3))
    face_w = held + Vector((0.12, 0, 0.21))
    cam(i, 0.0, (-0.62, 6.72, 1.6), hero_head_w + Vector((0, 0, -0.03)), 32)
    cam(i, 1.0, (-0.32, 6.78, 1.6), (-0.72, 7.0, 1.47), 32)
    cam(i, 2.0, (-0.2, 7.0, 1.53), face_w + Vector((0, 0, -0.03)), 38)
    cam(i, 4.4, (-0.22, 7.0, 1.53), face_w + Vector((0, 0, -0.03)), 38)
    belly_t = bwm(held, 0.0, (10, 0, 14), belly_pitch)
    cam(i, 5.0, belly_t + Vector((0.26, 0.05, -0.02)), belly_t, 40)
    cam(i, 6.6, belly_t + Vector((0.25, -0.04, -0.03)), belly_t, 42)
    feet_t = hp + Vector((0.02, 0.0, 0.012))
    cam(i, 7.3, (-0.98, 6.72, 1.38), feet_t + Vector((0.0, 0, 0.03)), 34)
    cam(i, 8.0, (-1.06, 6.8, 1.335), feet_t, 40)
    cam(i, 8.9, (-1.065, 6.8, 1.333), feet_t, 42)
    beak_t = bwm(hp, 0.0, (58, 0, 77))
    cam(i, 9.5, (-0.92, 6.82, 1.49), beak_t + Vector((-0.03, 0, -0.01)), 36)
    cam(i, 10.0, (-0.88, 6.8, 1.52), beak_t + Vector((-0.04, 0, 0.0)), 34)
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
    head(hero, i, 0.0, 0.0)
    head(hero, i, 0.8, h_tx)
    head(partner, 1, 9.0, stare(pp, py, Vector((-0.25, 7.6, 1.55))))
    head(partner, i, 0.0, stare(pp, py, Vector((-0.25, 7.6, 1.55))))
    head(partner, i, 0.8, h_rx)
    e_tx = eye_w(hp, hy0, h_tx, "tx")
    e_rx = eye_w(pp, py, h_rx, "rx")
    lk = (e_rx - e_tx).normalized()
    side = lk.cross(Vector((0, 0, 1))).normalized()   # -y side of the link for a link along +x
    ride_off = side * 0.075 + Vector((0, 0, 0.035))
    cam(i, 0.0, (-0.88, 6.8, 1.52), beak_t + Vector((-0.04, 0, 0.0)), 34)
    cam(i, 0.9, e_tx + lk * 0.42 + Vector((0, 0, 0.01)), e_tx, 34)
    cam(i, 1.9, e_tx + lk * 0.025, e_tx - lk * 0.1, 40)
    fill(i, C["red"], [(1.4, 0.0), (1.9, 1.0), (2.0, 1.0), (2.5, 0.0)])
    X = TX_X
    cam_jump(i, 2.0, (e_tx + lk * 0.025, e_tx - lk * 0.1, 40), ((X + 15.0, -0.4, 1.9), (X + 12.4, 0, 0.4), 26))
    cam(i, 3.0, (X + 7.0, 0.9, 1.5), (X + 3.0, 0, 0.1), 26)
    cam(i, 4.0, (X + 1.6, 0.6, 1.2), (X - 0.4, 1.0, 0.25), 26)
    cam(i, 4.6, (X + 0.2, -0.9, 1.4), (X + 3.0, 0, 0.0), 26)
    cam(i, 5.8, (X + 3.4, -0.9, 1.25), (X + 5.6, 0.0, 0.0), 26)
    cam(i, 7.0, (X + 7.6, -2.0, 1.2), (X + 10.0, 0, 0.1), 26)
    cam(i, 7.7, (X + 10.4, -3.4, 1.2), (X + 11.4, 0, 0.4), 26)
    cam(i, 8.1, (X + 14.6, -3.2, 1.3), (X + 12.5, 0, 0.4), 26)
    cam(i, 8.5, (X + 15.0, -3.0, 1.35), (X + 12.5, 0, 0.4), 26)
    fill(i, C["red"], [(8.28, 0.0), (8.55, 1.0), (8.65, 1.0), (9.1, 0.0)])
    cam_jump(i, 8.6, ((X + 15.0, -3.0, 1.35), (X + 12.5, 0, 0.4), 26), (e_tx + lk * 0.12 + ride_off, e_rx, 30))
    cam(i, 10.0, e_tx + lk * 1.15 + ride_off, e_rx, 30)
    for r in range(2):
        for c in range(4):
            p0 = hwm(hp, hy0, h_tx, lens_local("tx", r, c))
            o = beam("link_beam", p0, p0 + (e_rx - e_tx), 0.0016, C["red"], emit=6.0)
            V(o, 2, 8.6, 10.0); V(o, 3, 0, 2.0); V(o, 4, 0, 1.4)
    subj_at(s, i, 0.0, (-0.3, 7.6), yaw_face((-0.3, 7.6), (-1.06, 7.0)), interp="CONSTANT")
    narr(i, [(0.2, 1.8, "Right eye: transmit."), (2.0, 4.4, "Two DFB lasers glow deep in its skull."),
             (4.6, 6.8, "Eight Mach-Zehnder modulators."), (7.0, 8.0, "Eight lenses."), (8.2, 9.4, "Eight beams.")])
    lab(i, 0.0, 1.4, "ONE-EYED STARE: RIGHT EYE (TX) ACROSS THE AISLE")
    lab(i, 2.0, 8.6, "TX SILICON PHOTONICS CHIP")
    lab(i, 8.6, 10.0, "FREE-SPACE LINK TO THE BIRD ACROSS THE AISLE")
    fx(i, 1.4, 2.5, "[FX: red fill = lens glow, macro emerges]")
    fx(i, 4.6, 7.0, "[FX: optical-mode glow along waveguides, PAM4 shimmer]")

    # ---------------- S4
    i = 3
    off_dir = (-lk * math.cos(math.radians(32)) + side * math.sin(math.radians(32))).normalized()
    cam(i, 0.0, e_tx + lk * 1.15 + ride_off, e_rx, 30)
    cam(i, 1.0, e_rx - lk * 0.35 + ride_off, e_rx, 32)
    cam(i, 1.9, e_rx + off_dir * 0.032 + Vector((0, 0, 0.008)), e_rx, 36)
    fill(i, C["cyan"], [(1.4, 0.0), (1.9, 1.0), (2.0, 1.0), (2.5, 0.0)])
    X = RX_X
    cam_jump(i, 2.0, (e_rx + off_dir * 0.032 + Vector((0, 0, 0.008)), e_rx, 36), ((X - 2.2, -2.3, 1.8), (X, 0, 0.4), 26))
    cam(i, 3.2, (X + 1.4, -2.9, 1.7), (X + 2.6, 0, 0.45), 26)
    cam(i, 4.5, (X + 2.4, -2.5, 1.5), (X + 3.2, 0, 0.4), 26)
    cam(i, 6.0, (X + 5.4, -1.6, 1.2), (X + 8.0, 0, 0.1), 26)
    fill(i, C["cyan"], [(5.9, 0.0), (6.35, 1.0), (6.45, 1.0), (6.9, 0.0)])
    subj_pt = Vector((-0.3, 7.6, 1.5))
    h_pov = aim(pp, py, "rx", subj_pt)
    head(partner, i, 6.3, h_rx, interp="CONSTANT")
    head(partner, i, 6.4, h_pov, interp="CONSTANT")
    head(partner, i, 9.4, h_pov)
    head(partner, i, 10.0, h_rx)
    pov_e = eye_w(pp, py, h_pov, "rx")
    pov_c = pov_e + (subj_pt - pov_e).normalized() * 0.05
    cam_jump(i, 6.4, ((X + 5.6, -1.6, 1.2), (X + 8.0, 0, 0.1), 26), (pov_c, subj_pt, 16))
    cam(i, 9.4, pov_c + (subj_pt - pov_c).normalized() * 0.12, subj_pt + Vector((0, 0, 0.05)), 22)
    cam(i, 10.0, e_rx - lk * 0.12 + ride_off, e_tx, 26)
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
        V(q, i, 6.6, 9.3)

    # ---------------- S5
    i = 4
    out = Vector((-0.42, 7.0, 1.28))
    whip_end = e_tx + lk * 0.35 + ride_off
    cam(i, 0.0, e_rx - lk * 0.12 + ride_off, e_tx, 26)
    cam(i, 0.7, whip_end, e_tx, 28)
    cam(i, 1.4, (-0.05, 6.45, 1.62), out + Vector((0.0, 0, 0.12)), 30)
    exp_c = out + Vector((0.0, 0, 0.13))
    cam(i, 2.4, exp_c + Vector((0.28, -0.82, 0.24)), exp_c, 30)
    bird_at(hero, i, 0.0, hp, 0.0)
    bird_at(hero, i, 0.9, hp, 0.0)
    bird_at(hero, i, 1.8, out, 0.0)
    bird_at(hero, i, 8.0, out, 0.0)
    head(hero, i, 0.0, h_tx)
    head(hero, i, 0.9, h_tx)
    head(hero, i, 1.8, 0.0)
    ex = {"wing_r": (-30, -16, 30), "wing_l": (-30, 16, 30), "tail": (-40, 0, -8), "leash": (16, 0, -6), "feet": (0, 0, -8),
          "legs": (0, 0, -4), "band": (0, 0, -4)}
    hx = {"skull": (0, 0, 24), "dsp": (0, 0, 8), "tx_optics": (0, -5, 0), "rx_optics": (0, 5, 0), "mcu": (0, 0, 0)}
    for gdict, table in ((hero.groups, ex), (hero.h, hx)):
        for name, d in table.items():
            g = gdict[name]
            dd = tuple(v * MM for v in d)
            K(g, F(i, 1.9), loc=(0, 0, 0), interp="BEZIER")
            K(g, F(i, 2.7), loc=dd, interp="BEZIER")
            K(g, F(i, 8.0), loc=dd, interp="BEZIER")
            K(g, F(i, 8.6), loc=(0, 0, 0), interp="BEZIER")
    dsp_w = bwm(out, 0.0, HC + Vector((0, 0, 4.2 + 8)))
    cam(i, 3.2, dsp_w + Vector((0.08, -0.26, 0.09)), dsp_w, 34)
    cam(i, 4.4, dsp_w + Vector((0.05, -0.22, 0.08)), dsp_w, 38)
    chest_w = bwm(out, 0.0, (19, 0, 43))
    cam(i, 5.5, chest_w + Vector((0.16, -0.34, 0.06)), chest_w, 36)
    cam(i, 6.5, chest_w + Vector((0.05, -0.42, 0.02)), chest_w + Vector((-0.04, 0, -0.03)), 32)
    cam(i, 8.0, exp_c + Vector((0.2, -0.9, 0.2)), exp_c + Vector((-0.05, 0, 0)), 30)
    cam(i, 8.7, exp_c + Vector((0.25, -0.85, 0.16)), exp_c + Vector((0.0, 0.4, 0.0)), 30)
    cam(i, 10.0, exp_c + Vector((0.3, -0.95, 0.18)), exp_c + Vector((0.2, 1.4, 0.05)), 30)
    for k, name in enumerate(("NARROWCOM", "MERVALL", "MINLINEAR")):
        t = _text(name, 1.6 * MM, C["gold"], tuple((HC - P_H + Vector((0, 0, 5.45))) * MM), hero.h["dsp"], "CENTER", "CENTER",
                  fnt="Arial Black")
        V(t, i, 2.6 + 0.75 * k, 3.35 + 0.75 * k if k < 2 else 8.0)
    ls = 1.0 / S_BIRD
    labels = [("DSP (BRAIN)", hero.h["dsp"], HC - P_H + Vector((12, -12, 8)), 2.8, 8.0),
              ("MCU", hero.h["mcu"], HC - P_H + Vector((-12, -8, -10)), 4.4, 8.0),
              ("TX PIC + 2 DFB", hero.h["tx_optics"], EYE_C["tx"] + Vector((6, -9, 6)), 2.9, 8.0),
              ("PIN ARRAY + TIA", hero.h["rx_optics"], EYE_C["rx"] + Vector((-6, 9, -8)), 2.9, 8.0),
              ("POWER", hero.groups["chest"], Vector((19, -8, 52)), 4.8, 8.0),
              ("CRYSTAL", hero.groups["chest"], Vector((26, -8, 58)), 5.0, 8.0),
              ("EEPROM", hero.groups["chest"], Vector((24, -8, 31)), 5.2, 8.0),
              ("FEET = 60 PADS", hero.groups["feet"], Vector((6, -16, -8)), 3.0, 8.0),
              ("FIN FEATHERS = HEATSINK", hero.groups["wing_r"], Vector((-14, -14, 44)), 2.9, 4.4),
              ("FLEX RIBBONS", hero.groups["flex"], Vector((14, -16, 30)), 5.6, 8.0)]
    for txt, par, off, a, b in labels:
        wlab(txt, tuple(Vector(off) * MM), i, a, b, size=0.017 * ls, color=(0.75, 0.95, 1.0), parent=par)
    # crystal ticks
    cm = hero.groups["crystal_m"]
    for k, tt in enumerate([x * 0.25 for x in range(18, 34)]):
        key_emit(cm, F(i, tt), (0.8, 0.8, 0.85), 3.0 if k % 2 == 0 else 0.0)
    # I2C snitch line: from the belly pads to the hero's cell in the switch
    feet_w = bwm(out, 0.0, (16, 0, 20)) + Vector((0, 0, -0.008 * S_BIRD))
    cell_w = (-1.09, 7.0, 1.45)
    spine = beam("i2c_spine", feet_w, cell_w, 0.003, (0.3, 1.0, 0.4), emit=6.0)
    V(spine, i, 6.4, 8.0)
    for n in range(5):
        p = sph("i2c_bit", feet_w, 0.008, (0.3, 1.0, 0.4), emit=10.0, seg=8, rings=6)
        for c in range(3):
            ta = 6.4 + c * 0.5 + n * 0.1
            K(p, F(i, ta), loc=tuple(feet_w), interp="LINEAR")
            K(p, F(i, ta + 0.5), loc=cell_w, interp="LINEAR")
        V(p, i, 6.4, 8.0)
    ovt("CARD", "I2C -> SWITCH: TEMPERATURE / VCC / TX BIAS / TX POWER / RX POWER\nSUBJECT: HEART RATE ELEVATED", i, 6.5, 8.0)
    wlab("HATCHED BY OUTNOLIGHT", tuple(Vector((6.6, -16, 13)) * MM), i, 8.6, 10.0, size=0.016 * ls, color=C["orange"],
         parent=hero.groups["band"])
    subj_walk(s, i, 0.5, 4.5, (-0.3, 7.6), (0.5, 12.0), PI)
    narr(i, [(1.8, 4.2, "The brain: a DSP from NARROWCOM or MERVALL."),
             (4.6, 7.8, "A microcontroller reports temperature, laser current and light levels."), (8.2, 9.8, "Hatched by OUTNOLIGHT.")])
    lab(i, 1.6, 8.0, "EXPLODED VIEW (CRUDE PLACEHOLDER PARTS;\nDETAILED HERO MODEL IN PROGRESS)")
    lab(i, 8.2, 10.0, "THE HATCHERY")
    fx(i, 8.2, 10.0, "[FX: crossfade hall -> hatchery assembly line -> hall]")

    # ---------------- S6
    i = 5
    subj_walk(s, i, 0.0, 2.0, (0.0, 12.0), (0.0, 14.4), PI)
    subj_walk(s, i, 2.0, 4.4, (0.0, 14.4), (RC.x, RC.y), PI)
    subj_at(s, i, 8.6, (RC.x, RC.y), PI)
    # ring: every bird faces the Subject; its right (TX) eye looks along the ring to the next bird's left (RX) eye
    ring_yaw = [math.atan2(RC.y - p.y, RC.x - p.x) for p in RING]
    ring_h = [0.0] * N_RING
    for _ in range(4):
        for k in range(N_RING):
            nk, pk = (k + 1) % N_RING, (k - 1) % N_RING
            ht = aim(RING[k], ring_yaw[k], "tx", eye_w(RING[nk], ring_yaw[nk], ring_h[nk], "rx"), ring_h[k])
            hr = aim(RING[k], ring_yaw[k], "rx", eye_w(RING[pk], ring_yaw[pk], ring_h[pk], "tx"), ring_h[k])
            ring_h[k] = 0.5 * (ht + hr)   # one head, two eyes: split the difference (near zero for 8 birds)
    flyers = {4: hero, 1: comps[0], 6: comps[1]}
    starts = {4: (out, 0.0), 1: cps[0], 6: cps[1]}
    cam_end = None
    # waiting street birds (instances) on the other perches + 3 lamp birds (uplink, CCTV)
    cbody, chead = bpy.data.collections["CROWD_BODY"], bpy.data.collections["CROWD_HEAD"]
    street_inst = []
    for k in range(N_RING):
        if k in flyers:
            continue
        bi = instancer("ring_b", cbody, RING[k], ring_yaw[k], S_BIRD)
        hi = instancer("ring_h", chead, bwm(RING[k], ring_yaw[k], P_H), ring_yaw[k], S_BIRD)
        street_inst.append((k, bi, hi))
    lamp_birds = []
    lamp_yaw = [math.atan2(RC.y - p.y, RC.x - p.x) for p in LAMPS]
    for k, p in enumerate(LAMPS):
        pos = Vector((p.x, p.y, LAMP_Z))
        bi = instancer("lamp_b", cbody, pos, lamp_yaw[k], S_BIRD)
        hi = instancer("lamp_h", chead, bwm(pos, lamp_yaw[k], P_H), lamp_yaw[k], S_BIRD)
        lamp_birds.append((pos, lamp_yaw[k], bi, hi))
        for o in (bi, hi):
            V(o, i, 0, 10)
            V(o, 4, 8.0, 10.0)
    for k, bi, hi in street_inst:
        for o in (bi, hi):
            V(o, i, 0, 10)
            V(o, 4, 8.0, 10.0)
    # flying birds: out of the hall, over the street, onto their perches
    for k, b in flyers.items():
        p0, y0 = starts[k]
        j = (1, 4, 6).index(k)
        bird_at(b, i, 0.0 + 0.2 * j, p0, y0)
        bird_at(b, i, 1.3 + 0.2 * j, (0.25 * (j - 1), 12.2, 1.85), PI / 2)
        bird_at(b, i, 2.6 + 0.2 * j, (0.6 * (j - 1), 14.8, 2.2), PI / 2)
        bird_at(b, i, 4.2 + 0.2 * j, RING[k] + Vector((0, 0, 0.25)), ring_yaw[k])
        bird_at(b, i, 4.8 + 0.2 * j, RING[k], ring_yaw[k])
        bird_at(b, i, 8.1, RING[k], ring_yaw[k])
        b.perch, b.yaw = RING[k], ring_yaw[k]
        head(b, i, 0.0, 0.0)
        head(b, i, 4.8 + 0.2 * j, ring_h[k])
        head(b, i, 8.15, ring_h[k])
    # ring and lamp heads (instances)
    for k, bi, hi in street_inst:
        K(hi, F(i, 0), rot=(0, 0, ring_yaw[k] + stare(RING[k], ring_yaw[k], (0.0, 14.0, 1.55))), interp="BEZIER")
        K(hi, F(i, 4.0), rot=(0, 0, ring_yaw[k] + stare(RING[k], ring_yaw[k], (RC.x, RC.y, 1.55))), interp="BEZIER")
        K(hi, F(i, 4.9), rot=(0, 0, ring_yaw[k] + ring_h[k]), interp="BEZIER")
        K(hi, F(i, 8.15), rot=(0, 0, ring_yaw[k] + ring_h[k]), interp="BEZIER")
    # lamp bird 0: right eye tilted up to the satellite (head roll); lamp birds 1, 2 and 0: left eye to a CCTV camera
    up_pos, up_yaw, _, up_hi = lamp_birds[0]
    sat_mid = SAT0.lerp(SAT1, 0.65)
    h_up = aim(up_pos, up_yaw, "tx", sat_mid)
    e_up = eye_w(up_pos, up_yaw, h_up, "tx")
    el = math.atan2(sat_mid.z - e_up.z, (sat_mid.xy - e_up.xy).length)
    roll_up = -el
    K(up_hi, F(i, 0), rot=(0, 0, up_yaw), interp="BEZIER")
    K(up_hi, F(i, 5.6), rot=(roll_up, 0, up_yaw + h_up), interp="BEZIER")
    K(up_hi, F(i, 8.15), rot=(roll_up, 0, up_yaw + h_up), interp="BEZIER")
    links_cctv = [(1, 0), (2, 1), (0, 2)]   # (lamp bird, CCTV camera)
    cctv_h = {}
    for lb, cc in links_cctv:
        pos, yaw, _, hi = lamp_birds[lb]
        if lb == 0:
            continue
        hcv = aim(pos, yaw, "rx", CCTV[cc])
        cctv_h[lb] = hcv
        K(hi, F(i, 0), rot=(0, 0, yaw), interp="BEZIER")
        K(hi, F(i, 6.4), rot=(0, 0, yaw + hcv), interp="BEZIER")
        K(hi, F(i, 8.15), rot=(0, 0, yaw + hcv), interp="BEZIER")
    for k in range(N_RING):
        nk = (k + 1) % N_RING
        a = eye_w(RING[k], ring_yaw[k], ring_h[k], "tx")
        b_ = eye_w(RING[nk], ring_yaw[nk], ring_h[nk], "rx")
        o = beam("ring", a, b_, 0.008, C["red"], emit=6.0)
        V(o, i, 5.0 + 0.12 * k, 8.15)
    up = beam("uplink", e_up, sat_mid, 0.03, C["red"], emit=6.0)
    V(up, i, 5.8, 8.15)
    for lb, cc in links_cctv:
        pos, yaw, _, _ = lamp_birds[lb]
        e = eye_w(pos, yaw, cctv_h.get(lb, h_up), "rx") if lb != 0 else hwm(pos, yaw, h_up, EYE_C["rx"])
        o = beam("cctv_link", CCTV[cc], e, 0.012, C["cyan"], emit=5.0)
        V(o, i, 6.5 + 0.25 * lb, 8.15)
    # heads to camera at the end (one-eyed stare into the lens)
    camp = RING[4] + Vector((0.55, -1.0, 0.45))
    for k, bi, hi in street_inst:
        K(hi, F(i, 8.7), rot=(0, 0, ring_yaw[k] + stare(RING[k], ring_yaw[k], camp)), interp="BEZIER")
        K(hi, F(i, 10.0), rot=(0, 0, ring_yaw[k] + stare(RING[k], ring_yaw[k], camp)), interp="BEZIER")
    for b in comps:
        head(b, i, 8.7, stare(b.perch, b.yaw, camp))
    h_cam = aim(hero.perch, hero.yaw, "tx", camp)
    head(hero, i, 8.7, h_cam)
    head(hero, i, 10.0, h_cam)
    blink(hero, i, 9.35, 9.75, which=(0,))
    he = eye_w(hero.perch, hero.yaw, h_cam, "tx")
    hax = axis_w(hero.yaw, h_cam, "tx")
    cam(i, 0.0, exp_c + Vector((0.3, -0.95, 0.18)), exp_c + Vector((0.2, 1.4, 0.05)), 30)
    cam(i, 1.0, (0.15, 9.6, 1.75), (0.0, 13.0, 1.6), 28)
    cam(i, 2.2, (0.0, 12.5, 1.85), (0.0, 16.5, 1.6), 26)
    cam(i, 2.9, (0.2, 13.7, 2.05), RC + Vector((0, 0, 1.2)), 25)
    cam(i, 3.7, (3.0, 14.3, 5.0), RC + Vector((0, 0, 1.2)), 24)
    cam(i, 5.0, (0.8, 16.6, 9.5), RC + Vector((0, 0.2, 0.8)), 22)
    cam(i, 6.6, (-1.0, 16.2, 3.2), RC + Vector((0.4, 0.8, 1.45)), 22)
    cam(i, 8.0, camp + Vector((0.2, -0.3, 0.1)), RING[4] + Vector((0, 0, 0.25)), 26)
    cam(i, 9.0, he + (camp - he).normalized() * 0.5, he, 34)
    cam(i, 10.0, he + hax * 0.07, he, 45)
    cam_end = (he + hax * 0.07, he, 45)
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
    build(argv[0] if argv else "crude_v0_1.blend")
