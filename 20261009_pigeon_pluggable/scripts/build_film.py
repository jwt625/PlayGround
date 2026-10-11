"""Build the pigeon film (production v2, DevLog-011; v1 frozen as build_film_v1.py) as a single .blend. Started from crude v0.2.3 (frozen: build_crude_film_v0_2_3.py).

Usage: blender -b --python scripts/build_film.py -- <out.blend>
Env: FILM_DEBUG=1 keeps the review overlays (time code, [FX: ...] notes); FILM_SAMPLES / FILM_W / FILM_H set the render tier.
Writes scripts/audio/scene_events.json (SFX cue list for workstream A) at build time.
World layout as in DevLog-006: hall (aisle +y, racks from R's layout), facade + bird flap at y = 13, night street beyond,
TX / RX macro sets (T / X subassemblies x1000) at x = 200 / 400. Hero = H's GLB; crowd = proxy from the hero GLB.
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
ASM_SLOT = 199                      # v4l (owner): assembly slot 184 -> 199 film frames (+0.5 s, test-pass tail holds)
S5_EXTRA = 30 + (ASM_SLOT - 92)     # v4 (W4) +30, v4k +92 -> 122: S5 runs 0-14.07 s, S6/S7 start 122 frames later than v3
S5_END = 10.0 + S5_EXTRA / 30.0     # 14.0667 s (v4: 11.0); every S5 tail key (flash, cut to S6, street) is relative to it
S7_LEN = float(os.environ.get("FILM_S7_LEN", "3.0"))   # disclaimer card length (A sets the fast radio-ad read)
TOTAL = 1800 + S5_EXTRA + int(round(S7_LEN * 30))


def F(idx, t):
    """Frame of scene idx at scene time t (v4: scenes after S5 start S5_EXTRA frames later)."""
    return 1 + idx * 300 + int(round(t * 30)) + (S5_EXTRA if idx >= 5 else 0)


L.F = F    # blender_lib's V / ovt / narr use the same mapping
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


FILM_DEBUG = os.environ.get("FILM_DEBUG", "0") == "1"
_NARR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio", "narration.json")
NARR = json.load(open(_NARR_PATH))["lines"] if os.path.exists(_NARR_PATH) else None


def narr(idx, segs):
    """Captions: from workstream A's narration.json (VO = master clock) when present, else the hand timing in segs."""
    if NARR is None:
        L.narr(idx, segs)


def narr_from_json():
    n = 0
    for ln in NARR or []:
        idx = int(ln["scene"]) - 1
        if idx >= 6 or ln.get("t0") is None:
            continue   # S7: the disclaimer card carries the text
        L.narr(idx, [(ln["t0"] - 10.0 * idx, ln["t1"] - 10.0 * idx, ln["show"])])
        n += 1
    print("CAPTIONS FROM narration.json:", n)


TEXTS = []       # v4f: scene headers + beam note drawn in post by A (outputs/film_v4_texts.json); nothing rendered


def lab(i, t0, t1, s):
    TEXTS.append(dict(kind="header", text=s.replace("\n", " / ") if False else s, f0=F(i, t0), f1=F(i, t1)))


def fx(i, t0, t1, s):
    if FILM_DEBUG:
        ovt("FX", s, i, t0, t1)


# ------------------------------------------------------------------ SFX cue list (scripts/audio/scene_events.json)
EVENTS = []


EV_OBJ = {}   # id(event) -> object whose animated location gives the event position (for pan)


def EV(idx, t, name, kind, pos=None, gain=None, dur=None, obj=None, params=None):
    """Event at scene idx, scene time t (s); film time = 10 * idx + t. pan is added from the camera at build end."""
    e = {"t": round(10.0 * idx + t + (S5_EXTRA / 30.0 if idx >= 5 else 0.0), 3), "name": name, "kind": kind}
    if params:
        e["params"] = params
    if obj is not None:
        EV_OBJ[id(e)] = obj
    if pos is not None:
        e["pos"] = [round(float(c), 3) for c in pos]
    if gain is not None:
        e["gain"] = round(gain, 3)
    if dur is not None:
        e["dur"] = round(dur, 3)
    EVENTS.append(e)


def EVF(frame, name, kind, pos=None, gain=None):
    EV(0, (frame - 1) / 30.0, name, kind, pos, gain)


def flaps(idx, t0, t1, name, pos=None, rate=0.18, gain=0.6, obj=None):
    t = t0
    while t < t1:
        EV(idx, t, name, "wing_flap", pos, gain, obj=obj)
        t += rate


def _fc_vec(obj, path, frame, default):
    ad = obj.animation_data
    v = list(default)
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path == path:
                v[fc.array_index] = fc.evaluate(frame)
    return Vector(v)


def add_pan():
    """pan = camera-space x of the event position over the half-width of the view (-1 left .. 1 right)."""
    n = 0
    for e in EVENTS:
        obj = EV_OBJ.get(id(e))
        if obj is not None and "pos" not in e:
            f = e["t"] * 30.0 + 1
            p = _fc_vec(obj, "location", f, obj.location)
            e["pos"] = [round(c, 3) for c in p]
        if "pos" not in e:
            continue
        f = e["t"] * 30.0 + 1
        c = _fc_vec(L.CAM, "location", f, L.CAM.location)
        tg = _fc_vec(L.TGT, "location", f, L.TGT.location)
        lens = L.CAM.data.lens
        ad = L.CAM.data.animation_data
        if ad and ad.action:
            for fc in ad.action.fcurves:
                if fc.data_path == "lens":
                    lens = fc.evaluate(f)
        fwd = (tg - c).normalized()
        right = fwd.cross(Vector((0, 0, 1)))
        if right.length < 1e-6:
            continue
        right.normalize()
        d = Vector(e["pos"]) - c
        depth = d.dot(fwd)
        if depth <= 0.01:
            e["pan"] = 0.0
            continue
        e["pan"] = round(max(-1.0, min(1.0, d.dot(right) / (depth * 18.0 / lens))), 3)
        n += 1
    print("EVENTS WITH PAN", n)


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
    zf = L.HOLD_Z
    L.HOLD.location = (0.0, 0.0, -zf)
    L.HOLD.keyframe_insert("location", frame=f)
    L._interp(L.HOLD, "location", f, "CONSTANT")
    L.HOLD.scale = ((zf / 3.0) * L.LENS0 / lens,) * 3
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
    # v4g: auto handles on the key after a jump took their slope from the jump and overshot (empty frames after the
    # cut into the TX macro); vector handles keep the move after the cut on its own path
    for idb in (L.CAM, L.TGT, L.CAM.data, L.HOLD):
        ad = idb.animation_data
        for fc in (ad.action.fcurves if ad and ad.action else []):
            for kp in fc.keyframe_points:
                if abs(kp.co.x - f) < 0.01:
                    kp.handle_left_type = "VECTOR"
                    kp.handle_right_type = "VECTOR"
            fc.update()


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
    i, t0, t1 = 2, 0.9, 8.8
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
    sg = _text("OUNOLIGHT HATCHERY", 0.12, C["mag"], (0, 10.05, 2.25), None, "CENTER", "CENTER", rot=(PI / 2, 0, 0), fnt="Arial Black")
    sg.data.materials[0] = sign_m
    objs.append(sg)
    for o in objs:
        V(o, i, 8.0, 10.0)
    fade_mats([belt_m, robot_m, item_m, chip_m, shell_m, eye_m, back_m, sign_m], i,
              [(8.2, 0.0), (8.6, 1.0), (9.5, 1.0), (9.95, 0.0)])


# ------------------------------------------------------------------ visibility windows
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERO_GLB = os.path.join(ROOT_DIR, "assets", "hero", "pigeon_osfp_dr8.glb")
# ------------------------------------------------------------------ hall layout from workstream R (DevLog-007)
DC_DIR = os.path.join(ROOT_DIR, "assets", "components", "datacenter")
LAYOUT_DY = 1.0          # R's layout shifted +1 m in y so the hero rack L07 sits at y = 7.0 (v0.1 framing of S2-S5)
LAYOUT = json.load(open(os.path.join(DC_DIR, "hall_layout.json"))) if os.path.exists(os.path.join(DC_DIR, "hall_layout.json")) else None
CELLS = {}
if LAYOUT:
    NECK_EXT = {}
    for _c in LAYOUT["cells"]["list"]:
        CELLS[(_c["rack"], _c["hook"])] = (Vector(_c["feet_origin"]) + Vector((0, LAYOUT_DY, 0)), math.radians(_c["bird_yaw_deg"]))
        NECK_EXT[(_c["rack"], _c["hook"])] = float(_c.get("neck_extension_m", 0.0))   # R v2 tiers: heads reach out of the recess
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
HALL_VIS = [(0, 5.0, 10), (1, 0, 10), (2, 0, 10), (3, 0, 10), (4, 0, S5_END), (5, 0, 4.0)]   # hall evaluated only when seen
STREET_VIS = [(0, 0, 10), (4, S5_END - 0.25, S5_END), (5, 0, 10)]   # v4g: not during the assembly shots
# v2 safe zones (DevLog-010 section 9 item 11; frame = +-1.93 x +-2.41 overlay units at the 3 m / 28 mm reference):
# headers in the top 15 % (y > 1.69), the bottom 18 % (y < -1.54) carries the captions and the small beam note only.
L.OVL["NOTE"] = dict(size=0.1, color=(1.0, 0.85, 0.3), loc=(-1.85, -2.36), ax="LEFT", ay="BOTTOM", shadow=True,
                     font="Arial Bold", outline=0.007)
L.WRAP["NOTE"] = 80
L.OVL["CAP"] = dict(L.OVL["CAP"], loc=(0, -1.95))
L.OVL["LAB"] = dict(L.OVL["LAB"], size=0.145, loc=(-1.85, 2.3))
L.WRAP["LAB"] = 36
L.OVL["HUD"] = dict(size=0.15, color=(0.35, 1.0, 0.45), loc=(-1.8, 0.95), ax="LEFT", ay="TOP", shadow=True, font="Arial Bold",
                    outline=0.012)
L.WRAP["HUD"] = 22
BEAM_NOTE = "* beams invisible (1311 nm); rendered for viewing"


def vis(o, wins):
    for i, a, b in wins:
        V(o, i, a, b)
    return o


def note(i, t0, t1):
    TEXTS.append(dict(kind="beamnote", text=BEAM_NOTE, f0=F(i, t0), f1=F(i, t1)))


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
            if r["id"] == "L06" and r["type"] == "gb300_nvl72":   # v4e: neighbour of the hero rack, no front bundles
                root, mp = real_copy(cols[r["type"]], "rack_" + r["id"], loc, math.radians(r["rotation_z_deg"]))
                racks[r["id"]] = [root] + list(mp.values())
                hid = {o__.name for c_ in cols[r["type"]].children_recursive if c_.hide_render for o__ in c_.all_objects}
                for nm_, o_ in mp.items():
                    if nm_ in hid:
                        L._VIS[o_] = [(TOTAL + 5, TOTAL + 6)]
                    elif o_.type in ("MESH", "CURVE"):
                        objs.append(o_)
                        if any(k in nm_ for k in ("front_fiber_jumpers", "front_cat6_patch", "front_velcro")):
                            L06_BUNDLES.append(o_)      # dissolved only during the S2 cutaway (v4e)
                continue
            if r["id"] == HERO_CELL[0]:   # v2: real (non-instanced) copy, so the cutaway can dissolve single parts
                root, mp = real_copy(cols[r["type"]], "rack_" + r["id"], loc, math.radians(r["rotation_z_deg"]))
                racks[r["id"]] = [root] + list(mp.values())
                HERO_RACK.update(mp)
                continue
            o = instancer("rack_" + r["id"], cols[r["type"]], loc, math.radians(r["rotation_z_deg"]))
            racks[r["id"]] = [o]
            objs.append(o)
        ov = LAYOUT.get("overhead")
        if ov and os.path.exists(os.path.join(ROOT_DIR, ov["asset"])):   # R: busways, fibre raceways, ladder trays (world frame)
            oc = append_collection(os.path.join(ROOT_DIR, ov["asset"]), "ASSET_overhead_infrastructure")
            if oc is not None:
                objs.append(instancer("overhead", oc, (0, LAYOUT_DY, 0), 0.0))
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
        ld.energy = 240
        ld.color = (0.8, 0.88, 1.0)
        lo = bpy.data.objects.new("hall_area", ld)
        bpy.context.scene.collection.objects.link(lo)
        lo.location = (0, 1.5 + 3.2 * k, CEIL - 0.15)
        vis(lo, HALL_VIS)
    for x, y, col in [(-1.0, 3.0, C["mag"]), (1.0, 6.0, C["cyan"]), (-1.0, 9.0, C["cyan"]), (1.0, 11.5, C["mag"])]:
        vis(add_light("accent", "POINT", (x, y, 2.6), 70, col), HALL_VIS)
    for sx in (-1, 1):   # rack-face fills (shadows are on in the film look, AgX)
        for yy in (3.0, 7.0, 10.5):
            vis(add_light("face_fill", "AREA", (sx * 0.4, yy, 2.0), 60, (0.75, 0.8, 1.0), size=2.0, rot=(0, -sx * PI / 2, 0)), HALL_VIS)
    return racks


PAD_M = {}
HERO_RACK = {}   # source object -> real copy of the hero's rack (L07)
L06_BUNDLES = []
FOCUS_KEYS = []  # v4e: (frame, world point) the DOF focus empty follows in the labelled shots
DOF_WINDOWS = []
HEADERS = []     # v4e: scene headers drawn in post during the DOF shots (in-render text would be defocused)
CALLOUTS = []    # v4e: 2D callouts drawn in post by A -> outputs/film_v4_callouts.json (text, part, frames, side)



def real_copy(coll, name, loc, rotz):
    """Real copy of an asset collection (objects share mesh data) under a root empty at loc / rotz. Returns (root, map)."""
    root = empty(name + "_root", tuple(loc))
    root.rotation_euler = (0.0, 0.0, rotz)
    nc = bpy.data.collections.new(name + "_real")
    bpy.context.scene.collection.children.link(nc)
    mp = {}
    for o in coll.all_objects:
        c = o.copy()
        nc.objects.link(c)
        mp[o] = c
    off = Matrix.Translation(-Vector(coll.instance_offset))
    for o, c in mp.items():
        if o.parent in mp:
            c.parent = mp[o.parent]
        else:
            c.parent = root
            c.matrix_parent_inverse = off
    return root, {o.name: c for o, c in mp.items()}


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


TXFX = {}


def build_tx_sub():
    """T subassembly (origin = eye-plate outer surface centre, +X = light out) at TX_X, x1000. Visible S3 0.9-8.8."""
    i, t0, t1 = 2, 0.9, 8.8      # v4g: present from the cut at 1.0 s (was 1.8: empty frames 631-654, owner's 0:21 note)
    txroot, new = import_glb_root(TX_GLB, (TX_X, 0, 0), SUB_K)
    for o in new:
        V(o, i, t0, t1)
    meta = json.load(open(TX_GLB.replace(".glb", "_parts.json")))
    extra = []
    dm = VX.data_beam_material("VFX_data_macro_tx", VX.TX_RED, strength=7.0, symbol_length=0.22, speed=3.0)   # F7
    for k, ln in enumerate(meta.get("lane_map", [])):
        extra.append(VX.make_data_beam("tx_out_beam_%d" % k, W_tx((0.12, ln["y_out"], ln["z_out"])), W_tx((25.0, ln["y_out"], ln["z_out"])),
                                       radius=0.05, material=dm))
    cores = [o for o in new if o.name.startswith("tx_beam_core_TX")]
    VX.apply_data_look(cores, dm)
    print("TX macro data beams", len(extra), "GLB cores", len(cores))
    TXFX.clear()
    try:   # v3: T's light-up + RF glow layer (DevLog-008 section 8); replaces the GLB's static tx_beam_* tubes
        sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "assets", "optics_tx"))
        import tx_fx
        oroot = next((o for o in new if o.name.startswith("OPTICS_TX_ROOT")), txroot)
        fx = tx_fx.build_tx_fx(oroot)
        fxo = list(fx["objs"].values()) if isinstance(fx["objs"], dict) else list(fx["objs"])
        for o in fxo:
            V(o, i, t0, t1)
        for o in new:
            if o.name.startswith("tx_beam_"):
                L._VIS[o] = [(TOTAL + 5, TOTAL + 6)]
        TXFX.update(fx=fx, mod=tx_fx)
        print("TX FX objects", [o.name for o in fxo])
    except Exception as ex:
        print("TX FX FAILED", ex)
    for o in extra:
        V(o, i, 7.3, t1)
    for lo in (add_light("tx_sub_front", "AREA", tuple(W_tx((8, -4, 0))), 2500, size=14, rot=(0, PI / 2, 0)),
               add_light("tx_sub_top", "AREA", tuple(W_tx((-1, -4, 8))), 900, size=12)):
        V(lo, i, t0, t1)
    for txt, p, a, b in (("CW DFB LASER x2 (DIMENTUM)", (-1.2, 0.5, -3.3), 1.4, 3.6), ("SI MICROLENSES (DFB -> PIC)", (-1.0, -1.1, -1.7), 2.2, 3.6),
                         ("SIPH PIC: 8 MACH-ZEHNDER MODULATORS", (-1.2, -6.0, 3.9), 3.9, 6.6),
                         ("HS DIFFERENTIAL WIREBONDS (GSSG)", (-1.3, -9.8, 3.9), 5.0, 6.6),
                         ("FAU COLLIMATOR + STAIRCASE PRISM\n(1x8 -> 2x4)", (-1.0, -1.8, 1.6), 6.6, 7.9),
                         ("2x4 LENSLET EYE", (0.4, 0.0, 1.7), 7.3, 8.25),
                         ("MICRO ISOLATOR (FARADAY)", (-0.9, -1.35, -3.05), 2.2, 3.6)):
        wlab(txt, W_tx(p), i, a, b, size=0.16, color=(0.75, 0.95, 1.0))   # v4j: original in-place labels restored (owner)


RXFX = {}


def build_rx_sub():
    """X subassembly (origin = lenslet front surface centre, light arrives along -X) at RX_X, x1000. Visible S4 1.8-6.8."""
    i, t0, t1 = 3, 1.8, 7.0
    rxroot, new = import_glb_root(RX_GLB, (RX_X, 0, RX_Z0), SUB_K)
    for o in new:
        V(o, i, t0, t1)
    RXFX.clear()
    try:   # v3: X's cleaned fixtures + traced cyan light + amber signal glow; replaces the straight rx_in/int/down beams
        sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "assets", "optics_rx"))
        import rx_fx
        oroot = next((o for o in new if "ROOT" in o.name.upper() and o.type == "EMPTY"), rxroot)
        rx_fx.rx_clean_fixtures(oroot)
        fx = rx_fx.build_rx_fx(oroot)
        fxo = list(fx["objs"].values()) if isinstance(fx["objs"], dict) else list(fx["objs"])
        for o in fxo:
            V(o, i, t0, t1)
        RXFX.update(fx=fx, mod=rx_fx)
        print("RX FX objects", [o.name for o in fxo], "root", oroot.name)
    except Exception as ex:
        print("RX FX FAILED", ex)
    lens_m = mat((0.35, 0.7, 0.85), 0.05, 0.25, unique=True)   # EEVEE draft has no refraction: show the silica lenslets as lit glass
    lens_m.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.75
    set_blend(lens_m)
    for o in new:
        if o.name.startswith("rx_lenslet_array") and o.type == "MESH":
            for slot in o.material_slots:
                slot.material = lens_m
    extra = []
    dmr = VX.data_beam_material("VFX_data_macro_rx", VX.TX_RED, strength=7.0, symbol_length=0.22, speed=3.0)   # F7
    for zr, dy in ((0.425, -0.25), (-0.425, 0.25)):   # rows; the Si interleaver plates shift the rows by -+0.25 mm onto the 0.5 mm PD pitch
        for y in (-1.5, -0.5, 0.5, 1.5):
            yl = y + dy
            extra.append(VX.make_data_beam("rx_in", W_rx((25.0, y, zr)), W_rx((0.05, y, zr)), radius=0.05, material=dmr))
            extra.append(VX.make_data_beam("rx_int", W_rx((-0.5, y, zr)), W_rx((-3.95, yl, zr)), radius=0.03, material=dmr))
            extra.append(VX.make_data_beam("rx_down", W_rx((-3.95, yl, zr)), W_rx((-3.93, yl, -1.2)), radius=0.03, material=dmr))
    for o in extra:
        if RXFX:
            L._VIS[o] = [(TOTAL + 5, TOTAL + 6)]
        else:
            V(o, i, t0, t1)
    for lo in (add_light("rx_sub_top", "AREA", tuple(W_rx((-3, 0, 6))), 600, size=10),
               add_light("rx_sub_front", "AREA", tuple(W_rx((6, -3, 2))), 900, size=8, rot=(0, PI / 2, 0))):
        V(lo, i, t0, t1)
    for m in bpy.data.materials:   # item 12: silicon interleaver plates read as Si (grey-blue, glossy) with lit edges
        if m.name.startswith("silicon_plate_ar"):
            b_ = m.node_tree.nodes.get("Principled BSDF") if m.node_tree else None
            if b_ is None:
                continue
            b_.inputs["Base Color"].default_value = (0.32, 0.36, 0.44, 1)
            b_.inputs["Metallic"].default_value = 0.55
            b_.inputs["Roughness"].default_value = 0.22
            nt = m.node_tree
            lw = nt.nodes.new("ShaderNodeLayerWeight")
            lw.inputs["Blend"].default_value = 0.35
            mul = nt.nodes.new("ShaderNodeMath")
            mul.operation = "MULTIPLY"
            mul.inputs[1].default_value = 3.0
            nt.links.new(lw.outputs["Facing"], mul.inputs[0])
            b_.inputs["Emission Color"].default_value = (0.55, 0.75, 1.0, 1)
            nt.links.new(mul.outputs[0], b_.inputs["Emission Strength"])
    for txt, p, a, b in (("2x4 LENSLET ARRAY", (0.3, -0.6, 1.25), 2.0, 2.8), ("SI INTERLEAVER PLATES: 2x4 -> 1x8", (-2.1, 0.0, 1.2), 2.6, 3.2),
                         ("1x8 PIN PD ARRAY", (-3.5, -2.4, -0.6), 3.05, 4.5),
                         ("OCTAL TIA", (-4.8, 0.0, -0.4), 4.7, 6.5), ("GSSG DIFFERENTIAL BONDS PD -> TIA", (-4.1, 2.2, -0.6), 4.7, 6.2)):
        wlab(txt, W_rx(p), i, a, b, size=0.15, color=(0.75, 0.95, 1.0))   # v4j: original in-place labels restored (owner)


# ------------------------------------------------------------------ facade (datacenter end wall, seen from the street)
FLAPS = []


def build_facade():
    fy = FACADE_Y
    x0f, x1f = FLAP.x - 0.35, FLAP.x + 0.35
    z0f, z1f = FLAP.z - 0.25, FLAP.z + 0.25
    objs = []
    for xa, xb, za, zb in ((-6.0, -0.7, 0, 7.0), (0.7, 6.0, 0, 7.0), (-0.7, x0f, 2.5, 7.0), (x1f, 0.7, 2.5, 7.0), (x0f, x1f, 2.5, z0f),
                           (x0f, x1f, z1f, 7.0)):
        objs.append(box("facade", ((xa + xb) / 2, fy, (za + zb) / 2), (xb - xa, 0.12, zb - za), (0.06, 0.063, 0.072), rough=0.7))
    objs.append(box("door_frame_out", (0, fy + 0.07, 2.52), (1.5, 0.03, 0.04), C["cyan"], emit=4.0))
    for sx in (-1, 1):
        objs.append(box("door_jamb", (sx * 0.72, fy + 0.07, 1.25), (0.03, 0.03, 2.5), C["cyan"], emit=2.0))
        objs.append(box("wall_lamp", (sx * 1.4, fy + 0.09, 2.7), (0.2, 0.06, 0.1), (1.0, 0.9, 0.7), emit=5.0))
        for kk in range(3):
            objs.append(box("vent", (sx * (3.0 + 0.9 * kk), fy + 0.065, 4.6), (0.7, 0.01, 0.45), (0.2, 0.2, 0.22), rough=0.4))
    for zz in (1.2, 2.0, 5.6, 6.4):
        objs.append(box("facade_band", (0, fy + 0.062, zz), (12.0, 0.005, 0.03), (0.25, 0.27, 0.3)))
    objs.append(_text("EQUONOX", 0.45, C["cyan"], (2.6, fy + 0.08, 3.6), None, "CENTER", "CENTER", rot=(PI / 2, 0, PI),
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
CONTACTS = []   # (frame, pos A, pos B) of eye-contact events from animate_heads


class Head:
    """Head of a crowd instance (instancer object) or of a rig (neck empty); keys yaw h, roll and a forward bob."""

    def __init__(self, pos, yaw, inst=None, rig=None, ext=0.0):
        self.pos, self.yaw, self.inst, self.rig, self.ext = Vector(pos), yaw, inst, rig, ext
        self.cur = (0.0, 0.0, 0.0)
        self.busy = -1
        self.blocked = []

    def key(self, f, h, roll=0.0, bob=0.0, interp="BEZIER"):
        if self.inst is not None:
            fwd = Vector((math.cos(self.yaw), math.sin(self.yaw), 0))
            K(self.inst, f, loc=tuple(bwm(self.pos, self.yaw, P_H) + fwd * (self.ext + bob)), rot=(roll, 0, self.yaw + h),
              interp=interp)
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
                        CONTACTS.append((f, A.pos.copy(), B.pos.copy()))
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
PROXY_KEEP = ("skin_", "wing_", "tail_fan", "eye_", "beak_", "leg_", "foot_")
PROXY_DROP = ("screws", "prism", "thermal", "band", "seams")
PROXY = {}


def strap_parts(p, M, kind):
    """Pull-tab strap out of the tail (bird-local mm) for the proxy birds: 'cell' (over the cell lip) or 'free' (short tag)."""
    out = []
    if kind == "cell":
        lip = FEET_IN / (S_BIRD * MM) + 3.0
        out += [mseg("leash", (-58, 0, 8), (-40, 0, 1.0), 3.6, 0.7, M["leash"], p),
                mseg("leash", (-40, 0, 1.0), (lip, 0, 1.0), 3.6, 0.7, M["leash"], p),
                mseg("leash", (lip, 0, 1.0), (lip + 1, 0, -15), 3.6, 0.7, M["leash"], p),
                L.torus("leash_loop", tuple(Vector((lip + 1, 0, -18.5)) * MM), 2.8 * MM, 0.7 * MM, GREY, rot=(PI / 2, 0, 0), parent=p,
                        material=M["leash"], segs=16)]
    else:
        out += [mseg("leash", (-60, 0, 9), (-74, 0, 3), 3.0, 0.6, M["leash"], p),
                L.torus("leash_loop", tuple(Vector((-76.5, 0, 2.0)) * MM), 2.4 * MM, 0.6 * MM, GREY, rot=(PI / 2, 0, 0), parent=p,
                        material=M["leash"], segs=16)]
    return out


def build_proxy_meshes(ratio_body=0.18, ratio_head=0.45):
    """Light proxy of the hero GLB for the crowd: shell, head with eyes, beak, wings, tail, legs and feet; internals dropped.
    Body mesh in the bird frame (m at hero scale); head mesh relative to the neck pivot. Decimated, shared by all instances."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=HERO_GLB)
    new = [o for o in bpy.data.objects if o not in before]
    hroot = next(o for o in new if o.name.startswith("HERO_ROOT"))
    hroot.rotation_mode = "XYZ"
    hroot.rotation_euler = (0, 0, PI / 2)
    bpy.context.view_layer.update()
    sel = [o for o in new if o.type == "MESH" and o.name.startswith(PROXY_KEEP) and not any(d in o.name for d in PROXY_DROP)]
    heads, bodies = [], []
    for o in sel:
        c = sum((o.matrix_world @ Vector(v) for v in o.bound_box), Vector()) / 8.0 / MM
        (heads if (c.x > 30 and c.z > 68) else bodies).append(o)

    def merged(objs, name, offset, ratio):
        parts = []
        for o in objs:
            me = o.data.copy()
            me.transform(o.matrix_world)
            if offset is not None:
                me.transform(Matrix.Translation(offset))
            ob = bpy.data.objects.new(name + "_part", me)
            bpy.context.scene.collection.objects.link(ob)
            parts.append(ob)
        for ob in bpy.context.view_layer.objects:
            ob.select_set(False)
        for ob in parts:
            ob.select_set(True)
        bpy.context.view_layer.objects.active = parts[0]
        bpy.ops.object.join()
        res = bpy.context.view_layer.objects.active
        res.name = name
        n0 = len(res.data.polygons)
        mod = res.modifiers.new("dec", "DECIMATE")
        mod.ratio = ratio
        bpy.ops.object.modifier_apply(modifier=mod.name)
        print("PROXY", name, "faces", n0, "->", len(res.data.polygons))
        return res
    body = merged(bodies, "proxy_body", None, ratio_body)
    head = merged(heads, "proxy_head", -P_H * MM, ratio_head)
    for o in new:
        bpy.data.objects.remove(o, do_unlink=True)
    PROXY["body"], PROXY["head"] = body.data, head.data
    for o in (body, head):
        bpy.data.objects.remove(o, do_unlink=True)
    return PROXY


def bird_collections():
    cols = {}
    if HERO_GLB and os.path.exists(HERO_GLB):
        build_proxy_meshes()
        for name, strap in (("CROWD_BODY", "free"), ("FREE_BODY", "free")):   # P1: pull tab is a short tail tab, not shown in use
            c = bpy.data.collections.new(name)
            root = empty(name.lower() + "_root")
            bo = bpy.data.objects.new(name.lower() + "_mesh", PROXY["body"])
            bpy.context.scene.collection.objects.link(bo)
            bo.parent = root
            parts = strap_parts(root, BM, strap)
            move_to(c, [root, bo] + parts)
            cols[name] = c
        ch = bpy.data.collections.new("CROWD_HEAD")
        hroot = empty("crowd_head_root")
        ho = bpy.data.objects.new("crowd_head_mesh", PROXY["head"])
        bpy.context.scene.collection.objects.link(ho)
        ho.parent = hroot
        move_to(ch, [hroot, ho])
        cols["CROWD_HEAD"] = ch
        return cols
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
    neck_m = BM["neck"]
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=1.0, depth=1.0, location=(0.5, 0, 0), rotation=(0, PI / 2, 0))
    tmp = bpy.context.active_object
    tmp.data.transform(tmp.matrix_world)   # unit cylinder along +x from 0 to 1
    neck_mesh = tmp.data
    neck_mesh.materials.append(neck_m)
    for p_ in neck_mesh.polygons:
        p_.use_smooth = True
    bpy.data.objects.remove(tmp, do_unlink=True)
    for cell in sorted(CELLS):
        if cell in skip:
            continue
        pos, yaw = cell_pos(*cell)
        ext = NECK_EXT.get(cell, 0.0) if LAYOUT else 0.0
        fwd = Vector((math.cos(yaw), math.sin(yaw), 0))
        bi = instancer("cb", cols["CROWD_BODY"], pos, yaw, S_BIRD)
        hi = instancer("ch", cols["CROWD_HEAD"], bwm(pos, yaw, P_H) + fwd * ext, yaw, S_BIRD)
        objs = [bi, hi]
        if ext > 0.005:   # neck extension (R v2 tiers): iridescent neck segment from the body's neck top to the head pivot
            nk = bpy.data.objects.new("neck_ext", neck_mesh)
            bpy.context.scene.collection.objects.link(nk)
            base = bwm(pos, yaw, P_H) + Vector((0, 0, -0.012))
            nk.location = base - fwd * 0.01
            nk.rotation_euler = (0, 0, yaw)
            nk.scale = (ext + 0.02, 9.5 * MM * S_BIRD, 9.5 * MM * S_BIRD)
            objs.append(nk)
        for o in objs:
            if cell[0] == HERO_CELL[0]:   # hidden during the S2 cutaway of the hero rack
                V(o, 0, 5.0, 10); V(o, 1, 0, 6.8); V(o, 2, 0.8, 10); V(o, 3, 0, 10); V(o, 4, 0, 10); V(o, 5, 0, 4.0)
            else:
                vis(o, HALL_VIS)
        hd = Head(pos, yaw, inst=hi, ext=ext)
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
TREE_C = Vector((0.0, 18.5, 0.0))   # v2: E hook HOOK_tree_base_y18p5 (1.5 m further from the DC sign)
TZ0, TDZ = 0.55, 0.2
NODE_Z = TZ0 + TDZ * len(TREE)
BRANCH_COL = {"L": C["gold"], "C": C["mag"], "U": C["orange"], "P": (0.6, 0.65, 0.7), None: (0.75, 0.95, 1.0)}


def build_tree():
    hmats = holo_tree_materials()
    hm, gm, nm, lm, cm = hmats["holo"], hmats["gray"], hmats["node"], hmats["gold"], hmats["mag"]
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

    objs.append(wlab("CYBER PIGEON", (0.0, ty + 0.03, NODE_Z + 0.2), 0, 0, 6.2, size=0.075, color=(1, 0.5, 0.45)))
    objs.append(wlab("(5) 800G OSFP DR8: 8 LANES", (0.0, ty + 0.03, NODE_Z + 0.12), 0, 0, 6.2, size=0.055, color=C["gold"]))
    for txt, z, col in (("GOLD (0)-(6) = PLUGGABLE LINEAGE", TZ0 - 0.13, C["gold"]), ("MAGENTA = COHERENT", TZ0 - 0.2, C["mag"]),
                        ("GRAY = EXTINCT", TZ0 - 0.27, (0.6, 0.65, 0.7))):
        objs.append(wlab(txt, (0.62, ty + 0.02, z), 0, 0, 6.2, size=0.04, color=col))
    for o in objs:
        V(o, 0, 0, 6.2)
    for m in (hm, gm, nm, lm, cm):   # E's hologram has its own flicker/scanlines; fade out as the camera leaves
        VX.key_input(m, "Alpha", F(0, 5.0), 1.0)
        VX.key_input(m, "Alpha", F(0, 6.0), 0.0)


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
    objs.append(plane("street", (0, 23, 0.0), (18, 20.5), mat((0.03, 0.032, 0.04), 0.18), rot=(0, 0, 0)))
    for sx in (-1, 1):
        objs.append(plane("sidewalk", (sx * 4.75, 23, 0.01), (2.5, 20.5), mat((0.055, 0.055, 0.065), 0.6), rot=(0, 0, 0)))
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
            for dy in (-0.8, -0.25, 0.3, 0.85):   # fake wet-street reflection streaks under the sign
                objs.append(box("wet_streak", (sx * 4.9, 15 + 4 * k + dy, 0.013), (2.0, 0.07, 0.002), col, emit=0.9))
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
    lights = [add_light("street_moon", "SUN", (0, 20, 30), 0.45, (0.65, 0.72, 0.9), rot=(0.5, 0.2, 0.3)),
              add_light("street_fill", "AREA", (0, 19.5, 9), 220, (0.7, 0.74, 0.85), size=12),
              add_light("street_fill2", "AREA", (0, 15.0, 6), 110, (0.7, 0.75, 0.85), size=8, rot=(0.9, 0, 0))]
    for p in LAMPS:
        s = 1 if p.x < 0 else -1
        lights.append(add_light("street_lamp", "POINT", (p.x + s * 1.05, p.y, LAMP_H - 0.4), 380, (1.0, 0.78, 0.55)))
        pool = cyl("lamp_pool", (p.x + s * 1.05, p.y, 0.014), 0.9, 0.002, (1.0, 0.75, 0.5), verts=24, emit=0.35)
        objs.append(pool)
        vis(pool, STREET_VIS)
    for sx, col in ((-1, C["mag"]), (1, C["cyan"])):
        lights.append(add_light("neon_spill", "AREA", (sx * 5.5, 21, 3.5), 650, col, size=6, rot=(0, -sx * PI / 2, 0)))
    lights.append(add_light("facade_wash", "AREA", (0, 15.5, 5.5), 45, (0.8, 0.85, 1.0), size=6, rot=(-0.9, 0, 0)))
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


def make_proxy_bird(name):
    b = Bird()
    b.root = empty(name, (0, -500, 0))
    b.root.scale = (S_BIRD,) * 3
    bo = bpy.data.objects.new(name + "_body", PROXY["body"])
    bpy.context.scene.collection.objects.link(bo)
    bo.parent = b.root
    strap_parts(b.root, BM, "free")
    b.neck = empty(name + "_neck", tuple(P_H * MM), parent=b.root)
    ho = bpy.data.objects.new(name + "_head", PROXY["head"])
    bpy.context.scene.collection.objects.link(ho)
    ho.parent = b.neck
    b.hinge, b.lids, b.hinge_base = None, [], None
    return b


def stretch_strap(name, a, b, width, thick, m):
    """Thin box from empty a to empty b (Copy Location + Stretch To)."""
    me = bpy.data.meshes.new(name)
    w, t = width, thick
    vs = [(-w / 2, 0, -t / 2), (w / 2, 0, -t / 2), (w / 2, 0, t / 2), (-w / 2, 0, t / 2),
          (-w / 2, 1, -t / 2), (w / 2, 1, -t / 2), (w / 2, 1, t / 2), (-w / 2, 1, t / 2)]
    me.from_pydata(vs, [], [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)])
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    o.data.materials.append(m)
    c1 = o.constraints.new("COPY_LOCATION")
    c1.target = a
    c2 = o.constraints.new("STRETCH_TO")
    c2.target = b
    c2.rest_length = 1.0
    c2.volume = "NO_VOLUME"
    return o


def column_label(hero, sub, text, idx, t0, t1, col_mm, side, size=0.013, color=(0.75, 0.95, 1.0)):
    """Label in a fixed column beside the hero (bird frame, mm) with a leader line to the part (follows the explode)."""
    o = hero.by(sub)
    if o is None:
        print("LABEL SKIPPED (part not found):", sub)
        return
    mw = o.matrix_world
    c = sum((mw @ Vector(v) for v in o.bound_box), Vector()) / 8.0 if o.type == "MESH" else mw.translation.copy()
    anchor = empty("lbl_anchor", tuple(mw.inverted() @ c), parent=o)
    tip = empty("lbl_tip", tuple(Vector(col_mm) * MM), parent=hero.root)
    ax = "LEFT" if side > 0 else "RIGHT"
    t = _text(text, size / S_BIRD, color, tuple((Vector(col_mm) + Vector((side * 2.0, 0, 0))) * MM), hero.root, ax, "CENTER",
              billboard=True, fnt="Arial Bold")
    sh = _text(text, size / S_BIRD, (0, 0, 0), (size / S_BIRD * 0.05, -size / S_BIRD * 0.05, -0.0005), t, ax, "CENTER",
               fnt="Arial Bold")
    ld = stretch_strap("leader", tip, anchor, 0.00035, 0.00035, mat((0.45, 0.75, 0.85), 0.5, 1.2))
    dot = sph("leader_dot", (0, 0, 0), 0.0016, color, anchor, emit=3.0, seg=8, rings=6)
    dot.scale = (1.0 / S_BIRD,) * 3
    for x in (t, sh, ld, dot):
        V(x, idx, t0, t1)


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
_PLATE_M = {}


def add_plate(t_obj, idx, t0, t1, pad=(0.22, 0.5)):
    """Dark semi-opaque plate with a thin light outline behind a billboard text (child of the text, same window)."""
    if "fill" not in _PLATE_M:
        for k, col, a, st in (("fill", (0.0, 0.0, 0.0), 0.8, 1.0), ("edge", (0.45, 0.78, 0.95), 0.95, 1.2)):
            m = bpy.data.materials.new("label_plate_" + k)     # unlit: emission + transparency (no rack light on the plate)
            m.use_nodes = True
            nt_ = m.node_tree
            for n_ in list(nt_.nodes):
                nt_.nodes.remove(n_)
            oo = nt_.nodes.new("ShaderNodeOutputMaterial")
            tr_ = nt_.nodes.new("ShaderNodeBsdfTransparent")
            em_ = nt_.nodes.new("ShaderNodeEmission")
            em_.inputs["Color"].default_value = (*col, 1)
            em_.inputs["Strength"].default_value = st
            mx_ = nt_.nodes.new("ShaderNodeMixShader")
            mx_.inputs[0].default_value = a
            nt_.links.new(tr_.outputs[0], mx_.inputs[1])
            nt_.links.new(em_.outputs[0], mx_.inputs[2])
            nt_.links.new(mx_.outputs[0], oo.inputs["Surface"])
            if hasattr(m, "surface_render_method"):
                m.surface_render_method = "BLENDED"
            _PLATE_M[k] = m
    bpy.context.view_layer.update()
    bb = t_obj.bound_box
    x0, x1 = min(v[0] for v in bb), max(v[0] for v in bb)
    y0, y1 = min(v[1] for v in bb), max(v[1] for v in bb)
    w, h = (x1 - x0), (y1 - y0)
    hp_ = max(h, 0.6 * t_obj.data.size)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    outs = []
    for k, grow, dz in (("edge", 1.0 + 0.05, -0.04), ("fill", 1.0, -0.02)):
        pw, ph = (w + 2 * pad[0] * hp_) * grow, (hp_ + 2 * pad[1] * hp_ * 0.5) * grow
        q = plane("label_plate_" + k, (cx, cy, dz * t_obj.data.size), (pw, ph), _PLATE_M[k], rot=(0, 0, 0), parent=t_obj)
        q.visible_shadow = False
        V(q, idx, t0, t1)
        outs.append(q)
    return outs


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


# ------------------------------------------------------------------ production integration (C characters, E environment/VFX)
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))
sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "anim"))
import vfx as VX                    # noqa: E402  (workstream E, DevLog-013)
import pigeon_rig as PR             # noqa: E402  (workstream C, DevLog-012)
import behaviors as BH              # noqa: E402
import subject as SJ                # noqa: E402
import keys as KY                   # noqa: E402

ENV_DIR = os.path.join(ROOT_DIR, "assets", "components", "environment")
NECK_EXT_SCALE = 0.5                # coordinator: half of R's suggested neck extension (tier 2 0.126 m read goose-like)
ENV_SKIP = ("ENV_hooks", "EXT_R_bird_flap_vent")   # hooks are data only; the flap vent is appended separately (animated)
ENV_ALWAYS = ("ENV_dc_facade",)     # the facade is also the hall's end wall
ENV_INST = {}


def env_hooks(name):
    d = json.load(open(os.path.join(ENV_DIR, name + ".json")))
    return {h["name"]: h for h in d["hooks"]}


def append_env(name, wins):
    """Append E's street layout and instance each child collection; the street children get `wins`, the facade is always on."""
    root_name = "ENV_" + name
    with bpy.data.libraries.load(os.path.join(ENV_DIR, name + ".blend"), link=False) as (src, dst):
        dst.collections = [root_name]
    root = bpy.data.collections.get(root_name)
    insts = []
    for ch in root.children:
        if ch.name.split(".")[0] in ENV_SKIP:
            continue
        off = (0.0, 1.5, 0.0) if ch.name.split(".")[0] == "ENV_tree_projector" else (0, 0, 0)   # tree base y 18.5 (v2)
        o = instancer("env_" + name + "_" + ch.name, ch, off, 0.0)
        ENV_INST[ch.name.split(".")[0] + ("" if name == "street_opening" else "@end")] = o
        if ch.name.split(".")[0] in ENV_ALWAYS:
            if name == "street_opening":
                insts.append(o)
                continue
            V(o, 6, 0, 0)          # only the opening's facade is used (identical in both layouts)
            continue
        vis(o, wins)
        insts.append(o)
    return insts


def append_flap_vent():
    """R's bird flap vent at E's hook (0, 12.94, 3.25), rot Z -90 deg; the curtain swings open as the bird and camera pass."""
    vc = append_collection(os.path.join(DC_DIR, "bird_flap_vent.blend"), "ASSET_bird_flap_vent")
    root = empty("bird_flap_root", (0.0, FACADE_Y - 0.06, FLAP.z))
    root.rotation_euler = (0, 0, -PI / 2)
    flaps = []
    for o in vc.all_objects:
        if o.parent is None:
            o.parent = root
        if o.name.startswith("bird_flap_vent_flap"):
            o.rotation_mode = "XYZ"
            flaps.append(o)
    bpy.context.scene.collection.children.link(vc)
    return flaps


def append_hatchery(loc):
    """E's hatchery (2 s seamless loop) placed at loc; visible only for the S5 flashback. Returns (root, objects)."""
    path = os.path.join(ENV_DIR, "hatchery.blend")
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.collections = [c for c in src.collections if c.startswith("HATCH_")]
    objs = []
    for c in dst.collections:
        if c is None:
            continue
        bpy.context.scene.collection.children.link(c)
        objs += list(c.all_objects)
    root = next((o for o in objs if o.name.startswith("hatchery_root")), None)
    if root is None:
        root = next((o for o in bpy.data.objects if o.name.startswith("hatchery_root")), None)
        if root is not None and root.name not in bpy.context.scene.objects:
            bpy.context.scene.collection.objects.link(root)
    if root is not None:
        loose = [o for o in objs if o.parent is None and o is not root]
        for o in loose:     # keep everything with the set when it moves
            o.parent = root
            o.matrix_parent_inverse = root.matrix_world.inverted()
        print("HATCHERY objects", len(objs), "reparented", len(loose))
        root.location = loc
    for o in objs:
        V(o, 4, 8.0, 10.0)
    return root, objs


def holo_tree_materials():
    """E's hologram material for the tree (cyan / gray extinct / gold lineage / magenta coherent / red hero node)."""
    return {"holo": VX.hologram_material("VFX_tree_holo", C["holo"], strength=2.0, base=0.55),
            "gray": VX.hologram_material("VFX_tree_gray", (0.5, 0.6, 0.7), strength=1.2, base=0.5, seed=1.0),
            "gold": VX.hologram_material("VFX_tree_gold", C["gold"], strength=2.0, base=0.6, seed=2.0),
            "mag": VX.hologram_material("VFX_tree_mag", C["mag"], strength=2.0, base=0.6, seed=3.0),
            "node": VX.hologram_material("VFX_tree_node", C["red"], strength=4.0, base=0.8, seed=4.0)}


def vbeam(name, p0, p1, idx, t0, t1, color=None, radius=0.004, ramp=6):
    """E's beam (view-facing core, travelling pulses), grown from the source; visible [t0, t1] in scene idx."""
    o = VX.make_beam(name, tuple(p0), tuple(p1), radius=radius, color=color or VX.TX_RED)
    VX.key_beam(o, F(idx, t0), F(idx, t1), ramp=ramp, grow=True)
    V(o, idx, t0, t1 + 0.05)
    return o


class RigBird:
    """Adapter so the scene helpers (bird_at, head, beak, blink) drive a C rig."""

    def __init__(self, r, hero=False):
        self.r = r
        self.root = r.root
        self.neck = r.head
        self.lids = [r.lid.get("tx"), r.lid.get("rx")]
        self.hinge = None
        self.hinge_base = None


def make_hero_c():
    """Hero = H's GLB on C's rig (head/wings/tail/legs/lids pivots). Adds the S5 data the scene needs (rest centroids,
    explode offsets in parent space, contact and eye materials)."""
    r = PR.build_hero_rig("HERO", scale=S_BIRD, loc=(0, -500, 0), yaw=0.0, hide_leash=False)
    b = RigBird(r, hero=True)
    b.glb, b.by = r.glb, r.by
    bpy.context.view_layer.update()
    Mr = r.root.matrix_world.copy()
    Mri = Mr.inverted()
    b.cen = {}
    for o in r.glb:
        if o.type == "MESH":
            c = sum((o.matrix_world @ Vector(v) for v in o.bound_box), Vector()) / 8.0
        else:
            c = o.matrix_world.translation.copy()
        b.cen[o.name] = (Mri @ c) / MM          # bird frame, mm (hero scale)
    glb_set = set(r.glb)
    b.top = [o for o in r.glb if o.parent not in glb_set]
    hroot = next((o for o in r.glb if o.name.startswith("HERO_ROOT")), None)
    H3 = hroot.matrix_world.to_3x3() if hroot is not None else Mr.to_3x3()
    b.ex_local = {}
    b.ex_bird = {}
    for o in b.top:
        ex = o.get("explode")
        if ex is None or o.parent is None:
            continue
        w = H3 @ Vector(tuple(ex))                                  # world offset at rest
        b.ex_local[o.name] = o.parent.matrix_world.to_3x3().inverted() @ w
        b.ex_bird[o.name] = (Mr.to_3x3().inverted() @ w) / MM       # bird frame mm
    b.hinge = r.by("HOOK_beak_hinge")
    if b.hinge is not None:
        b.hinge.rotation_mode = "XYZ"
        b.hinge_base = b.hinge.rotation_euler.copy()
    lt, lr = r.by("eye_tx_lenslet_array"), r.by("eye_rx_lenslet_array")
    b.eye_tx = lt.active_material if lt else BM["eye_tx"]
    b.eye_rx = lr.active_material if lr else BM["eye_rx"]
    # v2: H's keyable materials from the .blend (the GLB drops node set-ups such as rim_strength)
    swap = {}
    for o in r.glb:
        for slot in getattr(o, "material_slots", []):
            m = slot.material
            if m is not None and m.name.split(".")[0].startswith("hero_"):
                nm = m.name.split(".")[0]
                if nm not in swap:
                    swap[nm] = hero_blend_material(nm) or m
                slot.material = swap[nm]
    b.hm = swap
    print("HERO keyable materials", sorted(swap))
    b.m_pads = {"tx": swap.get("hero_foot_pads_R_tx"), "rx": swap.get("hero_foot_pads_L_rx")}
    b.m_rim = {"tx": swap.get("hero_eye_rim_R_tx"), "rx": swap.get("hero_eye_rim_L_rx")}
    b.m_band, b.m_neck = swap.get("hero_leg_band_glow_amber"), swap.get("hero_neck_rim_accent")
    b.m_wings = [m for m in (swap.get("hero_wing_fins_fadeable"), swap.get("hero_wing_covert_fadeable")) if m is not None]
    # contacts (OSFP MSA pad lengths: GND longest, mates first; VCC second; signal shortest, last; pin map: GND = talon tips,
    # VCC + low-speed = belly pads, TX / RX signal = toe pads)
    gnd_m = pmat((0.75, 0.55, 0.2), 0.3, 1.0)
    gnd_m.name = "hero_gnd_tips"
    b.contact = {"gnd": [gnd_m], "pwr": [], "sig": [m for m in b.m_pads.values() if m is not None]}
    for o in r.glb:
        if o.type != "MESH":
            continue
        if "talon_tip_gnd" in o.name:
            for slot in o.material_slots:
                slot.material = gnd_m
        elif "belly_lowspeed_vcc_pads" in o.name:
            for slot in o.material_slots:
                if slot.material is not None:
                    slot.material = slot.material.copy()
                    b.contact["pwr"].append(slot.material)
    # DSP stamp twin (MERVALL) from H's .blend
    b.stamp = {"NARROWCOM": r.by("dsp_lid_marking"), "MERVALL": None}
    mv = hero_blend_material("hero_dsp_mark_MERVALL")
    if b.stamp["NARROWCOM"] is not None and mv is not None:
        t = b.stamp["NARROWCOM"].copy()
        t.data = t.data.copy()
        t.data.materials[0] = mv
        for c in b.stamp["NARROWCOM"].users_collection:
            c.objects.link(t)
        b.stamp["MERVALL"] = t
    return b


def glb_world_c(hero, sub, pos, yaw=0.0, explode=False):
    o = hero.by(sub)
    if o is None:
        return bwm(pos, yaw, HC)
    c = hero.cen[o.name].copy()
    if explode and o.name in hero.ex_bird:
        c += hero.ex_bird[o.name]
    return bwm(pos, yaw, c)


def eyew(rb, k, f):
    """World eye position of a C rig at frame f (evaluates the scene)."""
    return Vector(rb.r.eye_world(k, f)) if isinstance(rb, RigBird) else Vector(rb.eye_world(k, f))


def lens_offsets(k):
    """8 lenslet offsets (head-local mm) of eye k, relative to the eye centre."""
    return [UX[k] * ((c - 1.5) * LENS_PITCH) + Vector((0, 0, (r - 0.5) * LENS_PITCH * 1.05)) for r in range(2) for c in range(4)]


_HDM = {}


def _hall_data_mat():
    if "m" not in _HDM:
        _HDM["m"] = VX.data_beam_material("VFX_data_hall", VX.TX_RED, strength=9.0, symbol_length=0.02, speed=1.0)
    return _HDM["m"]


def bundle(rb, k, f, target_centre, idx, t0, t1, name):
    """8 parallel beams from the lenslets of rig rb's eye k (at frame f) to the same offsets around target_centre."""
    e = eyew(rb, k, f)
    M3 = rb.r.head.matrix_world.to_3x3()
    for n, off in enumerate(lens_offsets(k)):
        d = M3 @ (off * MM)
        o = VX.make_data_beam("%s_%d" % (name, n), tuple(e + d), tuple(Vector(target_centre) + d), radius=0.0016, material=_hall_data_mat())
        VX.key_beam(o, F(idx, t0), F(idx, t1), ramp=4, grow=True)
        V(o, idx, t0, t1 + 0.05)
    return e


# ------------------------------------------------------------------ v2 helpers (R hero cell, dissolving cutaway, H keyable materials)
HERO_BLEND = os.path.join(ROOT_DIR, "assets", "hero", "pigeon_osfp_dr8.blend")


def append_hero_cell(pos, yaw):
    """R's detailed mating cell (DevLog-007 section 11) at the hero's slot. Returns dict of parts."""
    with bpy.data.libraries.load(os.path.join(DC_DIR, "hero_cell_mating.blend"), link=False) as (src, dst):
        dst.collections = ["ASSET_hero_cell_mating"]
    c = dst.collections[0]
    bpy.context.scene.collection.children.link(c)
    objs = list(c.all_objects)
    by = {o.name.split(".")[0]: o for o in objs}
    root = by["ROOT_hero_cell_mating"]
    root.location = tuple(pos)
    root.rotation_euler = (0.0, 0.0, yaw)
    pads = {k: bpy.data.materials.get("MAT_dc_hero_pad_" + k) for k in ("TX", "RX", "GND", "VCC", "LS")}
    return dict(objs=objs, by=by, root=root, lip=by["hero_cell_mating_lip_pivot"], cutaway=by["hero_cell_mating_side_wall_cutaway"],
                pads=pads)


def principled(m):
    return next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if m and m.node_tree else None


def key_input_val(m, inp_name, frame, val, interp="LINEAR", node=None):
    """Key a node input (Principled by default) of material m."""
    n = node or principled(m)
    if n is None:
        return
    inp = n.inputs[inp_name]
    inp.default_value = val
    inp.keyframe_insert("default_value", frame=frame)
    for fc in m.node_tree.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            if int(round(kp.co[0])) == frame:
                kp.interpolation = interp


def pulse(m, keys, color=None):
    """Emission pulse keys [(frame, strength)] on a Principled material (optionally setting the emission colour)."""
    b = principled(m)
    if b is None:
        return
    if color is not None:
        b.inputs["Emission Color"].default_value = (*color, 1)
    for f, st in keys:
        key_input_val(m, "Emission Strength", f, st)


_FADED = {}


def faded_material(m, name_tag):
    """Copy of m with a Transparent mix (input 0 = opacity, keyable) in BLENDED mode; one copy per source material."""
    if m is None:
        return None
    if m.name in _FADED:
        return _FADED[m.name]
    f = m.copy()
    f.name = m.name + "_" + name_tag
    if not f.use_nodes or f.node_tree is None:
        f.use_nodes = True
    nt = f.node_tree
    mo = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output), None) or \
        next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if mo is None or not mo.inputs["Surface"].links:
        _FADED[m.name] = None
        return None
    srcs = mo.inputs["Surface"].links[0].from_socket
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    mix = nt.nodes.new("ShaderNodeMixShader")
    mix.name = "cutaway_fade"
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(srcs, mix.inputs[2])
    nt.links.new(mix.outputs[0], mo.inputs["Surface"])
    for attr, val in (("surface_render_method", "BLENDED"), ("blend_method", "BLEND")):
        if hasattr(f, attr):
            try:
                setattr(f, attr, val)
            except (TypeError, AttributeError):
                pass
    _FADED[m.name] = f
    return f


def dissolve(objs, keys, win, base_wins, tag="cut"):
    """Cutaway dissolve without pops: each object keeps its opaque self outside `win` (frames) and a twin with faded
    materials (opacity keys [(frame, 0..1)]) inside it. base_wins = the object's normal visibility windows (frames)."""
    twins = []
    for o in objs:
        if o.type not in ("MESH", "CURVE", "FONT"):
            continue
        t = o.copy()
        for c in o.users_collection:
            c.objects.link(t)
            break
        for k, slot in enumerate(t.material_slots):
            fm = faded_material(slot.material, tag)
            if fm is not None:
                slot.link = "OBJECT"
                slot.material = fm
        L._VIS[t] = [win]
        L._VIS[o] = [w for a, b in base_wins for w in ((a, min(b, win[0])), (max(a, win[1]), b)) if w[1] > w[0]]
        twins.append(t)
    for fm in set(_FADED.values()):
        if fm is None or fm.get("faded_keys"):
            continue
        fm["faded_keys"] = 1
        mix = fm.node_tree.nodes["cutaway_fade"]
        for f, a in keys:
            mix.inputs[0].default_value = a
            mix.inputs[0].keyframe_insert("default_value", frame=f)
    return twins


def seg_hits_box(p0, p1, mn, mx):
    """Segment p0-p1 intersects the axis-aligned box [mn, mx] (slab test)."""
    t0, t1 = 0.0, 1.0
    for i in range(3):
        d = p1[i] - p0[i]
        if abs(d) < 1e-9:
            if p0[i] < mn[i] or p0[i] > mx[i]:
                return False
            continue
        a, b = (mn[i] - p0[i]) / d, (mx[i] - p0[i]) / d
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return False
    return True


def world_box(o):
    pts = [o.matrix_world @ Vector(v) for v in o.bound_box]
    return Vector([min(p[i] for p in pts) for i in range(3)]), Vector([max(p[i] for p in pts) for i in range(3)])


def occluders(objs, cam_p, targets, pad=0.0):
    """Objects whose world bbox meets any sight line from cam_p to the target points."""
    out = []
    for o in objs:
        if o.type not in ("MESH", "CURVE", "FONT"):
            continue
        mn, mx = world_box(o)
        mn -= Vector((pad,) * 3)
        mx += Vector((pad,) * 3)
        if any(seg_hits_box(cam_p, Vector(t), mn, mx) for t in targets):
            out.append(o)
    return out


def hero_blend_material(name):
    """Material from H's .blend (keyable node set-ups that the GLB drops, e.g. the MERVALL stamp, rim_strength)."""
    m = bpy.data.materials.get(name + "_hb")
    if m is not None:
        return m
    with bpy.data.libraries.load(HERO_BLEND, link=False) as (src, dst):
        if name not in src.materials:
            return None
        dst.materials = [name]
    m = dst.materials[0]
    m.name = name + "_hb"
    return m


def cam_state(f):
    c = _fc_vec(L.CAM, "location", f, L.CAM.location)
    tg = _fc_vec(L.TGT, "location", f, L.TGT.location)
    lens = L.CAM.data.lens
    ad = L.CAM.data.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if fc.data_path == "lens":
                lens = fc.evaluate(f)
    return c, tg, lens


def in_view(p, f, margin=1.05, max_d=40.0):
    """World point p inside the camera frustum at frame f (portrait 540 x 674, horizontal sensor fit 36 mm)."""
    c, tg, lens = cam_state(f)
    fwd = (tg - c).normalized()
    right = fwd.cross(Vector((0, 0, 1)))
    if right.length < 1e-6:
        return False
    right.normalize()
    up = right.cross(fwd)
    d = Vector(p) - c
    depth = d.dot(fwd)
    if depth < 0.03 or depth > max_d:
        return False
    hx = depth * 18.0 / lens
    hy = hx * 674.0 / 540.0
    return abs(d.dot(right)) <= hx * margin and abs(d.dot(up)) <= hy * margin


GATED = {"link_blip", "link_drop", "landing_thump", "footstep", "wing_flap", "hop", "cell_dock", "cell_undock", "beak_latch",
         "head_servo", "contact_mate", "cctv_whir", "sat_ping", "robot_servo", "assembly_clunk"}


def ndc(p, f):
    """Screen position of world point p at frame f: (x, y) in [-1, 1] (y up) and depth; None behind the camera."""
    c, tg, lens = cam_state(f)
    fwd = (tg - c).normalized()
    right = fwd.cross(Vector((0, 0, 1))).normalized()
    up = right.cross(fwd)
    d = Vector(p) - c
    depth = d.dot(fwd)
    if depth < 0.01:
        return None
    hx = depth * 18.0 / lens
    return (round(d.dot(right) / hx, 2), round(d.dot(up) / (hx * 674.0 / 540.0), 2), round(depth, 2))


def gate_events():
    """Item 14: drop positional SFX whose source is off screen for more than ~1 s around the event."""
    keep, dropped = [], {}
    for e in EVENTS:
        if e["kind"] not in GATED or "pos" not in e or e.get("ungated"):
            keep.append(e)
            continue
        f = e["t"] * 30.0 + 1
        if any(in_view(e["pos"], f + df) for df in range(-30, 31, 3)):
            keep.append(e)
        else:
            dropped[e["kind"]] = dropped.get(e["kind"], 0) + 1
    EVENTS[:] = keep
    print("SFX GATING dropped", dropped, "kept", len(keep))


def subject_cam_check(s, f0, f1, step=2, warn=0.6):
    """Minimum camera distance to the Subject's head per scene (item 2: >= 0.6 m)."""
    worst = {}
    for f in range(f0, f1, step):
        hw = Vector(SJ.head_world(s, f))
        c, _, _ = cam_state(f)
        d = (c - hw).length
        sc = (f - 1) // 300
        if sc not in worst or d < worst[sc][0]:
            worst[sc] = (d, f)
    for sc, (d, f) in sorted(worst.items()):
        print("SUBJECT-CAM min S%d %.2f m at frame %d (t %.2f)%s" % (sc + 1, d, f, (f - 1) / 30.0, "  < %.1f" % warn if d < warn else ""))
    return worst


def main_build():
    bird_mats()
    racks = build_hall()
    tx_sub, rx_sub = os.path.exists(TX_GLB), os.path.exists(RX_GLB)
    build_tx_sub() if tx_sub else build_tx_macro()
    build_rx_sub() if rx_sub else build_rx_macro()
    HATCH_LOC = Vector((0.0, 0.0, -25.0))   # v3: near the origin (float32: 450-600 mm macro shots + overlay text garble at x = 600)
    # v3: M's assembly hatchery (DevLog-019) replaces E's loop for the S5 flashback
    sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "assembly"))
    import assembly_api as AA
    asm_root, asm_objs = AA.append_assembly(loc=tuple(HATCH_LOC))
    n_w = 0
    for o in asm_objs:
        ad = o.animation_data
        own_vis = ad is not None and ad.action is not None and any(fc.data_path in ("hide_render", "hide_viewport") for fc in ad.action.fcurves)
        if not own_vis:
            V(o, 4, 7.7, S5_END)
            n_w += 1
    print("ASSEMBLY objects", len(asm_objs), "windowed", n_w)
    for nm_, off_, en_ in (("asm_fill_key", (0.4, -1.6, 2.4), 260.0), ("asm_fill_side", (-1.8, -1.0, 1.8), 120.0)):   # v3: set read too dark
        lo_ = add_light(nm_, "AREA", tuple(HATCH_LOC + Vector(off_)), en_, (1.0, 0.97, 0.92), size=2.0)
        lo_.rotation_euler = (HATCH_LOC + Vector((0.0, 0.0, 1.0)) - lo_.location).to_track_quat("-Z", "Y").to_euler()
        V(lo_, 4, 7.7, S5_END)
    # E's street: opening for S1 (and the door view at the end of S5), ending for S6
    append_env("street_opening", [(0, 0, 10)])
    append_env("street_ending", [(4, S5_END - 0.25, S5_END), (5, 0, 10)])
    tp = ENV_INST.get("ENV_tree_projector")
    if tp is not None:            # F8: the tree moves to the separate intro; no hologram tree or projector in the street
        L._VIS[tp] = [(TOTAL + 5, TOTAL + 6)]
    hk = env_hooks("street_ending")
    cctv = [Vector(hk["HOOK_cctv_%d_lens" % k]["location"]) for k in range(3)]
    sat0 = Vector(hk["HOOK_satellite"]["location"])
    sat1 = Vector(hk["HOOK_satellite_pass_end"]["location"])
    dish0 = Vector(hk["HOOK_satellite_dish"]["location"])
    flaps_vent = append_flap_vent()
    hp, hy0 = cell_pos(*HERO_CELL)
    ap, ay = cell_pos(*ACROSS_CELL)
    cps = [cell_pos(*c) for c in COMP_CELLS]
    DH = Vector((hp.x + 1.16, 0.0, hp.z - 1.315))
    pad = PADS.get(HERO_CELL[0], Vector((-1.35, 7.0, 2.256)))
    out = Vector((-0.05, 7.0, 1.28)) + DH      # v4e: exploded view in the aisle centre, away from the rack faces
    exp_c = out + Vector((0.0, 0, 0.13))
    hero_head_w = bwm(hp, 0.0, HC)
    print("HERO CELL", tuple(round(v, 3) for v in hp), "pad", tuple(round(v, 3) for v in pad), "head", tuple(round(v, 3) for v in hero_head_w))
    HALL_F = [(F(0, 5.0), F(5, 4.0))]

    # ---------------- R's real hero cell + the L07 copy (cutaway dissolve instead of hiding the rack)
    hc = append_hero_cell(hp, hy0)
    for o in hc["objs"]:
        vis(o, HALL_VIS)
    for nm, o in HERO_RACK.items():
        if "slot_t0_c1" in nm and "chassis_u1" in nm:
            L._VIS[o] = [(TOTAL + 5, TOTAL + 6)]       # the hero slot itself comes from R's mating cell
        else:
            vis(o, HALL_VIS)
    cut_cam = hp + Vector((0.16, -0.48, 0.19))      # v4e: closer, centred (L06 front bundles removed; modules stay)
    cut_tgt = hp + Vector((0.035, 0.0, 0.095))
    grid = [hp + Vector((0.035 + u, 0.0, 0.095 + v)) for u in (-0.27, -0.13, 0.0, 0.13, 0.27) for v in (-0.3, -0.15, 0.0, 0.15, 0.3)]
    bpy.context.view_layer.update()
    occ = occluders([o for nm, o in HERO_RACK.items() if not ("slot_t0_c1" in nm and "chassis_u1" in nm)] + [hc["cutaway"]],
                    cut_cam, grid)
    W_CUT = (F(1, 7.2), F(2, 0.85))
    dissolve(occ + L06_BUNDLES, [(W_CUT[0], 1.0), (F(1, 7.55), 0.0), (F(2, 0.1), 0.0), (F(2, 0.75), 1.0)], W_CUT, HALL_F)
    print("CUTAWAY L06 bundles dissolved", len(L06_BUNDLES))
    # v4h (owner): no DOF / dimming in the S2 cutaway
    print("CUTAWAY dissolve", len(occ), sorted(o.name for o in occ)[:40])
    lip = hc["lip"]
    lip.rotation_mode = "XYZ"
    lip0 = lip.rotation_euler.copy()

    def lip_key(idx, t, up):
        K(lip, F(idx, t), rot=(lip0.x, lip0.y - 1.2 * up, lip0.z), interp="BEZIER")
    for idx, t, up in ((0, 0.0, 0.0), (0, 7.6, 0.0), (0, 7.9, 1.0), (0, 8.25, 1.0), (0, 8.45, 0.0), (1, 3.15, 0.0), (1, 3.4, 1.0),
                       (1, 9.35, 1.0), (1, 9.55, 0.0), (4, 0.5, 0.0), (4, 0.75, 1.0)):
        lip_key(idx, t, up)

    # ---------------- Subject (C: Gary v2, exact outfit) - S1 walk-in to the hero cell
    s = SJ.append_subject()
    MARK = Vector((hp.x + 0.8, hp.y - 0.12, 0.0))      # v3: further into the aisle (owner: off the shelf pigeons)
    SJ.place(s, F(0, 0.0), (1.05, 17.0, 0.0), 0.0)
    SJ.idle(s, F(0, 0.0), F(6, 0.0))
    f_arrive = SJ.walk_path(s, F(0, 5.0), [(1.05, 17.0), (1.15, 16.2), (0.55, 13.9), (0.1, 12.4), (0.1, 8.5), (0.0, 7.4),
                                          (MARK.x, MARK.y)], gait="walk_brisk", speed=1.55,
                            end_yaw=SJ.yaw_face((MARK.x, MARK.y), (hp.x, hp.y)))
    print("SUBJECT arrives", f_arrive)
    SJ.cup_show(s, F(1, 0.0), False)

    # ---------------- hero (H's GLB on C's rig)
    hero = make_hero_c()
    PR.rest_pose(hero.r, 1)
    if hero.m_neck is not None:      # item 6: brighter neck accent on the hero (H: rim_strength, default 0.8)
        rn = hero.m_neck.node_tree.nodes.get("rim_strength")
        if rn is not None:
            rn.inputs[1].default_value = 1.8
    if hero.m_band is not None:
        b_ = principled(hero.m_band)
        if b_ is not None:
            b_.inputs["Emission Strength"].default_value = 3.0
    # S5 call-outs (labels only; parody names from DevLog-015 section 3, roles from research/*.md)
    for sub, txt, ta, sd in (("dsp_die", "DSP: NARROWCOM / MERVALL", 1.9, "R"), ("tx_pic_siph_die", "SIPH PIC: TOWERING SEMI", 2.9, "R"),
                             ("tx_beam_cores_laser", "DFB X2: DIMENTUM", 3.1, "R"), ("rx_pin_pd_array_1x8", "PD: MACOMBO", 4.2, "L"),
                             ("rx_octal_tia", "TIA: SEMTREK", 5.4, "L"), ("flex_tx_host_lanes_neck_leg", "FLEX RIBBONS", 3.4, "L"),
                             ("foot_tx_edge_pads_x26", "FEET = 60 PADS", 3.0, "R"), ("pcb_sternum_board", "ASSEMBLY: FABRINOT", 3.6, "L")):
        o_ = hero.by(sub)        # v4e (owner): callouts are drawn in post (hero-loop style); here only the anchors
        if o_ is None:
            print("CALLOUT SKIPPED (part not found)", sub)
            continue
        CALLOUTS.append(dict(text=txt, obj=o_, f0=F(4, ta), f1=F(4, 7.3), scene=4, side=sd))
    ghost_m = pmat((0.12, 0.14, 0.18), 0.4, 0.2)
    set_blend(ghost_m)
    for tt, a in ((1.7, 1.0), (2.3, 0.06), (7.3, 0.06), (7.75, 1.0)):
        key_alpha(ghost_m, F(4, tt), a)
    ghosts = {}
    for o in list(hero.top):
        if not o.name.startswith("skin_") or o.type != "MESH":
            continue
        if "seams" in o.name or "screws" in o.name:
            for i in range(7):
                if i == 4:
                    V(o, 4, 0, 1.7)
                    V(o, 4, 8.1, S5_END)
                else:
                    V(o, i, 0, (S5_END if i == 4 else 10) if i < 6 else S7_LEN)
            continue
        g = o.copy()
        bpy.context.scene.collection.objects.link(g)
        for slot in g.material_slots:
            slot.link = "OBJECT"
            slot.material = ghost_m
        V(g, 4, 1.7, 8.1)
        ghosts[g] = o.name
        for i in range(7):
            if i == 4:
                V(o, 4, 0, 1.7)
                V(o, 4, 8.1, S5_END)
            else:
                V(o, i, 0, (S5_END if i == 4 else 10) if i < 6 else S7_LEN)

    # ---------------- companions, rack crowd (C head rigs), street birds (C full rigs)
    lib = PR.proxy_library()
    hero_mats = {sl.material for o in hero.glb for sl in getattr(o, "material_slots", []) if sl.material}
    n_dark = 0
    for me in lib.values():          # item 6: darker crowd, no hero signature on the proxies
        for m in getattr(me, "materials", []):
            if m is None or m in hero_mats or m.get("crowd_dark"):
                continue
            m["crowd_dark"] = 1
            b_ = principled(m)
            if b_ is None:
                continue
            nm = m.name.lower()
            if any(k in nm for k in ("eye", "lens", "iris")):
                continue
            col = b_.inputs["Base Color"].default_value
            b_.inputs["Base Color"].default_value = (col[0] * 0.55, col[1] * 0.55, col[2] * 0.55, col[3])
            if any(k in nm for k in ("band", "neck_rim", "rim_accent")):
                b_.inputs["Emission Strength"].default_value = 0.0
                rn = m.node_tree.nodes.get("rim_strength")
                if rn is not None:
                    rn.inputs[1].default_value = 0.0
            n_dark += 1
    print("CROWD materials darkened", n_dark)
    comps = []
    for k, (p_, y_) in enumerate(cps):
        r = PR.make_proxy_rig("COMP%d" % k, lib, tuple(p_), y_, mode="full")
        PR.rest_pose(r, 1)
        comps.append(RigBird(r))
        bird_at(comps[-1], 0, 0, p_, y_, interp="CONSTANT")

    def blocks_cutaway(pos):
        c = Vector(pos) + Vector((0.0, 0.0, 0.12))
        for t in grid:
            d = Vector(t) - cut_cam
            u = (c - cut_cam).dot(d) / d.length_squared
            if 0.0 < u < 0.9 and (cut_cam + d * u - c).length < 0.12:
                return True
        return False
    crowd, specs, removed = {}, [], []
    bird_racks = [rid for rid, r_ in ((r_["id"], r_) for r_ in LAYOUT["racks"]) if r_["type"] == "bird_q3400"]
    floor_objs = [o for rid in bird_racks for o in racks.get(rid, []) if o.type in ("MESH", "EMPTY")] + \
        [o for o in hc["objs"] if o.type == "MESH"]
    bpy.context.view_layer.update()
    floor_bvh = PR.surfaces_bvh(floor_objs)      # v3 (C): toes rest on the real slot floors
    feet_stats = []
    for cell in sorted(CELLS):
        if cell in (HERO_CELL, *COMP_CELLS):
            continue
        pos, yaw = cell_pos(*cell)
        if cell[0] == HERO_CELL[0] and blocks_cutaway(pos):
            removed.append(cell[1])      # empty cell: the bird would sit between the cutaway camera and the hero
            continue
        ext = NECK_EXT.get(cell, 0.0) * NECK_EXT_SCALE
        bid = cell[0] + "_" + cell[1]
        r = PR.make_proxy_rig(bid, lib, tuple(pos), yaw, mode="head", neck_ext=ext)
        feet_stats.append(PR.place_on_floor(r, 1, bvh=floor_bvh))
        PR.rest_pose(r, 1)
        crowd[bid] = r
        for o in r.objects:
            vis(o, HALL_VIS)
        bl = [(F(2, 7.9), F(3, 2.3))] if cell == ACROSS_CELL else []
        specs.append(BH.spec_from_rig(bid, r, group="rack", blocked=bl))
    print("CROWD removed for the cutaway", removed)
    if feet_stats:
        sup = [d.get("support", 0) for d in feet_stats if isinstance(d, dict)]
        print("CROWD feet: birds %d, min support %.2f, max shift %.3f m" % (len(sup), min(sup) if sup else -1,
              max((d.get("shift", 0) for d in feet_stats if isinstance(d, dict)), default=0)))
    across = RigBird(crowd[ACROSS_CELL[0] + "_" + ACROSS_CELL[1]])
    for k, rb in enumerate(comps):
        specs.append(BH.BirdSpec("COMP%d" % k, tuple(cps[k][0]), cps[k][1], group="rack", blocked=[(F(4, S5_END - 0.1), F(6, 0))]))
    street, sspecs = {}, []
    SP_EXTRA = [("R3_parapet", Vector(hk["HOOK_perch_x_R3_parapet"]["location"]), PI)]
    sperches = street_perches() + SP_EXTRA
    st_surf = [o for k_, o in ENV_INST.items() if k_.endswith("@end") and "satellite" not in k_ and "sky" not in k_.lower()]
    bpy.context.view_layer.update()
    st_bvh = PR.surfaces_bvh(st_surf)          # v4 (C): street birds rest on their posts / ledges / wires
    n_pl = 0
    for name, pos, yaw in sperches:
        r = PR.make_proxy_rig("ST_" + name, lib, tuple(pos), yaw, mode="full")
        try:
            PR.plant_on_surface(r, 1, bvh=st_bvh)
            n_pl += 1
        except Exception as ex:
            print("PLANT failed", name, ex)
        PR.rest_pose(r, 1)
        street[name] = r
        for o in r.objects:
            vis(o, STREET_VIS)
    pairs = [("rail_a", "bin"), ("ledge_a", "ledge_b"), ("lamp1_top", "lamp1_arm")]
    for name, pos, yaw in sperches:
        bl = [(F(5, 8.95), F(6, 0))]
        if name in ("R3_parapet", "roof_L", "ledge_b", "wire_20.7"):
            bl.append((F(5, 4.7), F(5, 6.9)))
        if name in ("rail_a", "rail_b"):
            bl.append((F(5, 6.8), F(5, 9.0)))
        sspecs.append(BH.spec_from_rig("ST_" + name, street[name], kind="street", group="street", blocked=bl, can_turn=False))
    forced = [(F(5, 2.7), "ST_" + a, "ST_" + b, 66) for a, b in pairs]

    # ---------------- Subject S2-S6 (after the hero's key frames that hold_bird needs are known)
    hy_face = SJ.yaw_face((MARK.x, MARK.y), (hp.x, hp.y))
    f_hold0 = F(1, 3.0)       # VO rev 4: beak bite 12.9 s, then the hop
    f_unpres = F(1, 6.95)     # VO rev 3: 'Ground' 17.84 'lands' 18.18 'first' 18.50 -> back on the pads ~18.4
    f_hold1 = f_unpres + 55
    present_dir = Vector((-0.15, -1.0, 0.45)).normalized()   # toward the -y camera side (Gary stands at +x, beside it)
    hb = SJ.hold_bird(s, f_hold0, f_hold1, tuple(hp), hy0, rig=hero.r, present_dir=tuple(present_dir), end="cell")
    HF = hb["frames"]
    print("HOLD_BIRD frames", HF)
    SPOT = Vector((0.35, 8.1, 0.0))
    f_w2 = SJ.walk_path(s, HF["end"] + 2, [(MARK.x, MARK.y), (-0.15, 7.45), (SPOT.x, SPOT.y)], gait="walk", speed=1.0,
                        end_yaw=SJ.yaw_face((SPOT.x, SPOT.y), (hp.x, hp.y)))
    SJ.cup_show(s, f_w2, True)
    SJ.sip_coffee(s, F(3, 3.0))
    print("SUBJECT at the S3/S4 spot", f_w2)
    # S5: leave through the door before 'They follow you outside' (S6 0.25)
    f5 = SJ.walk_path(s, F(4, 0.2), [(SPOT.x, SPOT.y), (0.45, 9.2), (0.3, 11.2), (0.12, 12.6), (0.05, 13.6), (-0.3, 15.2)],
                      gait="walk", speed=1.3)
    S6_MARK = Vector((-1.1, 19.3, 0.0))
    f6 = SJ.stroll_unaware(s, max(F(5, 0.0), f5 + 2), [(-0.3, 15.2), (-0.7, 17.5), (S6_MARK.x, S6_MARK.y), (-1.0, 22.0), (-0.8, 24.2)],
                           phone=True, sips=1, speed=0.9)      # v3 (owner): Gary is unaware and goes about his day
    print("SUBJECT S5 walk end", f5, "S6 mark", f6)
    track_hall = SJ.head_track(s, F(0, 6.0), F(5, 3.0), step=6)
    track_street = SJ.head_track(s, F(5, 0.0), F(6, 0.0), step=6) + [(F(0, 0.0) + k * 15, SJ.head_world(s, F(0, 0.0) + k * 15))
                                                                    for k in range(21)]
    track_street.sort(key=lambda x: x[0])
    sched_hall = BH.schedule(specs, F(0, 0.0), F(5, 4.0), seed=7, subject=track_hall,
                             force_observe=[(F(0, 8.0), F(1, 2.0)), (F(4, 0.5), F(4, 4.5))])
    sched_street = BH.schedule(sspecs, F(0, 0.0), F(6, 0.0), seed=11, subject=track_street,
                               force_observe=[(F(5, 2.4), F(5, 4.5)), (F(5, 7.0), F(5, 9.0))], forced_links=forced)
    rigs = dict(crowd)
    rigs.update({"COMP%d" % k: rb.r for k, rb in enumerate(comps)})
    BH.apply(sched_hall, rigs, blinks=True)
    BH.apply(sched_street, {"ST_" + n: r for n, r in street.items()}, blinks=True)
    print("SCHED hall", sched_hall.stats(), "street", sched_street.stats())

    # hero signature light in the rack shots (item 6): a soft spot on the hero's cell
    sp = add_light("hero_spot", "SPOT", tuple(hp + Vector((0.9, 0.6, 0.9))), 140.0, (1.0, 0.93, 0.85))
    sp.data.spot_size = math.radians(16)
    sp.data.spot_blend = 0.6
    sp.data.shadow_soft_size = 0.05
    d_ = (hero_head_w - sp.location).normalized()
    sp.rotation_euler = d_.to_track_quat("-Z", "Y").to_euler()
    vis(sp, [(0, 7.5, 10), (1, 0, 10), (2, 0, 2.0), (2, 8.3, 10), (3, 0, 2.0), (3, 6.9, 10), (4, 0, 2.4)])
    pad_glow = box("pad_charge_glow", (pad.x + 0.12, pad.y, pad.z + 0.004), (0.02, 0.5, 0.004), C["cyan"])
    pad_m = mat(C["cyan"], 0.3, 1.0, unique=True)
    pad_glow.data.materials[0] = pad_m
    vis(pad_glow, HALL_VIS)
    for tt, st in ((7.0, 1.0), (7.15, 9.0), (7.45, 6.0), (7.8, 1.0)):
        key_glow(pad_m, F(0, tt), C["cyan"], st)

    # ---------------- S1: intro (F8, separate; black placeholder until the intro agent delivers), street, flap, pad, cell
    i = 0
    fill(i, (0.0, 0.0, 0.0), [(0.0, 1.0), (3.8, 1.0), (4.4, 0.0)])   # the intro pass (F8) covers 0-4.5 s, cross-dissolve 4.5-5.3
    H0 = Vector((0.05, 17.62, 2.5))
    cam(i, 0.0, (0.12, 18.47, 2.56), H0 + Vector((0.06, 0, 0.12)), 32)
    cam(i, 4.1, (0.12, 18.47, 2.56), H0 + Vector((0.06, 0, 0.12)), 32)
    cam(i, 5.2, (0.1, 18.3, 2.62), H0 + Vector((0.03, 0, 0.14)), 32)
    cam(i, 5.65, (0.05, 17.4, 2.95), (0.0, 15.2, 3.05), 28)
    cam(i, 6.05, (0.0, 14.7, 3.22), (0.0, 13.0, 3.25), 26)
    cam(i, 6.5, (0.0, 13.0, FLAP.z - 0.02), (-0.25, 11.0, 3.0), 26)
    cam(i, 7.0, (-0.25, 10.6, 2.98), (-0.95, 8.0, 2.5), 26)
    cam(i, 7.4, (-0.45, 8.75, 2.75), pad + Vector((0.05, 0, 0.12)), 28)
    cam(i, 7.85, (-0.45, 7.65, 2.15), (-0.9, 7.0, 1.75), 28)
    cam(i, 8.25, (-0.55, 7.05, 1.95), hero_head_w + Vector((0.0, 0.0, -0.04)), 30)
    cam(i, 8.65, (-0.78, 6.55, 1.86), hero_head_w + Vector((0.0, 0.0, -0.04)), 30)     # 'Fancy name for a pigeon': clear hero
    cam(i, 9.4, (-0.82, 6.47, 1.86), hero_head_w + Vector((0.0, 0.02, -0.04)), 30)
    s1_end = (Vector((-0.92, 6.2, 1.86)), Vector((-0.9, 7.3, 1.72)), 28)
    cam(i, 10.0, *s1_end)
    for t, pos, yaw, pitch in ((0.0, (0.9, 17.3, 2.3), PI / 2, 0.0), (3.6, (0.9, 17.3, 2.3), PI / 2, 0.0),
                               (4.1, (0.3, 17.5, 2.42), PI / 2, 0.0), (4.45, tuple(H0), PI / 2, 0.0), (5.05, tuple(H0 + Vector((0, 0.03, 0.04))), PI / 2, 0.0),
                               (5.3, (0.0, 16.6, 2.75), -PI / 2, 0.05), (5.7, (0.0, 15.0, 3.1), -PI / 2, 0.05),
                               (6.15, (0.0, 13.2, 3.25), -PI / 2, 0.0), (6.35, (0.0, 12.4, 3.2), -PI / 2, 0.1),
                               (6.75, (-0.6, 10.0, 2.85), -PI / 2, 0.15), (7.0, tuple(pad + Vector((0.05, 0, 0.2))), -PI / 2, 0.0),
                               (7.15, tuple(pad), -PI / 2, 0.0), (7.4, tuple(pad), 0.0, 0.0),
                               (7.85, (-0.82, 7.0, hp.z + 0.06), 0.0, 0.0), (8.2, tuple(hp), 0.0, 0.0)):
        bird_at(hero, i, t, pos, yaw, pitch)
    PR.flap_cycle(hero.r, F(0, 3.5), F(0, 6.95))
    PR.wings_fold(hero.r, F(0, 7.2))
    PR.flap_cycle(hero.r, F(0, 7.45), F(0, 7.85), hz=5.0)
    PR.wings_fold(hero.r, F(0, 8.05))
    head(hero, i, 0.0, 0.0, interp="CONSTANT")
    head(hero, i, 4.55, 0.25)
    head(hero, i, 4.8, -0.25)
    head(hero, i, 5.05, 0.0)
    head(hero, i, 7.2, 0.0)
    head(hero, i, 7.3, 0.8)
    head(hero, i, 7.45, 0.0)
    beak(hero, i, 8.3, 0.0); beak(hero, i, 8.38, 1.0); beak(hero, i, 8.5, 0.0)
    PR.head_track(hero.r, F(0, 8.5), F(1, 1.5), lambda f: SJ.head_world(s, f), step=(8, 12), seed=3)
    PR.head_snap(hero.r, F(1, 1.55), 0.0, 0.0, 0.0)
    for piv in flaps_vent:
        r0 = piv.rotation_euler.copy()
        for tt, a in ((5.8, 0.0), (6.08, 1.35), (6.65, 1.35), (7.2, 0.0)):
            K(piv, F(i, tt), rot=(r0.x, r0.y + a, r0.z), interp="BEZIER")
    lab(i, 6.7, 7.45, "LANDING PAD: CHARGE + SYNC")
    lab(i, 7.45, 8.3, "800G OSFP DR8\nDR8: DRONE RECON, 8 LANES")
    lab(i, 8.3, 10.0, "HEADS TRACK THE SUBJECT")

    # ---------------- S2: wings, beak latch on R's lip, onto Gary's hand, TX/RX sides, belly, back in, mating cutaway
    i = 1
    lip_w = hp + Vector((0.158, 0.0, 0.191))
    cam(i, 0.0, *s1_end)
    cam(i, 1.5, (-0.9, 6.3, 1.8), hp + Vector((0.0, -0.02, 0.14)), 32)
    cam(i, 2.35, (-0.93, 6.6, 1.6), lip_w + Vector((-0.03, 0, 0.0)), 38)
    cam(i, 3.05, (-0.93, 6.62, 1.6), lip_w + Vector((-0.02, 0, 0.0)), 38)
    M_on = hb["matrices"][HF["raised"]]
    M_pr = hb["matrices"][HF["present"] + 4]
    bird_on = M_on.to_translation()
    belly = M_pr @ Vector((0.0, 0.0, 0.02))
    rgt = (M_pr.to_3x3() @ Vector((0.0, -1.0, 0.0))).normalized()      # bird's right side (TX foot / eye)
    cam(i, 3.7, (-0.9, 6.35, 1.55), (bird_on + hp) / 2.0 + Vector((0.05, 0, 0.05)), 30)
    def clampx(v):     # keep the hand-held views in the aisle (rack faces at x ~ -1.1)
        return Vector((max(v.x, -0.93), v.y, v.z))
    cam_pr = clampx(belly + present_dir * 0.52)
    c_r = clampx(belly + (present_dir + rgt * 0.4).normalized() * 0.5)
    c_l = clampx(belly + (present_dir - rgt * 0.4).normalized() * 0.5)
    print("S2 HAND belly", tuple(round(v, 3) for v in belly), "bird_on", tuple(round(v, 3) for v in bird_on), "cams",
          [tuple(round(v, 2) for v in c) for c in (cam_pr, c_r, c_l)], "rgt", tuple(round(v, 2) for v in rgt))
    cam(i, (HF["present"] - 301) / 30.0, cam_pr, belly, 34)
    cam(i, 6.0, c_r, belly, 38)
    cam(i, 6.95, c_l, belly, 38)
    cam(i, 7.6, (-0.9, 6.4, 1.58), hp + Vector((0.03, 0, 0.08)), 35)
    cam(i, 8.15, cut_cam, cut_tgt, 35)
    cam(i, 9.9, cut_cam + Vector((-0.004, 0.06, -0.01)), cut_tgt, 35)
    PR.wings(hero.r, F(1, 0.3), 0.0)
    PR.head_look_at(hero.r, HF["raised"], Vector(SJ.head_world(s, HF["raised"])), dur=3)
    PR.head_snap(hero.r, HF["flip0"] - 2, 0.0, 0.0, 0.0)
    PR.head_snap(hero.r, HF["in"] + 4, 0.0, 0.0, 0.0)
    beak(hero, i, 2.45, 0.0); beak(hero, i, 2.6, 1.0); beak(hero, i, 2.9, 0.0); beak(hero, i, 3.05, 0.0); beak(hero, i, 3.15, 1.0)
    beak(hero, i, 3.5, 1.0); beak(hero, i, 3.7, 0.0)
    beak(hero, i, 9.55, 0.0); beak(hero, i, 9.62, 1.0); beak(hero, i, 9.72, 0.0)
    for m in {hero.eye_tx, hero.eye_rx}:   # unplugged = no power = dark lenslets (rims pulse as pointers)
        b_ = principled(m)
        if b_ is None:
            continue
        st0 = b_.inputs["Emission Strength"].default_value or 4.0
        for f_, st, it in ((HF["hop0"], st0, "LINEAR"), (HF["hop0"] + 6, 0.0, "CONSTANT"), (F(1, 9.7), 0.0, "LINEAR"), (F(1, 9.8), st0, "CONSTANT")):
            key_strength(m, f_, st, it)
    rimR, rimL = hero.m_rim.get("tx"), hero.m_rim.get("rx")
    for m, keys in ((rimR, [(HF["hop0"], 3.0), (HF["hop0"] + 6, 0.0), (F(1, 4.55), 0.0), (F(1, 4.65), 9.0), (F(1, 5.0), 4.0),
                            (F(1, 6.0), 4.0), (F(1, 6.2), 0.0), (F(1, 9.7), 0.0), (F(1, 9.8), 3.0)]),
                    (rimL, [(HF["hop0"], 3.0), (HF["hop0"] + 6, 0.0), (F(1, 6.2), 0.0), (F(1, 6.3), 9.0), (F(1, 6.65), 4.0),
                            (F(1, 7.6), 4.0), (F(1, 7.8), 0.0), (F(1, 9.7), 0.0), (F(1, 9.8), 3.0)])):
        if m is not None:
            pulse(m, keys)
    t_g, t_p, t_s = (HF["in"] - 301) / 30.0 + 0.03, (HF["in"] - 301) / 30.0 + 0.28, (HF["in"] - 301) / 30.0 + 0.53
    padR, padL = hero.m_pads.get("tx"), hero.m_pads.get("rx")
    if padR is not None:
        pulse(padR, [(F(1, 4.55), 0.0), (F(1, 4.65), 9.0), (F(1, 5.0), 3.0), (F(1, 6.0), 3.0), (F(1, 6.2), 0.0),
                     (F(1, t_s) - 1, 0.0), (F(1, t_s), 9.0), (F(1, t_s + 0.45), 1.0)], color=C["red"])
    if padL is not None:
        pulse(padL, [(F(1, 6.2), 0.0), (F(1, 6.3), 9.0), (F(1, 6.65), 3.0), (F(1, 7.6), 3.0), (F(1, 7.8), 0.0),
                     (F(1, t_s) - 1, 0.0), (F(1, t_s), 9.0), (F(1, t_s + 0.45), 1.0)], color=C["cyan"])
    for cls, tt, col in (("gnd", t_g, C["brass"]), ("pwr", t_p, C["orange"])):
        for m in hero.contact[cls]:
            key_glow(m, F(i, tt) - 1, col, 0.0)
            key_glow(m, F(i, tt), col, 14.0)
            key_glow(m, F(i, tt + 0.45), col, 0.8, interp="CONSTANT")
    for k, tt, col in (("GND", t_g, C["brass"]), ("VCC", t_p, C["orange"]), ("LS", t_p, (0.3, 1.0, 0.4)), ("TX", t_s, C["red"]),
                       ("RX", t_s, C["cyan"])):
        m = hc["pads"].get(k)
        if m is not None:
            pulse(m, [(F(i, tt) - 1, 0.0), (F(i, tt), 8.0), (F(i, tt + 0.45), 1.2)], color=col)
    # contact labels with leader lines (item 11), placed for the cutaway camera (screen right = +x)
    for txt, sub, tt, lp in (("1 GND: TALON TIPS (LONGEST PADS)", "foot_tx_talon_tip_gnd", t_g, (0.1287, -0.1405, 0.005)),   # v4j2: inside frame (40 px)
                             ("2 POWER: BELLY PADS", "pcb_belly_lowspeed_vcc_pads_x8", t_p, (-0.0708, -0.0828, 0.1)),
                             ("3 SIGNAL: TOE PADS (TX / RX)", "foot_tx_edge_pads_x26", t_s, (0.1429, -0.1364, 0.06))):
        o = hero.by(sub)     # v4j: original in-place contact labels restored (owner)
        if o is None:
            print("CONTACT LABEL SKIPPED", sub)
            continue
        tgt_w = bwm(hp, 0.0, hero.cen[o.name])
        anchor = empty("contact_anchor", tuple(tgt_w))
        tip = empty("contact_tip", tuple(hp + Vector(lp)))
        t_obj = wlab(txt, tuple(hp + Vector(lp) + Vector((0.0, 0, 0.012))), i, tt, 10.0, size=0.0115,
                     color=(1.0, 0.85, 0.3))
        ld = stretch_strap("contact_leader", tip, anchor, 0.0008, 0.0008, mat((1.0, 0.85, 0.3), 0.5, 2.0))
        V(ld, i, tt, 10.0)
        dot = sph("contact_dot", tuple(tgt_w), 0.0022, (1.0, 0.85, 0.3), emit=4.0, seg=8, rings=6)
        V(dot, i, tt, 10.0)
        V(t_obj, 2, 0, 0.3)
        V(ld, 2, 0, 0.3)
        V(dot, 2, 0, 0.3)
    lab(i, 0.0, 1.9, "WING FINS = INTEGRATED HEAT SINK (IHS)")
    lab(i, 1.9, 3.6, "BEAK = LATCH ON THE CAGE LIP\nTAIL TAB = PULL TAB")
    lab(i, 4.5, 6.2, "RIGHT FOOT + RIGHT EYE = TX")
    lab(i, 6.2, 7.6, "LEFT FOOT + LEFT EYE = RX")
    lab(i, 7.9, 10.0, "CUTAWAY: GROUND -> POWER -> SIGNAL")

    # ---------------- S3: continuous pull-out of the cutaway, TX eye, macro, zoom back out to the hero
    i = 2
    a_spec = next(b for b in specs if b.id == across.r.name)
    PR.head_look_at(hero.r, F(2, 0.05), a_spec.eye_world("rx"), eye="tx", dur=20)
    e_tx = eyew(hero, "tx", F(2, 1.2))
    PR.head_look_at(across.r, F(2, 7.9), e_tx, eye="rx", dur=10)
    e_arx = eyew(across, "rx", F(2, 9.0))
    ax_t = (e_arx - e_tx).normalized()
    side_t = ax_t.cross(Vector((0, 0, 1))).normalized()
    lo = hero.by("tx_eye_lenslets_out_2x4")
    if lo is not None:
        tw = lo.copy()
        for c_ in lo.users_collection:
            c_.objects.link(tw)
            break
        VX.apply_data_look([tw], VX.data_emitter_material("VFX_data_tx_eye", VX.TX_RED, strength=7.0, rate=20, pitch=0.0023))
        L._VIS[lo] = [(1, F(2, 0.0)), (F(3, 1.4), TOTAL + 1)]
        L._VIS[tw] = [(F(2, 0.0), F(3, 1.4))]
    cam(i, 0.0, cut_cam + Vector((-0.004, 0.06, -0.01)), cut_tgt, 35)
    cam(i, 0.35, hp + Vector((0.32, -0.42, 0.24)), hp + Vector((0.1, 0.0, 0.18)), 32)
    cam(i, 0.7, e_tx + ax_t * 0.42 + Vector((0, 0, 0.01)), e_tx, 34)
    cam(i, 0.95, e_tx + ax_t * 0.025, e_tx - ax_t * 0.1, 40)
    fill(i, C["red"], [(0.6, 0.0), (0.92, 1.0), (1.02, 1.0), (1.4, 0.0)])
    X = TX_X
    if tx_sub:
        tx_keys = [(1.0, (1.5, -0.1, -1.4), (-1.75, -0.55, -2.5)), (2.0, (1.4, -0.2, -1.45), (-1.75, -0.6, -2.5)), (3.0, (1.2, -2.6, -1.0), (-1.7, -2.0, -1.8)),
                   (3.8, (1.3, -3.8, 1.8), (-1.7, -5.0, 0.3)), (5.0, (1.0, -7.2, 1.4), (-1.7, -7.6, 0.0)), (6.0, (1.4, -5.0, 2.4), (-1.7, -3.0, 0.5)),
                   (6.8, (1.0, -3.6, 1.9), (-1.6, -1.2, 0.0)), (7.4, (1.8, -3.2, 1.5), (-0.6, 0.0, 0.0)), (7.8, (3.2, -2.4, 1.2), (0, 0, 0)),
                   (8.25, (3.4, -2.25, 1.22), (0, 0, 0))]
        tx_keys = [(t, W_tx(p), W_tx(q)) for t, p, q in tx_keys]
    else:
        tx_keys = [(2.0, (X + 15.0, -0.4, 1.9), (X + 12.4, 0, 0.4)), (8.25, (X + 15.0, -3.0, 1.35), (X + 12.5, 0, 0.4))]
    if TXFX.get("fx"):
        TXFX["mod"].tx_light_up(TXFX["fx"], F(2, 1.05), F(2, 7.4))       # DFB on at 'lasers' (21.01 s)
        TXFX["mod"].tx_signal_glow(TXFX["fx"], F(2, 4.95), F(2, 8.3))    # RF in under 'Eight Mach-Zehnder modulators'
    cam_jump(i, 1.0, (e_tx + ax_t * 0.025, e_tx - ax_t * 0.1, 40), (tx_keys[0][1], tx_keys[0][2], 26))
    for t, p, q in tx_keys[1:]:
        cam(i, t, p, q, 26)
    fill(i, C["red"], [(7.98, 0.0), (8.25, 1.0), (8.35, 1.0), (8.8, 0.0)])
    off1 = side_t * 0.03 + Vector((0, 0, 0.03))
    off2 = side_t * 0.1 + Vector((0, 0, 0.07))
    cam_jump(i, 8.3, (tx_keys[-1][1], tx_keys[-1][2], 26), (e_tx + ax_t * 0.06 + off1, e_tx, 34))
    cam(i, 10.0, e_tx + ax_t * 0.5 + off2, e_tx + ax_t * 0.05, 30)
    bundle(hero, "tx", F(2, 9.0), e_arx, 2, 8.3, 10.0, "link_beam_tx")
    lab(i, 0.0, 0.9, "ONE-EYED STARE: RIGHT EYE (TX)")
    lab(i, 1.0, 8.3, "TX SUBASSEMBLY (x1000)" if tx_sub else "TX SILICON PHOTONICS CHIP")
    lab(i, 8.3, 10.0, "8 FREE-SPACE BEAMS TO THE BIRD ACROSS THE AISLE")
    note(i, 1.3, 8.2)
    note(i, 8.45, 10.0)

    # ---------------- S4: head turn with a ~70 deg orbit that reveals the sender, RX macro (PD array), POV
    i = 3
    PR.head_look_at(hero.r, F(3, 0.05), a_spec.eye_world("tx"), eye="rx", dur=24)
    e_rx = eyew(hero, "rx", F(3, 1.3))
    PR.head_look_at(across.r, F(3, 0.2), e_rx, eye="tx", dur=10)
    e_atx = eyew(across, "tx", F(3, 1.3))
    ax_r = (e_atx - e_rx).normalized()
    side_r = ax_r.cross(Vector((0, 0, 1))).normalized()
    off_dir = (ax_r * math.cos(math.radians(32)) + side_r * math.sin(math.radians(32))).normalized()
    c0 = e_tx + ax_t * 0.5 + off2
    hcw = hero_head_w
    a0 = math.atan2(c0.y - hcw.y, c0.x - hcw.x)
    r0 = (Vector((c0.x, c0.y, 0)) - Vector((hcw.x, hcw.y, 0))).length
    # v3 (owner): moving into the RX eye mirrors the TX exit in reverse (TX exit: e_tx + ax*0.06 + off1 -> e_tx + ax*0.5 + off2)
    off2r = -side_r * 0.1 + Vector((0, 0, 0.07))
    off1r = -side_r * 0.03 + Vector((0, 0, 0.03))
    rx_out = e_rx + ax_r * 0.5 + off2r
    ar = math.atan2(rx_out.y - hcw.y, rx_out.x - hcw.x)
    rr1 = (Vector((rx_out.x, rx_out.y, 0)) - Vector((hcw.x, hcw.y, 0))).length
    print("S4 HEAD-TURN ARC a0 %.0f -> %.0f deg" % (math.degrees(a0), math.degrees(ar)))
    cam(i, 0.0, c0, e_tx + ax_t * 0.05, 30)
    for k, tt in enumerate((0.35, 0.7)):
        u = (k + 1) / 3.0
        a = a0 + (ar - a0) * u
        rr = r0 + (max(r0, rr1) + 0.12 - r0) * math.sin(math.pi * u) + (rr1 - r0) * u
        p = hcw + Vector((math.cos(a) * rr, math.sin(a) * rr, 0.07))
        cam(i, tt, p, hcw.lerp(e_rx + ax_r * 0.05, u), 30)
    cam(i, 1.0, rx_out, e_rx + ax_r * 0.05, 30)
    cam(i, 1.45, e_rx + ax_r * 0.06 + off1r, e_rx, 34)
    cam(i, 1.9, e_rx + ax_r * 0.025, e_rx - ax_r * 0.1, 40)
    bundle(across, "tx", F(3, 1.3), e_rx, 3, 0.9, 1.45, "link_beam_rx")    # off before the camera enters the eye
    fill(i, C["cyan"], [(1.4, 0.0), (1.9, 1.0), (2.0, 1.0), (2.5, 0.0)])
    X = RX_X
    if rx_sub:   # item 12: on the PD array during "eight photodiodes" (S4 3.0-4.2); TIA held to 6.5
        rx_keys = [(2.0, (2.2, -2.6, 1.8), (-0.6, 0, 0)), (2.6, (-1.2, -3.7, 1.0), (-3.6, 0.0, -0.9)),
                   (3.05, (-4.0, -3.3, -0.5), (-3.93, 0.5, -1.28)), (4.0, (-4.3, -2.4, -0.95), (-3.93, 0.6, -1.28)),
                   (4.6, (-4.6, -2.3, -0.6), (-4.6, 0.0, -1.28)), (5.6, (-4.9, -1.6, 0.9), (-4.8, 0.0, -1.25)),
                   (6.5, (-5.0, -0.9, 0.55), (-4.85, 0.4, -1.2))]
        rx_keys = [(t, W_rx(p), W_rx(q)) for t, p, q in rx_keys]
        if RXFX.get("fx"):     # light lands on the PDs under 'Photodiodes: light in, current out'; amber signal under the TIA line
            RXFX["mod"].rx_light_arrive(RXFX["fx"], F(3, 1.95), F(3, 3.4))
            try:   # v4 (X): changing-bits glow starts when the first beam lands on a PD
                f_land = int(round(RXFX["mod"].rx_land_frames(RXFX["fx"], F(3, 1.95), F(3, 3.4))[0]))
            except Exception:
                f_land = F(3, 3.27)
            RXFX["mod"].rx_signal_glow(RXFX["fx"], f_land, F(3, 6.95))
            print("RX signal glow from frame", f_land)
    else:
        rx_keys = [(2.0, (X - 2.2, -2.3, 1.8), (X, 0, 0.4)), (6.5, (X + 5.4, -1.6, 1.2), (X + 8.0, 0, 0.1))]
    cam_jump(i, 2.0, (e_rx + ax_r * 0.025, e_rx - ax_r * 0.1, 40), (rx_keys[0][1], rx_keys[0][2], 26))
    for t, p, q in rx_keys[1:]:
        cam(i, t, p, q, 26)
    fill(i, C["cyan"], [(6.5, 0.0), (6.9, 1.0), (7.0, 1.0), (7.4, 0.0)])
    PR.head_look_at(hero.r, F(3, 6.9), subj_pt_fn(s, F(3, 8.4)), eye="rx", dur=1, overshoot=0.0)
    pov_e = eyew(hero, "rx", F(3, 7.2))
    subj_pt = subj_pt_fn(s, F(3, 8.4))
    pov_c = pov_e + (subj_pt - pov_e).normalized() * 0.05
    cam_jump(i, 6.95, (rx_keys[-1][1], rx_keys[-1][2], 26), (pov_c, subj_pt, 16))
    cam(i, 8.4, pov_c + (subj_pt - pov_c).normalized() * 0.08, subj_pt + Vector((0, 0, 0.03)), 19)
    cam(i, 9.5, pov_c + (subj_pt - pov_c).normalized() * 0.14, subj_pt + Vector((0, 0, 0.05)), 22)
    dpov = (subj_pt - pov_c).normalized()
    sidep = dpov.cross(Vector((0, 0, 1))).normalized()
    s4_end = (pov_e + sidep * 0.28 + Vector((0, 0, 0.1)) - dpov * 0.05, subj_pt + Vector((0, 0, 0.05)), 26)
    cam(i, 10.0, *s4_end)
    # v3 (owner): Gary is unaware - no glance into the lens
    gary_body = next((o for o in s.objs if o.type == "MESH" and o.name.endswith("_body")), s.root)
    blips = [crowd[k].root for k in list(crowd)[::12]]
    try:
        hud = VX.hud_overlay(bpy.context.scene, L.CAM, gary_body, F(3, 7.0), F(3, 9.95), style="intel", lock_at=F(3, 7.55),
                             target_name="GARY", blips=blips)
        print("HUD objects", len(hud))
    except Exception as ex:
        print("HUD FAILED", ex)
    lab(i, 0.0, 1.4, "HEAD TURN: LEFT EYE (RX)")
    lab(i, 2.0, 6.9, "RX: LENSLETS -> SI INTERLEAVER -> MIRROR -> 8 PHOTODIODES -> TIA" if rx_sub else
        "RX: LENSLETS -> PRISM -> PIN ARRAY -> TIA")

    note(i, 0.95, 6.9)


    # ---------------- S5: exploded hero centred (turned so the camera looks at the back wall), stamps, MCU, HUD, band, hatchery
    i = 4
    YAW5 = PI
    cam(i, 0.0, *s4_end)
    cam(i, 0.9, (-0.45, 7.75, 1.78), hero_head_w + Vector((0.1, 0, -0.05)), 30)
    cam(i, 1.8, exp_c + Vector((0.08, 1.05, 0.16)), exp_c, 28)
    bird_at(hero, i, 0.0, hp, 0.0)
    bird_at(hero, i, 0.9, hp, 0.0)
    bird_at(hero, i, 1.15, hp + Vector((0.18, 0, 0.03)), 0.0)
    bird_at(hero, i, 1.8, out, YAW5)
    bird_at(hero, i, S5_END - 0.1, out, YAW5)
    head(hero, i, 0.9, 0.0)
    head(hero, i, 1.8, 0.0)
    n_ex = 0
    for o in list(hero.top) + list(ghosts):
        src_name = ghosts.get(o, o.name)
        off = hero.ex_local.get(src_name)
        if off is None or o.hide_render:
            continue
        base = o.location.copy()
        for tt, d in ((1.6, 0.0), (2.4, 1.0), (7.3, 1.0), (7.75, 0.0)):
            K(o, F(i, tt), loc=tuple(base + off * d), interp="BEZIER")
        n_ex += 1
    mvt = hero.stamp.get("MERVALL")
    nar = hero.stamp.get("NARROWCOM")
    if mvt is not None and nar is not None and nar.animation_data and nar.animation_data.action:
        mvt.animation_data_create()
        mvt.animation_data.action = nar.animation_data.action.copy()
    print("HERO EXPLODE PARTS", n_ex)
    for m in hero.m_wings:      # item 8: wing panels fade to ~15 % while exploded (blended: dithered alpha is noisy at 1 sample)
        if hasattr(m, "surface_render_method"):
            m.surface_render_method = "BLENDED"
        b_ = principled(m)
        if b_ is not None:
            for tt, a in ((1.6, 1.0), (2.3, 0.15), (7.3, 0.15), (7.75, 1.0)):
                key_input_val(m, "Alpha", F(i, tt), a)
            for tt, v in ((1.6, 1.0), (2.3, 0.25), (7.3, 0.25), (7.75, 1.0)):     # darker while see-through (no white sheet)
                key_input_val(m, "Specular IOR Level", F(i, tt), 0.5 * v)
    acc, n_ = Vector(), 0
    for o in hero.top:
        if o.name in hero.cen and o.name in hero.ex_bird:
            acc += hero.cen[o.name] + hero.ex_bird[o.name]
            n_ += 1
    exp_ctr = bwm(out, YAW5, acc / max(1, n_)) + Vector((-0.14, 0.0, 0.0))    # shifted toward the label column (screen right)
    print("S5 exploded centre", tuple(round(v, 3) for v in exp_ctr), "parts", n_)
    dsp_w = glb_world_c(hero, "dsp_die", out, YAW5, explode=True)
    mcu_w = glb_world_c(hero, "elec_mcu", out, YAW5, explode=True)
    band_w = glb_world_c(hero, "leg_tx_band_eeprom_carrier", out, YAW5, explode=False)
    pts_mm = [hero.cen[o.name] + hero.ex_bird[o.name] for o in hero.top if o.name in hero.cen and o.name in hero.ex_bird
              and not o.name.startswith(("wing_", "skin_", "tail_", "leash"))]      # the visible exploded parts
    bmin = Vector([min(q[k] for q in pts_mm) for k in range(3)])
    bmax = Vector([max(q[k] for q in pts_mm) for k in range(3)])
    ctr_mm = (bmin + bmax) / 2.0
    h_w = (bmax.z - bmin.z) * MM * S_BIRD
    lab_ctr = bwm(out, YAW5, ctr_mm)
    d_orb = (h_w / 0.6) / (2 * 22.47 / 30.0)                # exploded bird ~60 % of the frame height at lens 30
    print("S5 ORBIT exploded bbox mm", tuple(round(v) for v in bmin), tuple(round(v) for v in bmax), "radius %.2f" % d_orb)
    for k in range(7):        # slow orbit only, no zoom in / out (v4e: closer, centred)
        u = k / 6.0
        a = math.radians(72.0 + 30.0 * u)
        cam(i, 2.45 + (7.3 - 2.45) * u, lab_ctr + Vector((math.cos(a) * d_orb, math.sin(a) * d_orb, 0.03 + 0.02 * math.sin(math.pi * u))),
            lab_ctr, 30)
    key5 = add_light("s5_hero_key", "AREA", tuple(lab_ctr + Vector((0.35, 0.8, 0.6))), 60.0, (1.0, 0.97, 0.92), size=0.8)
    key5.rotation_euler = (lab_ctr - key5.location).to_track_quat("-Z", "Y").to_euler()
    V(key5, i, 1.9, 7.6)          # the exploded hero keeps its light while the hall dims (v4e)
    FOCUS_KEYS.extend([(F(i, 1.95), lab_ctr), (F(i, 7.35), lab_ctr)])
    DOF_WINDOWS.append((F(i, 1.95), F(i, 7.35)))
    cam(i, 7.6, band_w + Vector((-0.02, 0.26, 0.04)), band_w, 40)
    cam(i, 7.84, band_w + Vector((-0.02, 0.22, 0.04)), band_w, 42)
    if nar is not None and mvt is not None:     # DSP lid stamp: NARROWCOM at ~42.6 s, MERVALL at ~43.4 s (scale "stamp")
        for o_, (ta, tb) in ((nar, (2.6, 3.4)), (mvt, (3.4, 7.85))):
            sc0 = o_.scale.copy()
            K(o_, F(i, ta), scale=tuple(sc0 * 1.6), interp="LINEAR")
            K(o_, F(i, ta + 0.1), scale=tuple(sc0), interp="BEZIER")
        L._VIS[nar] = L._VIS.get(nar, []) + [(F(0, 0), F(4, 0)), (F(4, 2.6), F(4, 3.4)), (F(4, 7.85), F(6, 0))]
        L._VIS[nar] = [w for w in L._VIS[nar] if w[0] < F(4, 0) or w[0] >= F(4, 2.6)]
        L._VIS[mvt] = [(F(4, 3.4), F(4, 7.85))]
    if hero.m_band is not None:
        pulse(hero.m_band, [(F(i, 7.55), 3.0), (F(i, 7.7), 12.0), (F(i, 7.9), 3.0)])
    # assembly hatchery (M, DevLog-019): die attach -> align + UV (isolator, Si lens) -> GSSG bonds -> lid press -> test, eyes on, hop
    ASM_F0 = F(i, 7.85)
    ASM_SPEED = 144.0 / 92.0      # v4 (M, DevLog-019 section 8): 144 build frames in a 92-frame slot (v3 62 + 30); v4k: same
    #  speed, M v4.2 ~288 build frames in the ASM_SLOT = 184 frame slot (W17)
    print("ASSEMBLY slot", ASM_SLOT, "build frames", AA._side().get("duration_frames"),
          "-> film frames %.1f" % (AA._side().get("duration_frames", 0) / ASM_SPEED))
    AA.key_assembly(asm_objs, ASM_F0, speed=ASM_SPEED)
    fill(i, (1.0, 1.0, 1.0), [(7.72, 0.0), (7.84, 1.0), (7.87, 1.0), (7.97, 0.0)])
    akeys = AA.camera_keys(asm_root, ASM_F0, speed=ASM_SPEED)
    prev = (band_w + Vector((-0.02, 0.22, 0.04)), band_w, 42)
    for k, (f, loc, tgt, lens, cut) in enumerate(akeys):
        t = (f - F(i, 0.0)) / 30.0

        if cut:
            cam_jump(i, round(t * 30) / 30.0, prev, (Vector(loc), Vector(tgt), lens))
            if k > 0:
                fill(i, (1.0, 1.0, 1.0), [(t - 0.06, 0.0), (t - 0.01, 0.85), (t + 0.02, 0.85), (t + 0.1, 0.0)])
        else:
            cam(i, t, Vector(loc), Vector(tgt), lens)
        prev = (Vector(loc), Vector(tgt), lens)
    a_last = prev
    cuts = sorted(f for f, l_, t_, ln, c in akeys if c and f > ASM_F0 + 1)
    shot_b = cuts[0] if cuts else ASM_F0 + 27
    shot_d = cuts[-1] if len(cuts) > 1 else ASM_F0 + 54     # M v4.1: A align, B bond, C enclosure close, D test stand
    note(i, 7.95, (shot_b - F(i, 0.0)) / 30.0)       # shots A and D show viewing-only beams (M)
    note(i, (shot_d - F(i, 0.0)) / 30.0 + 0.05, S5_END - 0.3)
    fill(i, (1.0, 1.0, 1.0), [(S5_END - 0.45, 0.0), (S5_END - 0.25, 1.0), (S5_END - 0.2, 1.0), (S5_END, 0.0)])
    s6_cam0 = (Vector((-0.1, 7.9, 1.75)), Vector((0.0, 12.0, 1.9)), 28)
    cam_jump(i, S5_END - 0.22, a_last, s6_cam0)
    print("ASSEMBLY f0", ASM_F0, "eyes_on", AA.event_frame("S5_eyes_on", ASM_F0, ASM_SPEED), "cams", len(akeys))
    belly_w = glb_world_c(hero, "pcb_belly_lowspeed", out, YAW5)
    cell_w = (hp.x + 0.09, hp.y, hp.z + 0.13)
    # v3: microcontroller telemetry (I2C spine + HUD card) dropped (owner)
    lab(i, 1.6, 7.3, "EXPLODED VIEW")
    lab(i, 7.95, S5_END - 0.25, "OUNOLIGHT HATCHERY")

    # ---------------- S6: flyers follow him out; Subject central; perched heads turn to him; bird-to-bird hops + sweep;
    # LEO train with satellite handoffs; CCTV links; end on the hero foreground + frozen Subject (VO rev 3 beats)
    i = 5

    def hook_perch(name):
        h = hk[name]
        return Vector(h["location"]), math.radians(h["yaw_deg"])
    FP = {"hero": hook_perch("HOOK_perch_x_scooter_bar"), "comp0": hook_perch("HOOK_perch_x_rail_post_23"),
          "comp1": hook_perch("HOOK_perch_hero")}
    flyers = [("hero", hero, (out, YAW5)), ("comp0", comps[0], cps[0]), ("comp1", comps[1], cps[1])]
    for j, (nm, b, (p0, y0)) in enumerate(flyers):
        perch, pyaw = FP[nm]
        bird_at(b, i, 0.0 + 0.25 * j, p0, y0)
        bird_at(b, i, 0.8 + 0.25 * j, (0.35 * (j - 1), 10.4, 2.0), PI / 2)
        bird_at(b, i, 1.6 + 0.25 * j, (0.3 * (j - 1), 12.9, 2.05), PI / 2)
        bird_at(b, i, 2.6 + 0.25 * j, (0.5 * (j - 1) - 0.8, 15.6, 2.0), PI / 2)
        bird_at(b, i, 4.1 + 0.25 * j, perch + Vector((0, 0, 0.3)), pyaw)
        bird_at(b, i, 4.5 + 0.25 * j, perch, pyaw)
        b.perch, b.yaw = perch, pyaw
        PR.flap_cycle(b.r, F(i, 0.1 + 0.25 * j), F(i, 4.35 + 0.25 * j))
        PR.wings_fold(b.r, F(i, 4.65 + 0.25 * j))
        head(b, i, 0.0, 0.0)
        PR.head_track(b.r, F(i, 4.7 + 0.25 * j), F(i, 7.0 if nm == "comp0" else 8.9), lambda f: SJ.head_world(s, f), step=(8, 14),
                      seed=20 + j)
    # Subject: to his mark (central), wary look-around, freeze; faces the camera side
    pass
    hperch = hero.perch
    hhead = hperch + Vector((0, 0, 0.21))
    pass
    gary_end = Vector(SJ.head_world(s, F(5, 9.5)))
    dgh = (Vector(hperch) - gary_end)
    dgh.z = 0.0
    dgh.normalize()
    sidev = dgh.cross(Vector((0, 0, 1)))
    cam_cc = hhead + dgh * 0.78 + sidev * 0.24 + Vector((0, 0, 0.15))
    camp = hhead + dgh * 0.5 + sidev * 0.14 + Vector((0, 0, 0.08))

    def eye_fn(r, k="tx"):
        return lambda f: Vector(r.eye_world(k, f))
    # 'Bird to bird. No fiber required.' (52.4-54.4): a beam hopping bird to bird + a searchlight sweep across the street
    hop_m = VX.data_beam_material("VFX_data_street", VX.TX_RED, strength=9.0, symbol_length=0.05, speed=2.0)
    # v3 (owner): 3-4 beams sweeping at any time + 2-3 point-to-point links that switch partners, S6 2.4-9.0 s
    SWEEPS = [("lamp1_top", [(2.4, 4.6, 15, 95, -38, -24), (4.7, 6.8, 95, 20, -30, -42), (6.9, 9.3, 20, 85, -40, -26)]),
              ("ledge_a", [(2.5, 4.5, -60, -5, -25, -35), (4.6, 6.9, -5, -70, -35, -22), (7.0, 9.3, -70, -15, -22, -34)]),
              ("ac_2", [(2.6, 4.8, 200, 250, -30, -20), (4.9, 7.0, 250, 190, -20, -33), (7.1, 9.3, 190, 240, -33, -24)]),
              ("roof_R", [(2.7, 5.0, 160, 230, -35, -45), (5.1, 7.2, 230, 170, -45, -32), (7.3, 9.3, 170, 215, -32, -42)])]
    n_sw = 0
    for bn, segs in SWEEPS:
        for (t0, t1, y0, y1, p0_, p1_) in segs:
            sw = VX.make_data_beam("sweep_%s_%d" % (bn, n_sw), (0, 0, 0), (0, 0, 1), radius=0.014, material=hop_m)
            VX.beam_sweep(sw, eye_fn(street[bn]), F(i, t0), F(i, t1), math.radians(y0), math.radians(y1), math.radians(p0_),
                          math.radians(p1_), length=11.0, wobble=0.02, seed=float(n_sw))
            V(sw, i, t0 - 0.05, t1 + 0.05)
            src = Vector(street[bn].root.location)
            endp = src + Vector((math.cos(math.radians(p1_)) * math.cos(math.radians(y1)),
                                 math.cos(math.radians(p1_)) * math.sin(math.radians(y1)), math.sin(math.radians(p1_)))) * 11.0
            EV(i, t0, "beam_sweep_%s_%d" % (bn, n_sw), "beam_sweep", pos=tuple(src), gain=0.35, dur=t1 - t0)
            EVENTS[-1]["_pos_end"] = (tuple(endp), F(i, t1))
            n_sw += 1
    LINKS = [["rail_a", "bin", "rail_b", "bin", "scooter"], ["wire_18.2", "ledge_b", "wire_19.0", "ledge_a"],
             ["lamp2_arm", "lamp1_arm", "ac_1", "lamp1_arm"]]
    for n, chain in enumerate(LINKS):
        lb = VX.make_data_beam("p2p_link_%d" % n, (0, 0, 0), (0, 0, 1), radius=0.012, material=hop_m)
        f_end = VX.beam_hops(lb, [eye_fn(street[c]) for c in chain], F(i, 2.45 + 0.2 * n), hop=4, hold=44 - 6 * n)
        V(lb, i, 2.4 + 0.2 * n, min(9.3, (f_end - F(5, 0.0)) / 30.0 + 0.05))
    print("S6 BEAMS sweeps", n_sw, "switching links", len(LINKS))
    # 'Satellite handoff. Again. Again.' (54.8-56.65): LEO train across the up-shot; the parapet bird's beam tracks and hands off
    sys.path.insert(0, os.path.join(ROOT_DIR, "scripts", "env"))
    import constellation as CN
    sats = CN.constellation_train(bpy.context.scene.collection, n=12, scale=0.5)     # v3: 2x smaller
    cn_root = sats[0].parent
    P_T0, P_T1 = 3.6, 8.2                                                            # v3: E's doubled pass = 2x faster
    PASS = (Vector(CN.S6_PASS_V3[0]), Vector(CN.S6_PASS_V3[1]))
    CN.constellation_pass(cn_root, PASS[0], PASS[1], F(i, P_T0), F(i, P_T1))
    for o in sats + [cn_root]:
        V(o, i, 3.6, 8.2)
    cc = ENV_INST.get("ENV_cable_car@end")
    hcc = hk.get("HOOK_cable_car")
    if cc is not None and hcc is not None:      # v3: cable car climbs slowly during S6
        p0c, p1c, hl = Vector(hcc["path_start"]), Vector(hcc["path_end"]), Vector(hcc["location"])
        for tt, u in ((0.0, 0.12), (10.0, 0.4)):
            K(cc, F(i, tt), loc=tuple(p0c.lerp(p1c, u) - hl), interp="LINEAR")
        print("CABLE CAR keyed")
    rt = ENV_INST.get("ENV_robotaxi@end")
    hrt = hk.get("HOOK_robotaxi")
    if rt is not None and hrt is not None and "path_start" in hrt:     # v4 (E): robotaxi creeps forward during S6
        hl = Vector(hrt["location"])
        for tt, pnt in ((0.0, hrt["path_start"]), (10.0, hrt["path_end"])):
            K(rt, F(i, tt), loc=tuple(Vector(pnt) - hl), interp="LINEAR")
        print("ROBOTAXI keyed")
    sat_inst = ENV_INST.get("ENV_satellite@end")
    if sat_inst is not None:     # the single satellite prop is replaced by the train
        L._VIS[sat_inst] = [(TOTAL + 5, TOTAL + 6)]
    upr = street["R3_parapet"]
    bpy.context.view_layer.update()
    f_t0, f_t1 = F(i, 4.85), F(i, 6.75)
    handoffs = [F(i, 5.22), F(i, 5.45), F(i, 5.68), F(i, 5.95), F(i, 6.24)]     # anchors 55.22 / 55.68 / 56.24 + in-between (faster train)
    tgt_sats = sats[1:7]

    def sat_at(o):
        return lambda f: (PASS[0].lerp(PASS[1], (f - F(i, P_T0)) / (F(i, P_T1) - F(i, P_T0)))
                          + (cn_root.matrix_world.to_3x3() @ o.location))
    mid_sat = sat_at(tgt_sats[1])(F(i, 5.7))
    PR.head_look_at(upr, F(i, 4.8), mid_sat, eye="tx", dur=5)
    up_m = VX.data_beam_material("VFX_data_uplink", VX.TX_RED, strength=10.0, symbol_length=0.5, speed=12.0)
    mimo_src = [street[n] for n in ("R3_parapet", "roof_L", "ledge_b", "wire_20.7")]
    for r_ in mimo_src[1:]:
        PR.head_look_at(r_, F(i, 4.8), mid_sat, eye="tx", dur=5)
    # 'Smile. The cameras are in on it.' (57.3-59.0): CCTV links (cyan, RX side)
    cctv_links = [(cctv[0], street["rail_b"]), (cctv[2], comps[0].r), (cctv[1], street["rail_a"])]
    for n, (c, rr) in enumerate(cctv_links):
        PR.head_look_at(rr, F(i, 6.9), c, eye="rx", dur=6)
        vbeam("cctv_link_%d" % n, c, Vector(rr.eye_world("rx", F(i, 7.6))), i, 7.8 + 0.15 * n, 9.3, color=VX.RX_CYAN, radius=0.015)
    for n, r in enumerate(list(street.values()) + [comps[1].r]):
        PR.head_look_at(r, F(i, 9.0) + (n % 5), gary_end, eye="auto", dur=4)
    PR.head_look_at(hero.r, F(i, 9.0), camp, eye="tx", dur=6)
    PR.tx_disable(hero.r, F(i, 9.55), F(i, 9.95), ramp=3)
    # camera: hall -> through the door -> left sidewalk (Subject central, perched birds left) -> up-shot -> CCTV -> hero + Subject
    up_aim = Vector((-3.2, 14.4, 1.4)) + Vector((math.cos(math.radians(75)), math.sin(math.radians(75)), math.tan(math.radians(30))))
    cc_aim = cam_cc + Vector((math.cos(math.radians(70)), math.sin(math.radians(70)), 0.04)) * 4.0
    cam(i, 0.0, *s6_cam0)
    cam(i, 1.0, (0.4, 8.7, 1.55), (0.0, 13.0, 1.9), 28)
    cam(i, 2.0, (0.1, 12.0, 2.05), (-0.5, 16.0, 1.8), 26)
    cam(i, 2.6, (-0.9, 13.6, 1.85), (-0.9, 17.5, 1.5), 24)
    cam(i, 3.2, (-2.6, 14.0, 1.65), (-0.6, 17.6, 1.45), 24)
    cam(i, 4.75, (-2.9, 14.2, 1.6), (-1.2, 19.6, 1.45), 24)
    cam(i, 5.05, (-3.2, 14.4, 1.4), up_aim, 18)
    cam(i, 6.75, (-3.18, 14.45, 1.4), up_aim + Vector((0.0, 0.0, 0.05)), 18)
    cam(i, 7.15, cam_cc, cc_aim, 21)      # v4: slightly wider so the billboards (SIGMOID, AGENTS.) stay in frame
    cam(i, 8.25, cam_cc + (camp - cam_cc) * 0.1, cc_aim, 21)
    wide = cam_cc + Vector((0.6, 0.3, 2.4))                  # 'Internet of Birds': wide network shot (stays on the street)
    cam(i, 8.7, wide, cc_aim + Vector((0, 0, 0.9)), 16)
    cam(i, 9.05, wide + Vector((0.05, 0.08, 0.0)), cc_aim + Vector((0, 0, 0.9)), 16)
    cam(i, 9.45, camp, hhead.lerp(gary_end, 0.3), 32)
    zoom_end = hhead + (camp - hhead) * 0.32 + Vector((0, 0, 0.01))       # v4 (W7): zoom into the bird on the scooter
    cam(i, 10.0, zoom_end, hhead + Vector((0, 0, 0.01)), 60)
    cam_end = (zoom_end, hhead + Vector((0, 0, 0.01)), 60)
    # MIMO uplinks (E): needs the S6 camera keys (links drop when a satellite leaves the frame)
    mres = VX.mimo_links([eye_fn(r_) for r_ in mimo_src], sats, F(i, 4.85), F(i, 6.75), k=2, cam=L.CAM, seed=3, radius=0.06,
                         material=up_m, isl=True)
    sched_up = [(s_[2], s_[1]) for s_ in mres["sessions"]]
    for s_ in mres["sessions"]:
        V(s_[4], i, (s_[2] - F(5, 0.0)) / 30.0 - 0.05, (s_[3] - F(5, 0.0)) / 30.0 + 0.1)
    for bm in mres.get("isl", []):
        V(bm, i, 4.85, 6.8)
    print("MIMO sessions", len(mres["sessions"]), "isl", len(mres.get("isl", [])))
    lab(i, 2.4, 4.45, "FREE-SPACE OPTICAL LINKS: BIRD TO BIRD")
    lab(i, 4.95, 6.85, "UPLINK: LEO SATELLITE HANDOFFS")
    lab(i, 7.3, 8.6, "DOWNLINK: STREET CCTV")
    lab(i, 8.6, 9.5, "THE INTERNET OF BIRDS")
    lab(i, 9.55, 10.0, "BLINK = TX DISABLE")
    note(i, 2.45, 4.45)
    note(i, 4.95, 6.85)
    note(i, 7.8, 9.3)
    KY.flush()
    subject_cam_check(s, F(0, 3.5), F(6, 0))
    for f in range(F(0, 3.5), F(6, 0), 15):
        hr = Vector(KY.values(hero.root, "location", f)) + Vector((0, 0, 0.2))
        print("NDC t %5.2f hero %s gary %s" % ((f - 1) / 30.0, ndc(hr, f), ndc(SJ.head_world(s, f), f)))

    # ---------------- world haze per scene (E): street on, hall low, macro sets off (only outside the fast tier)
    if os.environ.get("FILM_FAST", "1") != "1":
        wh = VX.world_haze(bpy.context.scene, 0.0035)
        for idx, t, d in ((0, 0.0, 0.0035), (0, 5.9, 0.0035), (0, 6.2, 0.0012), (2, 1.95, 0.0012), (2, 2.0, 0.0), (2, 8.3, 0.0),
                          (2, 8.35, 0.0012), (3, 1.95, 0.0012), (3, 2.0, 0.0), (3, 6.95, 0.0), (3, 7.0, 0.0012), (4, 8.08, 0.0012),
                          (4, 8.1, 0.002), (4, 9.75, 0.002), (4, 9.8, 0.0012), (5, 2.0, 0.0012), (5, 2.4, 0.0035)):
            wh.inputs["Density"].default_value = d
            wh.inputs["Density"].keyframe_insert("default_value", frame=F(idx, t))
        ad = bpy.context.scene.world.node_tree.animation_data
        for fc in (ad.action.fcurves if ad and ad.action else []):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"

    # ---------------- SFX cues (scripts/audio/scene_events.json); kinds in DevLog-011 section 3; positional ones gated (item 14)
    EV(0, 4.4, "street_ambience", "street_amb", dur=2.0)
    flaps(0, 4.5, 6.9, "hero_flight", gain=0.55, obj=hero.root)
    EV(0, 6.3, "camera_through_flap", "whoosh", pos=FLAP)
    EV(0, 5.85, "bird_flap_vent", "flap_vent", pos=FLAP)
    EV(0, 6.0, "hall_ambience_a", "hall_amb", dur=15.9)
    EV(2, 8.3, "hall_ambience_b", "hall_amb", dur=3.6)
    EV(3, 6.95, "hall_ambience_c", "hall_amb", dur=13.0)
    EV(0, 7.15, "hero_lands_pad", "landing_thump", pos=pad)
    EV(0, 7.15, "pad_charge", "pad_charge_hum", pos=pad, dur=0.6)
    EV(0, 7.45, "hero_hops_off_pad", "hop", pos=pad)
    flaps(0, 7.45, 7.85, "hero_drop", gain=0.4, obj=hero.root)
    EV(0, 8.2, "hero_docks", "cell_dock", pos=hp)
    EV(0, 8.38, "beak_bites_lip", "beak_latch", pos=hp)
    EV(0, 8.4, "heads_track_subject", "head_servo_crowd", dur=1.6)
    EV(1, 0.55, "hero_head_turn", "head_servo", pos=hp)
    EV(1, 2.9, "beak_bite", "beak_latch", pos=lip_w)
    EV(1, 3.15, "beak_release", "beak_open", pos=lip_w)
    EV(1, 3.2, "lip_swings_up", "cell_undock", pos=lip_w)
    hop_t = (HF["hop0"] - 301) / 30.0
    on_t = (HF["on"] - 301) / 30.0
    in_t = (HF["in"] - 301) / 30.0
    EV(1, hop_t, "hero_hops_out", "hop", pos=hp)
    flaps(1, hop_t, on_t, "hero_hop_flutter", gain=0.45, obj=hero.root)
    EV(1, on_t, "hero_lands_hand", "landing_thump", obj=hero.root, gain=0.5)
    EV(1, 4.6, "tx_side_glow", "eye_glow_on", obj=hero.root)
    EV(1, 6.3, "rx_side_glow", "eye_glow_on", obj=hero.root)
    EV(1, in_t - 0.3, "hero_hops_back", "hop", obj=hero.root)
    for nm, tt in (("gnd", t_g), ("pwr", t_p), ("sig", t_s)):
        EV(1, tt, "contact_" + nm, "contact_mate", pos=hp)
    EV(1, in_t, "hero_docks", "cell_dock", pos=hp)
    EV(1, 9.62, "beak_bites_lip", "beak_latch", pos=hp)
    EV(1, 9.75, "eyes_on", "eye_glow_on", pos=hp)
    EV(2, 0.1, "hero_head_turn", "head_servo", pos=hp)
    EV(2, 0.6, "fill_into_tx_eye", "fill_whoosh")
    EV(2, 1.0, "tx_pic", "pic_hum", dur=7.0)
    EV(2, 1.05, "dfb_on", "dfb_on")
    EV(2, 1.1, "pic_lightup", "pic_lightup", dur=6.3)
    EV(2, 4.95, "tx_rf_signal", "elec_glow", dur=3.35)
    EV(3, 3.27, "rx_signal", "elec_glow", dur=3.68)
    EV(2, 5.0, "mzm_pam4", "pam4_chatter", dur=2.5)
    EV(2, 7.2, "lenslets_launch_beams", "beam_launch")
    EV(2, 7.98, "fill_out_of_tx_eye", "fill_whoosh")
    EV(2, 8.4, "beams_cross_aisle", "beam_launch", pos=hp)
    EV(2, 8.45, "tx_link", "link_blip", pos=hp, params=None)
    EV(3, 0.1, "hero_head_turn", "head_servo", pos=hp)
    EV(3, 0.9, "rx_link", "link_blip", pos=hp)
    EV(3, 1.4, "fill_into_rx_eye", "fill_whoosh")
    EV(3, 2.0, "rx_chip", "tia_hum", dur=4.9)
    for k in range(8):
        EV(3, 2.9 + 0.06 * k, "pd_flash_%d" % k, "pd_flash", gain=0.7)
    EV(3, 6.5, "fill_out_of_rx_eye", "fill_whoosh")
    for k in range(1):
        EV(3, 7.1 + 0.4 * k, "hud_tick", "hud_tick")
    EV(3, 7.55, "pov_hud_lock", "hud_lock")
    for k in range(3):
        EV(3, 8.8 + 0.4 * k, "hud_tick", "hud_tick")
    EV(3, 9.55, "pull_out_of_pov", "whoosh", gain=0.6)
    EV(4, 0.9, "hero_floats_out", "cell_undock", pos=hp)
    EV(4, 1.6, "explode", "explode", pos=out, gain=1.6)
    EV(4, 2.6, "dsp_stamp_narrowcom", "assembly_clunk", pos=dsp_w, gain=0.8)
    EV(4, 3.4, "dsp_stamp_mervall", "assembly_clunk", pos=dsp_w, gain=0.8)
    EV(4, 7.6, "reassemble", "snap_back", pos=out, gain=1.6)
    EV(4, 7.8, "flashback_in", "whoosh")
    ev_ = lambda nm: (AA.event_frame(nm, ASM_F0, ASM_SPEED) - 1) / 30.0 - 40.0
    for nm, kind in (("S2_iso_place", "align_stage"), ("S2_lens2_place", "align_stage"), ("S2_uv_on", "uv_cure"),
                     ("S2_dfb_on", "dfb_on"), ("S4_press_down", "lid_press"), ("S4_lid_closed", "lid_press"), ("S5_heads_in", "align_stage"),
                     ("S5_eyes_on", "eye_glow_on"), ("S5_eyes_full", "eye_glow_on"), ("S5_heads_out", "align_stage"), ("S5_hop1", "hop")):
        EV(4, ev_(nm), "assembly_" + nm.lower(), kind, gain=0.7, **({"dur": 6 / 30.0 / ASM_SPEED * 1.0} if kind == "uv_cure" else {}))
    # v4k: M v4.2 lists every bond (S3_bond_NN, build frames); SFX = the bonds that form inside shot B (B cut .. C cut)
    _ev = AA._side()["events"]
    _shot_c = cuts[1] if len(cuts) > 2 else shot_b + 27
    _bonds = sorted(ASM_F0 + v / ASM_SPEED for k_, v in _ev.items() if k_.startswith("S3_bond_") and k_[8:].isdigit())
    _bonds = [f for f in _bonds if shot_b <= f < _shot_c]
    if _bonds:
        EV(4, (_bonds[0] - 1) / 30.0 - 40.0, "assembly_wire_bonds", "wire_bonder", gain=0.6,
           params={"count": len(_bonds), "step": round(_ev.get("S3_bond_period", 5) / 30.0 / ASM_SPEED, 3)})
    else:
        EV(4, (shot_b - 1) / 30.0 - 40.0 + 0.05, "assembly_wire_bonds", "wire_bonder", gain=0.6,
           params={"count": 6, "step": round(5 / 30.0 / ASM_SPEED, 3)})      # bonds forming in shot B (27 build frames)
    print("ASSEMBLY wire-bond SFX", len(_bonds), "bonds in shot B")
    EV(4, S5_END - 0.25, "flashback_out", "whoosh")
    EV(5, 0.0, "street_ambience", "street_amb", dur=10.0)
    f_door = next((f for f in range(F(4, 0), F(6, 0)) if KY.values(s.root, "location", f)[1] >= 13.06), None)
    if f_door is not None:
        EVF(f_door, "subject_exits_door", "door", pos=(0.0, 13.06, 1.2))
        print("SUBJECT exits the door at frame", f_door, "t %.2f s" % ((f_door - 1) / 30.0))
    EV(5, 2.2, "camera_through_door", "whoosh")
    for j in range(3):
        flaps(5, 0.1 + 0.25 * j, 4.35 + 0.25 * j, "flyer_%d" % j, gain=0.45, obj=flyers[j][1].root)
        EV(5, 4.5 + 0.25 * j, "flyer_%d_lands" % j, "landing_thump", pos=FP[flyers[j][0]][0])
    for k in range(5):
        EV(5, 2.45 + k * 10 / 30.0, "bird_to_bird_hop_%d" % k, "link_blip", gain=0.55)
    EV(5, 2.5, "street_sweep", "beam_launch", gain=0.5)
    EV(5, 4.95, "satellite_uplink", "sat_ping")
    for k, h in enumerate(handoffs):
        EVF(h, "satellite_handoff_%d" % k, "sat_ping")
    for k, (f_, j_) in enumerate(sorted(sched_up)):
        if f_ > F(5, 4.9):
            EVF(f_, "mimo_reconfig_%d" % k, "mimo_reconfig")
    for n in range(3):
        EV(5, 7.8 + 0.15 * n, "cctv_%d" % n, "cctv_whir")
    EV(5, 9.0, "heads_to_subject", "head_servo_crowd", dur=0.5)
    EV(5, 9.4, "blink", "blink", pos=hperch)
    EV(6, 0.0, "disclaimer", "disclaimer_whoosh", dur=2.0)
    for f in range(F(0, 3.4), F(6, 0), 15):   # footsteps from the Subject's root motion (moving > 0.15 m in 0.5 s)
        p0 = KY.values(s.root, "location", f)
        p1 = KY.values(s.root, "location", f + 15)
        if (Vector(p1) - Vector(p0)).length > 0.15:
            EVF(f, "footstep", "footstep", pos=tuple(p0), gain=0.4)
    for n, lk in enumerate(sched_hall.links):
        sb = next((b for b in specs if b.id == lk.tx), None)
        if sb is None or (Vector(sb.pos) - hp).length > 4.0 or n % 3:
            continue
        EVF(lk.f0, "crowd_link", "link_blip", pos=sb.pos, gain=0.2)
        EVF(lk.f1, "crowd_link_drop", "link_drop", pos=sb.pos, gain=0.15)
    for n, lk in enumerate(sched_street.links):
        sb = next((b for b in sspecs if b.id == lk.tx), None)
        if n % 2 or sb is None:
            continue
        EVF(lk.f0, "street_link", "link_blip", gain=0.25, pos=sb.pos)

    # ---------------- S7
    i = 6
    V(box("black", (0, -300, 0), (14, 0.1, 18), (0, 0, 0), emit=100), i, 0, S7_LEN)
    L.OVL["DISC"] = dict(L.OVL["DISC"], loc=(0.0, 2.0), ax="CENTER")
    L.OVL["SRC"] = dict(L.OVL["SRC"], loc=(0.0, -0.05), ax="CENTER")
    L.OVL["CRED"] = dict(size=0.055, color=(0.8, 0.8, 0.8), loc=(0.0, -0.75), ax="CENTER", ay="TOP", shadow=False)
    L.WRAP["CRED"] = 110
    ovt("CRED", "BIRD PHOTOS (Wikimedia Commons, CC BY / CC BY-SA): M. Barera, E. Willoughby, M. Dickison, shankar s., Quartl, "
                "H. Zell, MrPanyGoff, B. Ross, JJ Harrison, C. J. Sharp, J. Janssen, Cephas, L. Hulbert; dodo: after R. Savery (PD). "
                "VO: Kokoro-82M (Apache-2.0). Music + SFX: procedural. Full credits in the description.", i, 0, S7_LEN)
    # v4k (owner 2026-10-10): the original disclaimer back, verbatim (as in v3 / crude); replaces the shortened W11 text
    ovt("DISC", "DISCLAIMER:\nPigeons are real. Parody; not\naffiliated with any company or bird.\nOne typical 800G DR8 build shown.\n"
                "Real DR8 uses 16 fibers via MPO;\nhere the eyes do free-space optics\n(stylized lenslet arrays).\n"
                "Modules cannot see. 1311 nm beams\ndrawn visible. Real pigeons: 4 toes.", i, 0, S7_LEN)
    src = ("SOURCES\nOSFP MSA Module Specification Rev 5.2 / 5.22; FS and Approved Networks 800G DR8 datasheets\n"
           "Credo / DustPhotonics (CW DFB butt-coupled to SiPh); Gigalight DR8 block diagram; Semtech GN1825 octal TIA\n"
           "Stiller et al., Nature 2024 and Prum et al., Nature 2015 (bird phylogeny); RFC 1149 (1990-04-01)\n"
           "Form-factor dates: SFF/SNIA, OIF, MSA releases; XPO MSA (Arista, 2026)")
    ovt("SRC", src, i, 0, S7_LEN)
    cam_jump(6, 0.0, cam_end, ((0, -306, 0), (0, -300, 0), 28))
    return s


def subj_pt_fn(s, f):
    return Vector(SJ.head_world(s, f))



def setup_vfx(scn):
    """Film look until workstream E delivers its library: compositor fog glow (neon, eyes, beams) and low-density haze
    volumes in the hall and the street (EEVEE volumetrics, coarse tiles)."""
    scn.use_nodes = True
    nt = scn.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    gl = nt.nodes.new("CompositorNodeGlare")
    gl.glare_type = "FOG_GLOW"
    gl.quality = "MEDIUM"
    gl.threshold = 1.15
    gl.size = 7
    gl.mix = -0.35
    out = nt.nodes.new("CompositorNodeComposite")
    nt.links.new(rl.outputs["Image"], gl.inputs["Image"])
    nt.links.new(gl.outputs["Image"], out.inputs["Image"])
    for attr, val in (("volumetric_tile_size", "16"), ("volumetric_samples", 24), ("volumetric_end", 40.0)):
        if hasattr(scn.eevee, attr):
            setattr(scn.eevee, attr, val)
    for name, loc, size, dens, col, wins in (("haze_hall", (0, 6.1, CEIL / 2), (6.4, 13.8, CEIL), 0.009, (0.75, 0.82, 1.0), HALL_VIS),
                                             ("haze_street", (0, 23.5, 6.0), (18.0, 21.0, 12.0), 0.008, (0.8, 0.82, 0.95), STREET_VIS)):
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        o = bpy.context.active_object
        o.name = name
        o.scale = size
        m = bpy.data.materials.new(name + "_vol")
        m.use_nodes = True
        t = m.node_tree
        for n in list(t.nodes):
            t.nodes.remove(n)
        vol = t.nodes.new("ShaderNodeVolumePrincipled")
        vol.inputs["Density"].default_value = dens
        vol.inputs["Color"].default_value = (*col, 1)
        mo = t.nodes.new("ShaderNodeOutputMaterial")
        t.links.new(vol.outputs[0], mo.inputs["Volume"])
        o.data.materials.append(m)
        o.visible_shadow = False
        vis(o, wins)


def setup_label_shots(scn, fstop=3.2, dim=0.42, ramp=8):
    """v4e (owner): blur + darken the background in the labelled shots: camera DOF on the subject, hall lights dimmed."""
    cam = L.CAM
    foc = empty("dof_focus", (0.0, 0.0, 0.0))
    for f, p in sorted(FOCUS_KEYS):
        K(foc, f, loc=tuple(p), interp="CONSTANT")
    cam.data.dof.focus_object = foc
    cam.data.dof.use_dof = False         # v4h: DOF only inside the S5 orbit window (was on all film at f/128)
    cam.data.dof.keyframe_insert("use_dof", frame=1)
    for f0, f1 in DOF_WINDOWS:
        for f, v in ((f0 - ramp, True), (f1 + ramp + 1, False)):
            cam.data.dof.use_dof = v
            cam.data.dof.keyframe_insert("use_dof", frame=f)
    for fc in cam.data.animation_data.action.fcurves:
        if fc.data_path == "dof.use_dof":
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"
    big = 128.0
    keys = [(1, big)]
    for f0, f1 in DOF_WINDOWS:
        keys += [(f0 - ramp, big), (f0, fstop), (f1, fstop), (f1 + ramp, big)]
    for f, v in sorted(keys):
        cam.data.dof.aperture_fstop = v
        cam.data.dof.keyframe_insert("aperture_fstop", frame=f)
    lights = [o for o in bpy.data.objects if o.type == "LIGHT" and o.name.split(".")[0] in ("hall_area", "accent", "face_fill")]
    for lo in lights:
        e0 = lo.data.energy
        for f0, f1 in DOF_WINDOWS:
            for f, v in ((f0 - ramp, e0), (f0, e0 * dim), (f1, e0 * dim), (f1 + ramp, e0)):
                lo.data.energy = v
                lo.data.keyframe_insert("energy", frame=f)
        lo.data.energy = e0
    print("LABEL SHOTS: DOF windows", DOF_WINDOWS, "dimmed lights", len(lights))


def export_callouts(scn, path, ramp=6):
    """Anchors of the post-drawn callouts: {"W", "H", "frames": {frame: [{text, x, y, side, a}]}} (y from the top)."""
    from bpy_extras.object_utils import world_to_camera_view
    W, H = scn.render.resolution_x, scn.render.resolution_y
    cam = scn.camera

    def anchor(o):
        mw = o.matrix_world
        if o.type == "MESH":
            return sum((mw @ Vector(v) for v in o.bound_box), Vector()) / 8.0
        return mw.translation.copy()
    CALLOUTS[:] = [c for c in CALLOUTS if c.get("scene") == 4]     # v4h (owner): callouts only in the S5 exploded view
    for c in CALLOUTS:          # fixed column per label (balanced by hand where not given)
        if "side" not in c:
            scn.frame_set((c["f0"] + c["f1"]) // 2)
            c["side"] = "R" if world_to_camera_view(scn, cam, anchor(c["obj"])).x > 0.5 else "L"
    frames = {}
    fr = sorted({f for c in CALLOUTS for f in range(c["f0"], c["f1"] + ramp + 1)})
    for f in fr:
        scn.frame_set(f)
        row = []
        for c in CALLOUTS:
            if not (c["f0"] <= f <= c["f1"] + ramp):
                continue
            a = min(1.0, (f - c["f0"] + 1) / float(ramp)) if f < c["f0"] + ramp else 1.0
            if f > c["f1"]:
                a = max(0.0, 1.0 - (f - c["f1"]) / float(ramp))
            co = world_to_camera_view(scn, cam, anchor(c["obj"]))
            if co.z <= 0 or (c.get("inframe") and not (0.03 <= co.x <= 0.97 and 0.03 <= co.y <= 0.97)):
                continue
            row.append({"text": c["text"], "x": round(co.x * W, 1), "y": round((1.0 - co.y) * H, 1), "side": c["side"], "a": round(a, 3)})
        if row:
            frames[str(f)] = row
    with open(path, "w") as fh:
        json.dump({"W": W, "H": H, "generator": "scripts/build_film.py export_callouts",
                   "frames": frames}, fh, indent=0)
    print("CALLOUTS", len(CALLOUTS), "frames", len(frames), path)


def export_texts(scn, path, first=136):
    """v4f: {"W", "H", "frames": {frame: [{kind: header|beamnote, text, a}]}} for every main-pass frame with a header or the
    beam note (same windows as the former in-render overlay; t1 exclusive, no fades)."""
    frames = {}
    for t in sorted(TEXTS, key=lambda d: (d["f0"], d["kind"])):
        for f in range(max(first, t["f0"]), min(t["f1"], TOTAL + 1)):
            frames.setdefault(str(f), []).append({"kind": t["kind"], "text": t["text"], "a": 1.0})
    with open(path, "w") as fh:
        json.dump({"W": scn.render.resolution_x, "H": scn.render.resolution_y, "generator": "scripts/build_film.py export_texts",
                   "frames": frames}, fh, indent=0)
    print("TEXTS", len(TEXTS), "frames", len(frames), path)


def build(out_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    # draft speed: 1 sample, 540x674 (about half of 1080x1350, even height for H.264), no shadows, persistent data between frames
    scn.eevee.taa_render_samples = int(os.environ.get("FILM_SAMPLES", "1"))
    tier = os.environ.get("FILM_TIER", "draft")   # preview 360x450 / draft 540x674 / review 1080x1350
    tw, th = {"preview": (360, 450), "draft": (540, 674), "review": (1080, 1350)}.get(tier, (540, 674))
    scn.render.resolution_x = int(os.environ.get("FILM_W", tw))
    scn.render.resolution_y = int(os.environ.get("FILM_H", th))
    scn.render.use_persistent_data = True
    for attr, val in (("use_shadows", False), ("use_raytracing", False), ("use_gtao", False)):
        if hasattr(scn.eevee, attr):
            setattr(scn.eevee, attr, val)
    scn.render.fps = 30
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.view_settings.view_transform = "AgX"
    scn.frame_start = 1
    scn.frame_end = TOTAL
    # E's look (DevLog-013): EEVEE preset with screen-space reflections + volumetrics, AgX Punchy (coordinator), night sky world
    # owner (2026-10-10): fast render until after v2 or until detailed VFX: 1 sample, no ray tracing, no volumetrics
    FAST = os.environ.get("FILM_FAST", "1") == "1"
    VX.setup_eevee(scn, os.environ.get("FILM_PRESET", "draft"), raytracing=not FAST, volumetrics=not FAST, view_transform="AgX",
                   look="AgX - Punchy")
    scn.eevee.taa_render_samples = int(os.environ.get("FILM_SAMPLES", "1" if FAST else "8"))
    if FAST:
        for attr, val in (("use_shadows", os.environ.get("FILM_SHADOWS", "0") == "1"), ("shadow_ray_count", 1), ("shadow_step_count", 1),
                          ("use_gtao", False), ("use_volumetric_shadows", False)):
            if hasattr(scn.eevee, attr):
                setattr(scn.eevee, attr, val)
    scn.render.use_persistent_data = True
    scn.world = VX.night_sky_world(scn)
    # overlay plane 6 cm in front of the lens (v0: 0.8 m), so walls closer than 0.8 m no longer hide captions/labels
    L.HOLD_Z = 0.06
    L.HOLD_S0 = L.HOLD_Z / 3.0
    L.setup_camera()
    L.CAM.data.clip_start = 0.004
    main_build()
    if FILM_DEBUG:
        for i in range(6):
            for sec in range(10):
                L.ovt("TC", "S%d  0:%02d" % (i + 1, sec), i, sec, sec + 1)
        for sec in range(2):
            L.ovt("TC", "S7  0:%02d" % sec, 6, sec, sec + 1)
    # v4 (W2): VO captions are burned in post by A from the VO timings; nothing rendered here
    if os.environ.get("FILM_COMP", "1") == "1":   # v2 (coordinator): glare/bloom only; no grain or vignette
        N = VX.setup_compositor(scn, bloom_threshold=2.5, bloom_size=6, bloom_mix=-0.55, vignette=0.0, grain=0.0)
        if isinstance(N, dict) and "bloom" in N:
            N["bloom"].quality = "MEDIUM"
    else:
        scn.render.use_compositing = False
    for m in bpy.data.materials:   # text / HUD overlays (pure emission): brighter under AgX so white reads white
        if m.node_tree and any(n.type == "EMISSION" for n in m.node_tree.nodes) and not m.name.startswith("VFX"):
            for n in m.node_tree.nodes:
                if n.type == "EMISSION" and n.inputs["Strength"].default_value <= 1.0:
                    n.inputs["Strength"].default_value = 1.5
    L.finalize_visibility(TOTAL)
    setup_label_shots(scn)
    add_pan()
    for e in EVENTS:      # beam_sweep: pan of the sweep end at its end frame (A pans the sweep)
        pe = e.pop("_pos_end", None)
        if pe is None:
            continue
        c, tg, lens = cam_state(pe[1])
        fwd = (tg - c).normalized()
        right = fwd.cross(Vector((0, 0, 1))).normalized()
        d = Vector(pe[0]) - c
        depth = max(0.01, d.dot(fwd))
        e.setdefault("params", {})["pan_end"] = round(max(-1.0, min(1.0, d.dot(right) / (depth * 18.0 / lens))), 3)
        e["ungated"] = True
    gate_events()
    for e in EVENTS:
        e.pop("ungated", None)
    EVENTS.sort(key=lambda e: e["t"])
    ev_path = os.path.join(ROOT_DIR, "scripts", "audio", "scene_events.json")
    with open(ev_path, "w") as fh:
        json.dump({"schema": "pigeon film scene_events v1", "fps": 30, "duration_s": TOTAL / 30.0,
                   "generator": "scripts/build_film.py", "events": EVENTS}, fh, indent=1)
    print("EVENTS", len(EVENTS), ev_path)
    if CALLOUTS:
        export_callouts(scn, os.path.join(ROOT_DIR, "outputs", "film_v4_callouts.json"))
    export_texts(scn, os.path.join(ROOT_DIR, "outputs", "film_v4_texts.json"))
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=out_path)
    print("SAVED", out_path)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    build(argv[0] if argv else "film_v3.blend")
