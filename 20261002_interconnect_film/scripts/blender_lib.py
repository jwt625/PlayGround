"""Helpers for the crude storyboard build (Blender 4.2, run headless). v0.1

Scene index i occupies frames 1 + i*300 ... (i+1)*300 and world x offset i*100.
All scene coordinates passed to helpers are scene-local (x, y, z); LP() adds the offset.
"""
import math
import textwrap

import bpy

FPS = 30
SCN_FRAMES = 300
LENS0 = 28.0
HOLD_Z = 0.8             # overlay plane distance from camera (in front of all geometry)
HOLD_S0 = HOLD_Z / 3.0   # overlay layout numbers were designed for a plane 3 m away

CAM = None
TGT = None
HOLD = None
_MATS = {}
_VIS = {}  # obj -> list of (f0, f1) visible windows (frames, f1 exclusive)
PI = math.pi


def F(idx, t):
    return 1 + idx * SCN_FRAMES + int(math.floor(t * FPS + 0.5))  # round half up


def OX(idx):
    return idx * 100.0


def LP(idx, p):
    return (OX(idx) + p[0], p[1], p[2])


# ---------------------------------------------------------------- materials
def mat(color, rough=0.85, emit=0.0, unique=False):
    key = (tuple(round(c, 3) for c in color), rough, emit)
    if not unique and key in _MATS:
        return _MATS[key]
    m = bpy.data.materials.new("m%d" % len(bpy.data.materials))
    m.use_nodes = True
    nt = m.node_tree
    if emit and emit >= 100:  # pure emission (text, black card)
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        e = nt.nodes.new("ShaderNodeEmission")
        e.inputs["Color"].default_value = (*color, 1)
        e.inputs["Strength"].default_value = 1.0
        o = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(e.outputs[0], o.inputs[0])
    else:
        b = nt.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*color, 1)
        b.inputs["Roughness"].default_value = rough
        if emit:
            b.inputs["Emission Color"].default_value = (*color, 1)
            b.inputs["Emission Strength"].default_value = emit
    if not unique:
        _MATS[key] = m
    return m


def key_emit(m, frame, color, strength=None, interp="CONSTANT"):
    """Keyframe a unique Principled material's base+emission color (and strength)."""
    b = m.node_tree.nodes["Principled BSDF"]
    for nm in ("Base Color", "Emission Color"):
        b.inputs[nm].default_value = (*color, 1)
        b.inputs[nm].keyframe_insert("default_value", frame=frame)
    if strength is not None:
        b.inputs["Emission Strength"].default_value = strength
        b.inputs["Emission Strength"].keyframe_insert("default_value", frame=frame)
    ad = m.node_tree.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                if int(round(kp.co[0])) == frame:
                    kp.interpolation = interp


def seq_material(first_png, nframes, start_frame, strength=1.0):
    """Emission material showing an image sequence; frame 'start_frame' (global) shows the first image."""
    m = bpy.data.materials.new("seq")
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    tex = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(first_png)
    img.source = "SEQUENCE"
    tex.image = img
    tex.interpolation = "Closest"
    iu = tex.image_user
    iu.frame_duration = nframes
    iu.frame_start = start_frame
    iu.frame_offset = 0
    iu.use_auto_refresh = True
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


def _fin(o, name, color, parent, rough, emit, smooth=False):
    o.name = name
    o.data.materials.append(mat(color, rough, emit))
    if smooth:
        for p in o.data.polygons:
            p.use_smooth = True
    if parent is not None:
        o.parent = parent
    return o


def box(name, loc, size, color, parent=None, rot=None, rough=0.85, emit=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = bpy.context.active_object
    o.scale = size
    if rot:
        o.rotation_euler = rot
    return _fin(o, name, color, parent, rough, emit)


def cyl(name, loc, r, h, color, axis="Z", parent=None, verts=24, rough=0.85, emit=0.0, scale=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=h, location=loc)
    o = bpy.context.active_object
    if axis == "X":
        o.rotation_euler = (0, PI / 2, 0)
    elif axis == "Y":
        o.rotation_euler = (PI / 2, 0, 0)
    if scale:
        o.scale = scale
    return _fin(o, name, color, parent, rough, emit, smooth=verts >= 16)


def sph(name, loc, r, color, parent=None, scale=None, rough=0.85, emit=0.0, seg=20, rings=12):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, segments=seg, ring_count=rings, location=loc)
    o = bpy.context.active_object
    if scale:
        o.scale = scale
    return _fin(o, name, color, parent, rough, emit, smooth=True)


def torus(name, loc, R, r, color, rot=None, parent=None, scale=None, emit=0.0, material=None, segs=32):
    bpy.ops.mesh.primitive_torus_add(major_radius=R, minor_radius=r, major_segments=segs, minor_segments=8, location=loc)
    o = bpy.context.active_object
    if rot:
        o.rotation_euler = rot
    if scale:
        o.scale = scale
    o = _fin(o, name, color, parent, 0.6, emit, smooth=True)
    if material is not None:
        o.data.materials.clear()
        o.data.materials.append(material)
    return o


def empty(name, loc=(0, 0, 0), parent=None):
    o = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    if parent is not None:
        o.parent = parent
    return o


def plane(name, loc, size, material, rot=(PI / 2, 0, 0), parent=None):
    """Plane facing -Y by default (rot x = 90 deg); UV up = world up."""
    bpy.ops.mesh.primitive_plane_add(size=1, location=loc)
    o = bpy.context.active_object
    o.scale = (size[0], size[1], 1)
    o.rotation_euler = rot
    o.name = name
    o.data.materials.append(material)
    if parent is not None:
        o.parent = parent
    return o


# ---------------------------------------------------------------- animation
def _interp(obj, path, frame, interp):
    ad = obj.animation_data
    if not ad or not ad.action:
        return
    for fc in ad.action.fcurves:
        if fc.data_path == path:
            for kp in fc.keyframe_points:
                if int(round(kp.co[0])) == frame:
                    kp.interpolation = interp


def K(obj, frame, loc=None, rot=None, scale=None, interp="LINEAR"):
    if loc is not None:
        obj.location = loc
        obj.keyframe_insert("location", frame=frame)
        _interp(obj, "location", frame, interp)
    if rot is not None:
        obj.rotation_euler = rot
        obj.keyframe_insert("rotation_euler", frame=frame)
        _interp(obj, "rotation_euler", frame, interp)
    if scale is not None:
        obj.scale = scale
        obj.keyframe_insert("scale", frame=frame)
        _interp(obj, "scale", frame, interp)


def at(obj, idx, pts, interp="LINEAR"):
    """Key scene-local locations: pts = [(t, (x, y, z)), ...]."""
    for t, p in pts:
        K(obj, F(idx, t), loc=LP(idx, p), interp=interp)


def sc(obj, idx, pts, interp="LINEAR"):
    for t, s in pts:
        if not isinstance(s, (tuple, list)):
            s = (s, s, s)
        K(obj, F(idx, t), scale=s, interp=interp)


def rt(obj, idx, pts, interp="LINEAR"):
    for t, r in pts:
        K(obj, F(idx, t), rot=r, interp=interp)


def V(obj, idx, t0=0.0, t1=10.0):
    """Register a visibility window in scene idx (seconds, t1 exclusive)."""
    _VIS.setdefault(obj, []).append((F(idx, t0), F(idx, t1)))
    return obj


def VA(obj, f0, f1):
    _VIS.setdefault(obj, []).append((f0, f1))
    return obj


def finalize_visibility(total_frames):
    for obj, wins in _VIS.items():
        wins = sorted(wins)
        events = []
        if wins[0][0] > 1:
            events.append((1, True))
        for f0, f1 in wins:
            events.append((f0, False))
            events.append((min(f1, total_frames + 1), True))
        by_frame = {}
        for f, hidden in events:
            if f in by_frame and by_frame[f] is False:
                continue
            by_frame[f] = hidden
        for f in sorted(by_frame):
            h = by_frame[f]
            obj.hide_render = h
            obj.hide_viewport = h
            obj.keyframe_insert("hide_render", frame=f)
            obj.keyframe_insert("hide_viewport", frame=f)


# ---------------------------------------------------------------- camera
def setup_camera():
    global CAM, TGT, HOLD
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36
    cd.lens = LENS0
    cd.clip_end = 2000
    cd.clip_start = 0.05
    CAM = bpy.data.objects.new("CAM", cd)
    bpy.context.scene.collection.objects.link(CAM)
    TGT = empty("CAM_TARGET")
    c = CAM.constraints.new("TRACK_TO")
    c.target = TGT
    c.track_axis = "TRACK_NEGATIVE_Z"
    c.up_axis = "UP_Y"
    HOLD = empty("OVERLAY_HOLDER", (0, 0, -HOLD_Z), parent=CAM)
    bpy.context.scene.camera = CAM
    return CAM


def shot(idx, t0, t1, p0, a0, p1=None, a1=None, lens=LENS0, lens1=None, ease="LINEAR"):
    """One continuous camera move from t0 to t1 (seconds); hard cut at t1."""
    p1 = p0 if p1 is None else p1
    a1 = a0 if a1 is None else a1
    lens1 = lens if lens1 is None else lens1
    f0 = F(idx, t0)
    f1 = max(F(idx, t1) - 1, f0)
    K(CAM, f0, loc=LP(idx, p0), interp=ease)
    K(TGT, f0, loc=LP(idx, a0), interp=ease)
    CAM.data.lens = lens
    CAM.data.keyframe_insert("lens", frame=f0)
    _interp(CAM.data, "lens", f0, ease)
    HOLD.scale = (HOLD_S0 * LENS0 / lens,) * 3
    HOLD.keyframe_insert("scale", frame=f0)
    _interp(HOLD, "scale", f0, ease)
    if f1 > f0:
        K(CAM, f1, loc=LP(idx, p1), interp="CONSTANT")
        K(TGT, f1, loc=LP(idx, a1), interp="CONSTANT")
        CAM.data.lens = lens1
        CAM.data.keyframe_insert("lens", frame=f1)
        _interp(CAM.data, "lens", f1, "CONSTANT")
        HOLD.scale = (HOLD_S0 * LENS0 / lens1,) * 3
        HOLD.keyframe_insert("scale", frame=f1)
        _interp(HOLD, "scale", f1, "CONSTANT")
    else:
        _interp(CAM, "location", f0, "CONSTANT")
        _interp(TGT, "location", f0, "CONSTANT")


# ---------------------------------------------------------------- text
WHITE = (1, 1, 1)
OVL = {
    "CAP": dict(size=0.27, color=WHITE, loc=(0, -1.5), ax="CENTER", ay="CENTER", shadow=True),
    "BIG": dict(size=0.62, color=(1.0, 0.9, 0.25), loc=(0, 0.3), ax="CENTER", ay="CENTER", shadow=True),
    "LAB": dict(size=0.17, color=(0.35, 0.95, 1.0), loc=(-1.85, 1.9), ax="LEFT", ay="TOP", shadow=True),
    "CARD": dict(size=0.1, color=(0.85, 0.85, 0.85), loc=(-1.85, -2.37), ax="LEFT", ay="BOTTOM", shadow=True),
    "FX": dict(size=0.115, color=(1.0, 0.85, 0.2), loc=(-1.85, 2.35), ax="LEFT", ay="TOP", shadow=True),
    "TC": dict(size=0.1, color=(0.6, 0.6, 0.6), loc=(1.85, -2.37), ax="RIGHT", ay="BOTTOM", shadow=False),
    "DISC": dict(size=0.19, color=WHITE, loc=(-1.8, 1.9), ax="LEFT", ay="TOP", shadow=False),
    "SRC": dict(size=0.062, color=WHITE, loc=(-1.8, -0.1), ax="LEFT", ay="TOP", shadow=False),
}
WRAP = {"CAP": 24, "BIG": 10, "LAB": 32, "CARD": 62, "FX": 48, "TC": 20, "SRC": 100, "DISC": 40}


def _text(body, size, color, loc, parent, ax, ay, billboard=False, rot=None):
    cu = bpy.data.curves.new("t", "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = ax
    cu.align_y = ay
    o = bpy.data.objects.new("txt", cu)
    bpy.context.scene.collection.objects.link(o)
    o.data.materials.append(mat(color, emit=100))
    o.location = loc
    o.visible_shadow = False
    if rot:
        o.rotation_euler = rot
    if parent is not None:
        o.parent = parent
    if billboard:
        c = o.constraints.new("TRACK_TO")
        c.target = CAM
        c.track_axis = "TRACK_Z"
        c.up_axis = "UP_Y"
    return o


def ov(kind, body, f0, f1):
    """Camera-attached overlay text visible for frames [f0, f1)."""
    d = OVL[kind]
    lines = []
    for ln in body.split("\n"):
        lines += textwrap.wrap(ln, WRAP[kind]) or [""]
    body = "\n".join(lines)
    objs = []
    if d["shadow"]:
        sh = _text(body, d["size"], (0, 0, 0), (d["loc"][0] + 0.02, d["loc"][1] - 0.02, -0.01), HOLD, d["ax"], d["ay"])
        objs.append(sh)
    o = _text(body, d["size"], d["color"], (d["loc"][0], d["loc"][1], 0), HOLD, d["ax"], d["ay"])
    objs.append(o)
    for x in objs:
        VA(x, f0, f1)
    return o


def ovt(kind, body, idx, t0, t1):
    return ov(kind, body, F(idx, t0), F(idx, t1))


def narr(idx, segs):
    """segs = [(t0, t1, text)]; splits into <=3-word / <=22-char caption chunks timed by word count."""
    for t0, t1, text in segs:
        words = text.split()
        chunks, cur = [], []
        for w in words:
            if cur and (len(cur) >= 3 or len(" ".join(cur + [w])) > 22):
                chunks.append(cur)
                cur = []
            cur.append(w)
        if cur:
            chunks.append(cur)
        total = sum(len(c) for c in chunks)
        t = t0
        for c in chunks:
            dt = (t1 - t0) * len(c) / total
            ovt("CAP", " ".join(c), idx, t, t + dt)
            t += dt


def wl(body, pos, idx, t0, t1, size=0.3, color=(1.0, 1.0, 1.0)):
    """World-space billboard label (white with a dark offset shadow copy as a child)."""
    o = _text(body, size, color, LP(idx, pos), None, "CENTER", "CENTER", billboard=True)
    sh = _text(body, size, (0, 0, 0), (size * 0.05, -size * 0.05, -0.01), o, "CENTER", "CENTER")
    V(o, idx, t0, t1)
    V(sh, idx, t0, t1)
    return o


# ---------------------------------------------------------------- cast
class Cast:
    pass


def make_cast():
    c = Cast()
    skin = (0.92, 0.72, 0.55)
    # Gary
    c.g = empty("GARY", (-1000, 0, 0))
    c.gt = empty("GARY_TILT", parent=c.g)
    sph("g_body", (0, 0, 0.55), 1.0, (0.2, 0.4, 0.85), c.gt, scale=(0.3, 0.24, 0.42))
    sph("g_head", (0, 0, 1.08), 0.2, skin, c.gt)
    sph("g_hat", (0, 0, 1.2), 0.22, (1.0, 0.55, 0.1), c.gt, scale=(1, 1, 0.55))
    cyl("g_brim", (0, -0.05, 1.17), 0.27, 0.02, (1.0, 0.55, 0.1), parent=c.gt)
    sph("g_nose", (0, -0.2, 1.05), 0.035, (0.95, 0.65, 0.5), c.gt)
    for sx in (-1, 1):
        sph("g_ear", (sx * 0.2, 0, 1.08), 0.045, skin, c.gt)
        sph("g_eye", (sx * 0.075, -0.17, 1.1), 0.05, (1, 1, 1), c.gt)
        sph("g_pup", (sx * 0.075, -0.215, 1.1), 0.025, (0, 0, 0), c.gt)
        cyl("g_arm", (sx * 0.34, 0, 0.62), 0.055, 0.42, skin, parent=c.gt)
        cyl("g_leg", (sx * 0.1, 0, 0.17), 0.075, 0.36, (0.15, 0.25, 0.55), parent=c.gt)
    # bullet holes (through-holes along Y), hidden until their hit time; scale (s,s,1) shrinks the radius only
    c.holes = []
    for i, (hx, hz) in enumerate([(0.08, 0.62), (-0.1, 0.5), (0.12, 0.42), (-0.06, 0.72), (0.0, 0.33)]):
        h = cyl("hole%d" % (i + 1), (hx, 0, hz), 0.065, 0.62, (0.02, 0.02, 0.02), axis="Y", parent=c.gt)
        c.holes.append(h)
    # head hole (through the head along X) and Gary's own gun (S3 self-shot)
    c.head_hole = cyl("headhole", (0, 0, 1.08), 0.05, 0.5, (0.02, 0.02, 0.02), axis="X", parent=c.gt)
    c.self_gun = [
        cyl("sg_barrel", (0.62, 0, 1.1), 0.04, 0.8, (0.45, 0.45, 0.5), axis="X", parent=c.gt),
        cyl("sg_barrel2", (0.62, 0, 1.18), 0.04, 0.8, (0.45, 0.45, 0.5), axis="X", parent=c.gt),
        box("sg_stock", (1.15, 0, 1.0), (0.3, 0.1, 0.1), (0.6, 0.35, 0.18), c.gt),
        cyl("sg_arm", (0.5, 0, 0.8), 0.05, 0.5, skin, parent=c.gt),
    ]
    c.self_flash = sph("sg_flash", (0.2, 0, 1.12), 0.14, (1.0, 0.85, 0.2), c.gt, emit=8.0)
    c.self_ring = torus("sg_ring", (0.0, 0, 1.12), 0.18, 0.045, (0.85, 0.85, 0.85), rot=(0, PI / 2, 0), parent=c.gt)
    # Manager
    c.m = empty("MANAGER", (-1000, 0, 0))
    sph("m_body", (0, 0, 0.7), 1.0, (0.3, 0.3, 0.38), c.m, scale=(0.36, 0.27, 0.52))
    sph("m_head", (0, 0, 1.33), 0.23, (0.95, 0.6, 0.55), c.m)
    c.head_red = sph("m_head_red", (0, 0, 1.33), 0.235, (0.95, 0.12, 0.08), c.m)
    sph("m_nose", (0, -0.235, 1.3), 0.04, (0.9, 0.5, 0.45), c.m)
    for sx in (-1, 1):
        sph("m_ear", (sx * 0.23, 0, 1.33), 0.05, (0.95, 0.6, 0.55), c.m)
    box("m_tie", (0, -0.255, 0.72), (0.08, 0.02, 0.42), (0.8, 0.1, 0.1), c.m)
    c.brows = []
    for sx in (-1, 1):
        sph("m_eye", (sx * 0.085, -0.2, 1.37), 0.04, (1, 1, 1), c.m)
        sph("m_pup", (sx * 0.085, -0.235, 1.37), 0.02, (0, 0, 0), c.m)
        b = box("m_brow", (sx * 0.085, -0.215, 1.45), (0.12, 0.03, 0.025), (0.05, 0.03, 0.02), c.m, rot=(0, sx * -0.4, 0))
        c.brows.append((b, sx))
        cyl("m_leg", (sx * 0.12, 0, 0.2), 0.08, 0.42, (0.2, 0.2, 0.25), parent=c.m)
    c.mouth = box("m_mouth", (0, -0.225, 1.26), (0.1, 0.02, 0.035), (0.05, 0.0, 0.0), c.m)
    c.steam = [sph("steam_ear", (0, 0, 0), 0.06, (0.95, 0.95, 0.95), c.m, seg=10, rings=6) for _ in range(6)]
    # right arm holder with shotgun
    c.arm = empty("m_arm", (0.36, 0, 1.0), parent=c.m)
    cyl("m_armcyl", (0, -0.2, 0), 0.06, 0.45, (0.3, 0.3, 0.38), axis="Y", parent=c.arm)
    c.gun_parts = [
        box("gun_stock", (0, -0.05, -0.02), (0.07, 0.3, 0.12), (0.6, 0.35, 0.18), c.arm),
        cyl("gun_barrel", (0, -0.55, 0.01), 0.04, 0.85, (0.45, 0.45, 0.5), axis="Y", parent=c.arm),
        cyl("gun_barrel2", (0.075, -0.55, 0.01), 0.04, 0.85, (0.45, 0.45, 0.5), axis="Y", parent=c.arm),
    ]
    c.flash = sph("flash", (0.04, -1.1, 0.01), 0.18, (1.0, 0.85, 0.2), c.arm, emit=8.0)
    c.smoke_ring = torus("smoke_ring", (0.04, -1.3, 0.01), 0.2, 0.05, (0.85, 0.85, 0.85), rot=(PI / 2, 0, 0), parent=c.arm)
    # left arm holder with an active-optical-cable bundle (S6 only)
    c.larm = empty("m_larm", (-0.36, 0, 1.0), parent=c.m)
    c.whip_parts = [
        cyl("w_arm", (0, -0.2, 0), 0.06, 0.45, (0.3, 0.3, 0.38), axis="Y", parent=c.larm),
        cyl("w_handle", (0, -0.5, 0), 0.04, 0.3, (0.1, 0.1, 0.12), axis="Y", parent=c.larm),
    ]
    aoc_cols = [(1.0, 0.5, 0.1), (0.3, 0.9, 1.0), (1.0, 0.9, 0.2), (1.0, 0.4, 0.7), (0.3, 1.0, 0.4)]
    for k, col in enumerate(aoc_cols):
        ox_, oz_ = 0.03 * math.cos(k * 1.256), 0.03 * math.sin(k * 1.256)
        c.whip_parts.append(cyl("aoc_cable", (ox_, -1.55, oz_), 0.014, 2.0, col, axis="Y", parent=c.larm, emit=0.6))
        c.whip_parts.append(box("aoc_end", (ox_, -2.58, oz_), (0.07, 0.16, 0.05), (0.6, 0.6, 0.65), c.larm))
    c.printout = box("printout", (0.0, -0.6, 0.0), (0.4, 0.02, 0.55), (1, 1, 1), c.arm)
    return c


def yaw_to(src, dst):
    dx, dy = dst[0] - src[0], dst[1] - src[1]
    return math.atan2(dx, -dy)


def gary_at(c, idx, t, pos, yaw=0.0, tilt=0.0, interp="CONSTANT"):
    f = F(idx, t)
    K(c.g, f, loc=LP(idx, pos), rot=(0, 0, yaw), interp=interp)
    K(c.gt, f, rot=(tilt, 0, 0), interp=interp)


def mgr_at(c, idx, t, pos, yaw=0.0, interp="CONSTANT"):
    f = F(idx, t)
    K(c.m, f, loc=LP(idx, pos), rot=(0, 0, yaw), interp=interp)


BROW_BASE = (0.12, 0.03, 0.025)
MOUTH_BASE = (0.1, 0.02, 0.035)


def face(c, idx, t, anger, interp="LINEAR"):
    """Manager expression: anger 0..1 (brows steepen, mouth opens)."""
    f = F(idx, t)
    for b, sx in c.brows:
        K(b, f, rot=(0, sx * -(0.4 + 0.65 * anger), 0), interp=interp)
    K(c.mouth, f, scale=(MOUTH_BASE[0] * (1 + 0.5 * anger), MOUTH_BASE[1], MOUTH_BASE[2] * (1 + 3.2 * anger)), interp=interp)


def park(c, idx, t=0.0):
    gary_at(c, idx, t, (-900, 0, 0))
    mgr_at(c, idx, t, (-900, 0, 0))
    K(c.arm, F(idx, t), rot=(1.45, 0, 0), interp="CONSTANT")
    K(c.larm, F(idx, t), rot=(1.45, 0, 0), interp="CONSTANT")
    face(c, idx, t, 0.0, interp="CONSTANT")


def topple(c, idx, t_bang, gary_pos, gary_yaw):
    """Gary falls backwards after a hit and stays down until the end of the scene."""
    fall = min(0.9, 9.85 - t_bang)
    K(c.g, F(idx, t_bang), loc=LP(idx, gary_pos), rot=(0, 0, gary_yaw), interp="CONSTANT")
    K(c.gt, F(idx, t_bang), rot=(0, 0, 0), interp="LINEAR")
    K(c.gt, F(idx, t_bang + 0.28 * fall), rot=(0.18, 0, 0), interp="BEZIER")
    K(c.gt, F(idx, t_bang + fall), rot=(-1.25, 0, 0), interp="BEZIER")
    K(c.gt, F(idx, 9.9), rot=(-1.25, 0, 0), interp="CONSTANT")


HITS = []  # (hole object, bang frame)


def shoot(c, idx, mgr_pos, gary_pos, t_turn, t_bang, hole, walk=None, stare_yaw=None, gary_yaw=None, caption=True):
    """Manager aims at Gary from t_turn and fires at t_bang; Gary gets hole #hole and topples.

    walk = (t0, from_pos, t1): Manager walks from from_pos at t0 to mgr_pos at t1 (facing stare_yaw if given).
    """
    aim = yaw_to(mgr_pos, gary_pos)
    pre = aim if stare_yaw is None else stare_yaw
    if walk is not None:
        t0, fp, t1 = walk
        K(c.m, F(idx, t0), loc=LP(idx, fp), rot=(0, 0, pre), interp="LINEAR")
        K(c.m, F(idx, t1), loc=LP(idx, mgr_pos), rot=(0, 0, pre), interp="CONSTANT")
        t_app = t0
    else:
        t_app = min(t_turn, 0.0)
    K(c.arm, F(idx, t_app), rot=(1.45, 0, 0), interp="CONSTANT")
    K(c.m, F(idx, t_turn), loc=LP(idx, mgr_pos), rot=(0, 0, aim), interp="CONSTANT")
    K(c.arm, F(idx, t_turn), rot=(1.45, 0, 0), interp="BEZIER")
    K(c.arm, F(idx, t_turn + 0.35), rot=(0.12, 0, 0), interp="LINEAR")
    K(c.arm, F(idx, t_bang), rot=(0.12, 0, 0), interp="LINEAR")
    K(c.arm, F(idx, t_bang + 0.07), rot=(-0.35, 0, 0), interp="BEZIER")
    K(c.arm, F(idx, t_bang + 0.4), rot=(0.12, 0, 0), interp="BEZIER")
    for g in c.gun_parts:
        V(g, idx, 0, 10)
    V(c.flash, idx, t_bang, t_bang + 0.12)
    V(c.smoke_ring, idx, t_bang + 0.05, t_bang + 0.6)
    K(c.smoke_ring, F(idx, t_bang + 0.05), scale=(0.4, 0.4, 0.4), interp="LINEAR")
    K(c.smoke_ring, F(idx, t_bang + 0.6), scale=(1.8, 1.8, 1.8), interp="CONSTANT")
    K(c.smoke_ring, F(idx, t_bang + 0.05), loc=(0.04, -1.3, 0.01), interp="LINEAR")
    K(c.smoke_ring, F(idx, t_bang + 0.6), loc=(0.04, -2.1, 0.01), interp="CONSTANT")
    if caption:
        ovt("BIG", "BANG", idx, t_bang, t_bang + 0.6)
    VA(c.holes[hole - 1], F(idx, t_bang), 10 ** 6)
    HITS.append((c.holes[hole - 1], F(idx, t_bang)))
    gy = gary_yaw if gary_yaw is not None else yaw_to(gary_pos, mgr_pos)
    K(c.g, F(idx, t_turn), loc=LP(idx, gary_pos), rot=(0, 0, gy), interp="CONSTANT")
    topple(c, idx, t_bang, gary_pos, gy)


def self_shoot(c, idx, gary_pos, gary_yaw, t_gun, t_bang, caption=True):
    """Gary pulls out a shotgun at t_gun and shoots himself in the head at t_bang."""
    for g in c.self_gun:
        V(g, idx, t_gun, 10)
    V(c.self_flash, idx, t_bang, t_bang + 0.12)
    V(c.self_ring, idx, t_bang + 0.05, t_bang + 0.6)
    K(c.self_ring, F(idx, t_bang + 0.05), scale=(0.4, 0.4, 0.4), interp="LINEAR")
    K(c.self_ring, F(idx, t_bang + 0.6), scale=(1.8, 1.8, 1.8), interp="CONSTANT")
    K(c.self_ring, F(idx, t_bang + 0.05), loc=(0.0, 0, 1.12), interp="LINEAR")
    K(c.self_ring, F(idx, t_bang + 0.6), loc=(-0.8, 0, 1.12), interp="CONSTANT")
    if caption:
        ovt("BIG", "BANG", idx, t_bang, t_bang + 0.6)
    VA(c.head_hole, F(idx, t_bang), 10 ** 6)
    HITS.append((c.head_hole, F(idx, t_bang)))
    topple(c, idx, t_bang, gary_pos, gary_yaw)


def schedule_holes(step_frames=45, factor=0.9, floor=0.3):
    """Holes slowly recover after each hit (in steps), but never fully heal."""
    for obj, f_bang in HITS:
        s = 1.0
        f = f_bang
        K(obj, f, scale=(1, 1, 1), interp="CONSTANT")
        while f < 1860:
            f += step_frames
            s = max(floor, s * factor)
            K(obj, f, scale=(s, s, 1), interp="CONSTANT")


# ---------------------------------------------------------------- materials with alpha
def set_blend(m):
    try:
        m.surface_render_method = "BLENDED"
    except Exception:
        m.blend_method = "BLEND"
    return m


def key_alpha(m, frame, a, interp="LINEAR"):
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Alpha"].default_value = a
    b.inputs["Alpha"].keyframe_insert("default_value", frame=frame)
    ad = m.node_tree.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            if "Alpha" in fc.data_path:
                for kp in fc.keyframe_points:
                    if int(round(kp.co[0])) == frame:
                        kp.interpolation = interp


def fade_quad(idx, keys):
    """Full-frame black fade behind the captions. keys = [(t, alpha)] in scene-i seconds."""
    m = mat((0, 0, 0), 1.0, 0.0, unique=True)
    m.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.0
    set_blend(m)
    q = plane("fade", (0, 0, -0.15), (30, 30), m, rot=(0, 0, 0), parent=HOLD)
    for t, a in keys:
        key_alpha(m, F(idx, t), a)
    VA(q, F(idx, keys[0][0]), F(idx, keys[-1][0]) + 1)
    return q


# ---------------------------------------------------------------- curve helper (logo marks)
def curve_line(name, pts, bevel, color, parent=None, loc=(0, 0, 0), emit=0.0):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = bevel
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for k, (x, y, z) in enumerate(pts):
        sp.points[k].co = (x, y, z, 1)
    o = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(o)
    o.data.materials.append(mat(color, 0.6, emit))
    o.location = loc
    if parent is not None:
        o.parent = parent
    return o


def valley_mark(name, loc, size, color, parent=None, emit=0.0):
    """Parody mark: a rounded valley (an arch with the tip flipped upside down) over a wave line."""
    pts = [(-0.5 * size + size * k / 16.0,
            0, -0.5 * size * math.sin(PI * k / 16.0)) for k in range(17)]
    wave = [(-0.9 * size + 1.8 * size * k / 24.0, 0, -0.75 * size + 0.12 * size * math.sin(2 * PI * k / 12.0)) for k in range(25)]
    a = curve_line(name + "_v", pts, 0.05 * size, color, parent, loc, emit)
    b = curve_line(name + "_w", wave, 0.04 * size, color, parent, loc, emit)
    return [a, b]


# ---------------------------------------------------------------- other people
class Person:
    pass


def make_person(name, idx, loc, shirt, logo, logo_color=(1, 1, 1), skin=(0.9, 0.7, 0.55), scale=1.0, yaw=0.0,
                win=(0, 10), logo_size=0.1, badge=None):
    """Clay person with a parody logo on the shirt. Returns a Person with root, arm_r (holder), mouth, parts."""
    p = Person()
    r = empty(name, LP(idx, loc))
    r.rotation_euler = (0, 0, yaw)
    r.scale = (scale, scale, scale)
    p.root = r
    parts = [sph(name + "_b", (0, 0, 0.55), 1.0, shirt, r, scale=(0.3, 0.24, 0.42)),
             sph(name + "_h", (0, 0, 1.05), 0.19, skin, r),
             sph(name + "_nose", (0, -0.19, 1.02), 0.035, skin, r),
             sph(name + "_e1", (-0.07, -0.16, 1.08), 0.04, (1, 1, 1), r),
             sph(name + "_e2", (0.07, -0.16, 1.08), 0.04, (1, 1, 1), r),
             sph(name + "_p1", (-0.07, -0.195, 1.08), 0.02, (0, 0, 0), r),
             sph(name + "_p2", (0.07, -0.195, 1.08), 0.02, (0, 0, 0), r),
             cyl(name + "_l1", (-0.1, 0, 0.17), 0.075, 0.36, (0.2, 0.2, 0.3), parent=r),
             cyl(name + "_l2", (0.1, 0, 0.17), 0.075, 0.36, (0.2, 0.2, 0.3), parent=r)]
    p.mouth = box(name + "_mouth", (0, -0.185, 0.96), (0.08, 0.02, 0.025), (0.1, 0.0, 0.0), r)
    parts.append(p.mouth)
    p.arm_l = empty(name + "_al", (-0.34, 0, 0.8), parent=r)
    parts.append(cyl(name + "_alc", (0, 0, -0.18), 0.055, 0.42, skin, parent=p.arm_l))
    p.arm_r = empty(name + "_ar", (0.34, 0, 0.8), parent=r)
    parts.append(cyl(name + "_arc", (0, 0, -0.18), 0.055, 0.42, skin, parent=p.arm_r))
    if badge is not None:
        parts.append(box(name + "_badge", (0, -0.248, 0.6), (0.34, 0.012, 0.16), badge, r))
    parts.append(_text(logo, logo_size, logo_color, (0, -0.262, 0.6), r, "CENTER", "CENTER", rot=(PI / 2, 0, 0)))
    # gun held by the right arm (hidden until used): barrel along the arm's -Z (forward when the arm is raised)
    p.gun = [cyl(name + "_barrel", (0, 0, -0.65), 0.035, 0.8, (0.45, 0.45, 0.5), parent=p.arm_r),
             box(name + "_stock", (0, 0, -0.12), (0.06, 0.08, 0.3), (0.6, 0.35, 0.18), p.arm_r)]
    p.flash = sph(name + "_flash", (0, 0, -1.1), 0.16, (1.0, 0.85, 0.2), p.arm_r, emit=8.0)
    p.hole = cyl(name + "_hole", (0.05, 0, 0.55), 0.065, 0.62, (0.02, 0.02, 0.02), axis="Y", parent=r)
    for o in parts:
        V(o, idx, *win)
    p.parts = parts
    p.idx = idx
    p.win = win
    p.loc = loc
    p.yaw = yaw
    return p


def person_at(p, t, pos, yaw, tilt=0.0, interp="CONSTANT"):
    f = F(p.idx, t)
    K(p.root, f, loc=LP(p.idx, pos), rot=(tilt, 0, yaw), interp=interp)


def shout(p, t0, t1, period=0.25):
    """Mouth opens and closes (shouting)."""
    t = t0
    k = 0
    while t < t1:
        wide = (k % 2 == 0)
        K(p.mouth, F(p.idx, t), scale=((0.11, 0.02, 0.07) if wide else (0.08, 0.02, 0.025)), interp="CONSTANT")
        t += period * (0.8 + 0.4 * ((k * 7) % 3) / 2.0)
        k += 1
    K(p.mouth, F(p.idx, t1), scale=(0.08, 0.02, 0.025), interp="CONSTANT")


def brawl(p, t0, t1, toward, seed=0, amp=0.18):
    """Shove / punch loop: arms swing, body lurches toward a neighbour and back, with yaw wobble."""
    import random as _r
    rng = _r.Random(seed)
    base = p.loc
    t = t0
    k = 0
    d = (toward[0] - base[0], toward[1] - base[1])
    n = math.hypot(*d) or 1.0
    d = (d[0] / n, d[1] / n)
    while t < t1:
        lunge = (k % 2 == 0)
        off = amp * (1.0 if lunge else -0.3)
        pos = (base[0] + d[0] * off, base[1] + d[1] * off, 0)
        K(p.root, F(p.idx, t), loc=LP(p.idx, pos), rot=(0.0, 0, p.yaw + rng.uniform(-0.35, 0.35)), interp="CONSTANT")
        K(p.arm_r, F(p.idx, t), rot=((-1.3 if lunge else 0.2), 0, rng.uniform(-0.4, 0.4)), interp="CONSTANT")
        K(p.arm_l, F(p.idx, t), rot=((0.2 if lunge else -1.2), 0, rng.uniform(-0.4, 0.4)), interp="CONSTANT")
        t += 0.17 + 0.05 * rng.random()
        k += 1
    K(p.root, F(p.idx, t1), loc=LP(p.idx, base), rot=(0.0, 0, p.yaw), interp="CONSTANT")
    K(p.arm_r, F(p.idx, t1), rot=(0, 0, 0), interp="CONSTANT")
    K(p.arm_l, F(p.idx, t1), rot=(0, 0, 0), interp="CONSTANT")


def person_shoot(p, t_draw, t_bang, aim_yaw, pos, caption_hole=True):
    """Person pulls a gun, aims (yaw), fires; muzzle flash only."""
    for g in p.gun:
        V(g, p.idx, t_draw, p.win[1])
    V(p.flash, p.idx, t_bang, t_bang + 0.12)
    K(p.root, F(p.idx, t_draw), loc=LP(p.idx, pos), rot=(0, 0, aim_yaw), interp="CONSTANT")
    K(p.arm_r, F(p.idx, t_draw), rot=(0.2, 0, 0), interp="BEZIER")
    K(p.arm_r, F(p.idx, t_draw + 0.25), rot=(-1.45, 0, 0), interp="LINEAR")
    K(p.arm_r, F(p.idx, t_bang), rot=(-1.45, 0, 0), interp="LINEAR")
    K(p.arm_r, F(p.idx, t_bang + 0.06), rot=(-1.8, 0, 0), interp="BEZIER")
    K(p.arm_r, F(p.idx, t_bang + 0.3), rot=(-1.45, 0, 0), interp="CONSTANT")


def person_fall(p, t, pos, yaw, dur=0.7):
    """Person shot at time t: hole appears, falls backwards (tilt about local X) and stays down."""
    VA(p.hole, F(p.idx, t), 10 ** 6)
    K(p.root, F(p.idx, t), loc=LP(p.idx, pos), rot=(0, 0, yaw), interp="LINEAR")
    K(p.root, F(p.idx, t + 0.25 * dur), loc=LP(p.idx, pos), rot=(0.15, 0, yaw), interp="BEZIER")
    K(p.root, F(p.idx, min(t + dur, 9.85)), loc=LP(p.idx, pos), rot=(-1.45, 0, yaw), interp="BEZIER")
    K(p.root, F(p.idx, 9.9), loc=LP(p.idx, pos), rot=(-1.45, 0, yaw), interp="CONSTANT")


def shoot_person(c, idx, mgr_pos, target_pos, t_turn, t_bang):
    """Manager aims at a person (not Gary) and fires; the target's fall is keyed separately."""
    aim = yaw_to(mgr_pos, target_pos)
    K(c.m, F(idx, t_turn), loc=LP(idx, mgr_pos), rot=(0, 0, aim), interp="CONSTANT")
    K(c.arm, F(idx, t_turn), rot=(1.45, 0, 0), interp="BEZIER")
    K(c.arm, F(idx, t_turn + 0.35), rot=(0.12, 0, 0), interp="LINEAR")
    K(c.arm, F(idx, t_bang), rot=(0.12, 0, 0), interp="LINEAR")
    K(c.arm, F(idx, t_bang + 0.07), rot=(-0.35, 0, 0), interp="BEZIER")
    K(c.arm, F(idx, t_bang + 0.4), rot=(0.12, 0, 0), interp="BEZIER")
    for g in c.gun_parts:
        V(g, idx, 0, 10)
    V(c.flash, idx, t_bang, t_bang + 0.12)
    V(c.smoke_ring, idx, t_bang + 0.05, t_bang + 0.6)
    K(c.smoke_ring, F(idx, t_bang + 0.05), scale=(0.4, 0.4, 0.4), loc=(0.04, -1.3, 0.01), interp="LINEAR")
    K(c.smoke_ring, F(idx, t_bang + 0.6), scale=(1.8, 1.8, 1.8), loc=(0.04, -2.1, 0.01), interp="CONSTANT")


def ear_steam(c, idx, t0, t1):
    """Steam puffs rising from the Manager's ears while angry."""
    for k, o in enumerate(c.steam):
        side = -1 if k % 2 == 0 else 1
        t = t0 + 0.15 * k
        V(o, idx, t, t1)
        while t < t1:
            K(o, F(idx, t), loc=(side * 0.26, 0, 1.4), scale=(0.4, 0.4, 0.4), interp="LINEAR")
            K(o, F(idx, t + 0.5), loc=(side * 0.4, 0, 2.0), scale=(1.5, 1.5, 1.5), interp="CONSTANT")
            t += 0.55


# ---------------------------------------------------------------- bench + scope
def make_bench_scope(idx, center, vis, screen_mat, w=2.8, d=1.1):
    """Lab bench (top, legs, shelf) with a bench oscilloscope and clutter. center = scene-local (x, y).
    Returns the screen plane object. The screen faces -Y."""
    cx, cy = center
    X, Y = OX(idx) + cx, cy
    top_z = 0.9
    objs = [box("bench_top", (X, Y, top_z), (w, d, 0.08), (0.62, 0.45, 0.28)),
            box("bench_shelf", (X, Y, 0.35), (w - 0.1, d - 0.1, 0.05), (0.5, 0.38, 0.25)),
            box("bench_back", (X, Y + d / 2 - 0.02, 1.1), (w, 0.04, 0.5), (0.45, 0.45, 0.5))]
    for sx in (-1, 1):
        for sy in (-1, 1):
            objs.append(box("bench_leg", (X + sx * (w / 2 - 0.06), Y + sy * (d / 2 - 0.06), top_z / 2), (0.08, 0.08, top_z), (0.3, 0.3, 0.34)))
    # scope body, screen, knobs, handle, probe
    sz = top_z + 0.04 + 0.36
    objs.append(box("scope_body", (X, Y, sz), (1.15, 0.55, 0.72), (0.42, 0.42, 0.46)))
    scr = plane("scope_screen", (X - 0.12, Y - 0.282, sz + 0.06), (0.84, 0.525), screen_mat)
    objs.append(scr)
    objs.append(box("scope_bezel", (X - 0.12, Y - 0.276, sz + 0.06), (0.92, 0.01, 0.6), (0.12, 0.12, 0.14)))
    for j in range(4):
        objs.append(cyl("knob", (X + 0.45, Y - 0.29, sz + 0.25 - 0.14 * j), 0.045, 0.04, (0.8, 0.8, 0.85), axis="Y"))
    objs.append(box("scope_handle", (X, Y, sz + 0.4), (0.7, 0.06, 0.05), (0.2, 0.2, 0.22)))
    objs.append(cyl("probe_cable", (X + 0.7, Y - 0.35, top_z + 0.06), 0.015, 0.9, (0.1, 0.1, 0.1), axis="X"))
    objs.append(cyl("mug", (X - 1.05, Y - 0.2, top_z + 0.1), 0.07, 0.14, (0.9, 0.9, 0.9)))
    objs.append(box("dmm", (X + 1.0, Y - 0.1, top_z + 0.09), (0.28, 0.2, 0.14), (0.9, 0.7, 0.1)))
    objs.append(cyl("spool", (X - 0.85, Y + 0.2, top_z + 0.08), 0.1, 0.1, (0.2, 0.5, 0.9)))
    for o in objs:
        V(o, idx, *vis)
    return scr
