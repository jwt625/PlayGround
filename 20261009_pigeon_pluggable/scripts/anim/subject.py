"""The Subject (monitored human) = Gary: Gary v2 rig, exact outfit + motion_v2 actions (copied from the interconnect
film); optional coffee cup prop; an unused hoodie variant kept behind outfit="hoodie". Blender 4.2, headless-safe.

Character space (from the Gary v2 asset): ROOT_<id> at the floor between the feet, Z up, FRONT = -Y, character left = +X.
World yaw convention used here: yaw 0 = facing world -Y; facing a direction d: yaw = atan2(d.x, -d.y) (same as
build_crude_film.yaw_face / asm.yaw_to).

NLA layout on the armature (bottom -> top): base (idle, held) < loco (walk strips, time-mapped) < act (startle, flinch,
...) < arm (cup/hand overlays, REPLACE on the right-arm IK bones) < look (head/neck/chest/eye overlays, ADD).
All functions take absolute scene frames.
"""
import math
import os

import bpy
from mathutils import Euler, Matrix, Vector

try:
    from . import keys as KY
except ImportError:
    import keys as KY

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(HERE))
CHAR_DIR = os.path.join(ROOT_DIR, "assets", "components", "characters")
GARY_BLEND = os.path.join(CHAR_DIR, "gary_v2.blend")
MOTION_BLEND = os.path.join(CHAR_DIR, "motion_v2", "gary_v2_motion_v2.blend")
FPS = 30
D2R = math.pi / 180.0

HIDE_PARTS = ("hardhat", "bib", "bib_buckles", "straps", "pocket", "pen", "buttons", "buckle", "pouches", "pliers_handles",
              "pliers_jaw", "cable_tie_roll", "belt")
OUTFIT = {   # base colours (linear RGB) for the hoodie variant (design choices)
    "hoodie": (0.16, 0.17, 0.19),       # heather charcoal
    "jeans": (0.035, 0.05, 0.10),       # dark denim
    "sneaker": (0.62, 0.62, 0.60),      # off-white
    "sole": (0.80, 0.80, 0.78),
    "string": (0.75, 0.75, 0.72),
    "cup": (0.85, 0.83, 0.78),
    "cup_sleeve": (0.30, 0.17, 0.08),
    "cup_lid": (0.05, 0.05, 0.05),
}
TRACKS = ("base", "loco", "act", "arm", "look")


class Subject:
    """root (ROOT_gary_v2 object), arm (armature object), coll, aid (asset id), cup (object or None), objs, look_n."""
    pass


# ================================================================================================ build
def _mat_copy(src, name, color):
    m = src.copy()
    m.name = name
    _set_color(m, color)
    return m


def _set_color(m, color):
    if m is None or not m.use_nodes:
        return
    for n in m.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            inp = n.inputs["Base Color"]
            if not inp.is_linked:
                inp.default_value = (*color, 1.0)
    m.diffuse_color = (*color, 1.0)


def _bone_parent(o, arm, bone):
    mw = o.matrix_world.copy()
    o.parent = arm
    o.parent_type = "BONE"
    o.parent_bone = bone
    bpy.context.view_layer.update()
    o.matrix_world = mw


def _tube(name, p0, p1, r0, r1, mat, coll, segs=16, cap=True):
    """Tapered tube between two points (world/armature rest coordinates)."""
    import bmesh
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=cap, segments=segs, radius1=r0, radius2=r1, depth=d.length)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    rot = d.normalized().to_track_quat("Z", "Y")
    o.matrix_world = Matrix.Translation((p0 + p1) / 2) @ rot.to_matrix().to_4x4()
    return o


def _ball(name, c, r, mat, coll):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=r)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.matrix_world = Matrix.Translation(c)
    return o


def _torus(name, c, R, r, mat, coll, scale=(1, 1, 1), rot=(0, 0, 0)):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    segs, rings = 32, 12
    verts = []
    for i in range(segs):
        a = 2 * math.pi * i / segs
        row = []
        for j in range(rings):
            b = 2 * math.pi * j / rings
            x = (R + r * math.cos(b)) * math.cos(a)
            y = (R + r * math.cos(b)) * math.sin(a)
            z = r * math.sin(b)
            row.append(bm.verts.new((x, y, z)))
        verts.append(row)
    for i in range(segs):
        for j in range(rings):
            bm.faces.new((verts[i][j], verts[(i + 1) % segs][j], verts[(i + 1) % segs][(j + 1) % rings], verts[i][(j + 1) % rings]))
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.matrix_world = Matrix.Translation(c) @ Euler(rot).to_matrix().to_4x4() @ Matrix.Diagonal((*scale, 1.0))
    return o


def _bone_pts(arm, b):
    bb = arm.data.bones[b]
    return arm.matrix_world @ bb.head_local, arm.matrix_world @ bb.tail_local


def build_hoodie(s, colors=None):
    """Casual film look: hide hard hat, bib, straps, tool belt and tools; shirt -> hoodie colour; long sleeves (bone-parented
    tubes); hood rolled behind the neck; drawstrings; overalls legs -> jeans; boots -> sneakers."""
    col = dict(OUTFIT, **(colors or {}))
    aid = s.aid
    by = {o.name.replace(aid + "_", ""): o for o in s.objs}
    for k in HIDE_PARTS:
        o = by.get(k)
        if o is not None:
            o.hide_render = o.hide_viewport = True
    M = lambda k: bpy.data.materials.get("MAT_characters_%s_%s" % (aid, k))   # noqa: E731
    _set_color(M("shirt"), col["hoodie"])
    _set_color(M("overalls"), col["jeans"])
    _set_color(M("boots"), col["sneaker"])
    _set_color(M("sole"), col["sole"])
    hood = M("shirt")
    strm = _mat_copy(hood, "MAT_subject_string", col["string"])
    arm, coll = s.arm, s.coll
    bpy.context.view_layer.update()
    made = []
    for side in ("L", "R"):
        sg = 1 if side == "L" else -1
        h, t = _bone_pts(arm, "upper_arm_" + side)
        made.append((_tube("subj_sleeve_up_" + side, h + Vector((0, 0, 0.01)), t + (t - h).normalized() * 0.02, 0.074, 0.067, hood, coll),
                     "upper_arm_" + side))
        h2, t2 = _bone_pts(arm, "forearm_" + side)
        made.append((_tube("subj_sleeve_lo_" + side, h2 - (t2 - h2).normalized() * 0.02, t2 + (t2 - h2).normalized() * 0.01, 0.067, 0.062,
                           hood, coll), "forearm_" + side))
        made.append((_ball("subj_elbow_" + side, h2, 0.069, hood, coll), "forearm_" + side))
        made.append((_torus("subj_cuff_" + side, t2 + (t2 - h2).normalized() * 0.006, 0.056, 0.012, hood, coll,
                            rot=tuple((t2 - h2).normalized().to_track_quat("Z", "Y").to_euler())), "forearm_" + side))
        _ = sg
    # hood: thick rolled collar sitting behind and around the neck
    made.append((_torus("subj_hood", Vector((0.0, 0.035, 1.37)), 0.135, 0.05, hood, coll, scale=(1.0, 0.85, 1.0),
                        rot=(-14 * D2R, 0, 0)), "chest"))
    made.append((_tube("subj_hood_back", Vector((0.0, 0.15, 1.42)), Vector((0.0, 0.17, 1.28)), 0.10, 0.06, hood, coll), "chest"))
    torso, front = hoodie_torso(s, hood)
    for sx in (-1, 1):
        y0 = front(1.33, sx * 0.045) - 0.004
        y1 = front(1.19, sx * 0.05) - 0.006
        made.append((_tube("subj_string_%d" % (sx + 1), Vector((sx * 0.045, y0, 1.33)), Vector((sx * 0.05, y1, 1.19)),
                           0.006, 0.006, strm, coll, segs=8), "chest"))
    for o, b in made:
        _bone_parent(o, arm, b)
        s.objs.append(o)
    s.objs.append(torso)
    s.outfit = [o for o, _ in made] + [torso]
    return s.outfit


TORSO_BONES = ((0.90, "hips"), (1.02, "spine_1"), (1.145, "spine_2"), (1.28, "chest"))


def hoodie_torso(s, mat, z0=0.855, z1=1.41, dz=0.02, n_ang=36, margin=0.014, sources=("overalls", "shirt", "body", "creases")):
    """Hoodie body: a lofted shell fitted around the torso (max radial extent of the overalls/shirt/body vertices per height
    and angle, plus margin), hem below the waist, skinned to hips/spine/chest. Returns (object, front(z, x) -> y)."""
    import bmesh
    aid = s.aid
    skip = ("upper_arm", "forearm", "hand", "thigh", "shin", "foot", "neck", "head", "clavicle", "index", "middle", "ring",
            "pinky", "thumb", "jaw", "eye", "lid")
    pts = []
    for nm in sources:
        o = bpy.data.objects.get("%s_%s" % (aid, nm))
        if o is None:
            continue
        vg = {g.index: g.name for g in o.vertex_groups}
        for v in o.data.vertices:
            if v.co.z < z0 - 0.03 or v.co.z > z1 + 0.03:
                continue
            bad = sum(g.weight for g in v.groups if vg.get(g.group, "").startswith(skip) and not vg.get(g.group, "").startswith("thigh"))
            if nm != "overalls" and bad > 0.3:
                continue
            if nm == "overalls" and v.co.z > 0.86 and abs(v.co.x) > 0.24:
                continue
            pts.append(v.co.copy())
    nz = int(round((z1 - z0) / dz)) + 1
    zs = [z0 + i * dz for i in range(nz)]
    yc = []
    for z in zs:
        sel = [p for p in pts if abs(p.z - z) < dz]
        yc.append((min(p.y for p in sel) + max(p.y for p in sel)) / 2 if sel else 0.0)
    R = [[0.0] * n_ang for _ in zs]
    for iz, z in enumerate(zs):
        for p in pts:
            if abs(p.z - z) > 0.025:
                continue
            a = math.atan2(p.y - yc[iz], p.x)
            ia = int(round(a / (2 * math.pi) * n_ang)) % n_ang
            r = math.hypot(p.x, p.y - yc[iz])
            for k in (-1, 0, 1):
                j = (ia + k) % n_ang
                R[iz][j] = max(R[iz][j], r * (1.0 if k == 0 else 0.985))
    for row in R:                       # fill empty bins and smooth around the ring
        for j in range(n_ang):
            if row[j] == 0.0:
                row[j] = max(row[(j - 1) % n_ang], row[(j + 1) % n_ang])
    for _ in range(2):
        R = [[(row[(j - 1) % n_ang] + 2 * row[j] + row[(j + 1) % n_ang]) / 4 for j in range(n_ang)] for row in R]
    for _ in range(1):
        R = [[(R[max(0, i - 1)][j] + 2 * R[i][j] + R[min(nz - 1, i + 1)][j]) / 4 for j in range(n_ang)] for i in range(nz)]
    R = [[max(r, 0.08) + margin for r in row] for row in R]
    # monotone hem: below the widest hip ring the shell drops straight (no tuck-in)
    me = bpy.data.meshes.new("subj_hoodie_torso")
    bm = bmesh.new()
    rings = []
    for iz, z in enumerate(zs):
        k = 1.0
        if z > z1 - 0.07:              # round the shoulders in toward the neck
            k = 1.0 - 0.45 * ((z - (z1 - 0.07)) / 0.07) ** 1.5
        ring = [bm.verts.new((R[iz][j] * k * math.cos(2 * math.pi * j / n_ang),
                              yc[iz] + R[iz][j] * k * math.sin(2 * math.pi * j / n_ang), z)) for j in range(n_ang)]
        rings.append(ring)
    hem = [bm.verts.new((v.co.x * 0.93, yc[0] + (v.co.y - yc[0]) * 0.93, z0 + 0.012)) for v in rings[0]]
    rings.insert(0, hem)
    for a, b in zip(rings[:-1], rings[1:]):
        for j in range(n_ang):
            bm.faces.new((a[j], a[(j + 1) % n_ang], b[(j + 1) % n_ang], b[j]))
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    for pl in me.polygons:
        pl.use_smooth = True
    o = bpy.data.objects.new("subj_hoodie_torso", me)
    s.coll.objects.link(o)
    o.parent = s.root
    gs = {nm: o.vertex_groups.new(name=nm) for _, nm in TORSO_BONES}
    for v in me.vertices:
        z = v.co.z
        for (za, na), (zb, nb) in zip(TORSO_BONES[:-1], TORSO_BONES[1:]):
            if z <= TORSO_BONES[0][0]:
                gs["hips"].add([v.index], 1.0, "REPLACE")
                break
            if za <= z <= zb:
                u = (z - za) / (zb - za)
                gs[na].add([v.index], 1 - u, "REPLACE")
                gs[nb].add([v.index], u, "REPLACE")
                break
        else:
            gs["chest"].add([v.index], 1.0, "REPLACE")
    md = o.modifiers.new("arm", "ARMATURE")
    md.object = s.arm
    sub = o.modifiers.new("sub", "SUBSURF")
    sub.levels = sub.render_levels = 1

    def front(z, x):
        iz = max(0, min(nz - 1, int(round((z - z0) / dz))))
        best = None
        for j in range(n_ang):
            a = 2 * math.pi * j / n_ang
            px, py = R[iz][j] * math.cos(a), yc[iz] + R[iz][j] * math.sin(a)
            if py < yc[iz] and (best is None or abs(px - x) < abs(best[0] - x)):
                best = (px, py)
        return best[1]
    return o, front


def make_cup(s, colors=None, hand="R"):
    """Coffee cup (tapered paper cup, sleeve, lid) parented to HOOK_hand_<hand>; axis opposite the finger direction,
    held in front of the palm. Returns the cup root object."""
    col = dict(OUTFIT, **(colors or {}))
    import bmesh
    hook = next(o for o in s.objs if o.name.startswith("HOOK_hand_" + hand))
    base = bpy.data.materials.get("MAT_characters_%s_shirt" % s.aid)
    mc = _mat_copy(base, "MAT_subject_cup", col["cup"])
    ms = _mat_copy(base, "MAT_subject_cup_sleeve", col["cup_sleeve"])
    ml = _mat_copy(base, "MAT_subject_cup_lid", col["cup_lid"])
    root = bpy.data.objects.new("subj_cup", None)
    root.empty_display_size = 0.05
    s.coll.objects.link(root)
    parts = [("subj_cup_body", 0.032, 0.043, 0.0, 0.135, mc), ("subj_cup_sleeve", 0.0385, 0.0425, 0.045, 0.105, ms),
             ("subj_cup_lid", 0.045, 0.040, 0.135, 0.152, ml)]
    for nm, r0, r1, z0, z1, m in parts:
        me = bpy.data.meshes.new(nm)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=20, radius1=r0, radius2=r1, depth=z1 - z0,
                              matrix=Matrix.Translation((0, 0, (z0 + z1) / 2)))
        bm.to_mesh(me)
        bm.free()
        me.materials.append(m)
        for p in me.polygons:
            p.use_smooth = True
        o = bpy.data.objects.new(nm, me)
        s.coll.objects.link(o)
        o.parent = root
        s.objs.append(o)
    # hook local frame: +Y = finger direction (down at rest), +Z = palm normal (toward the body), +X = forward
    root.parent = hook
    root.matrix_parent_inverse.identity()
    # cup axis (+Z of the cup) = -hook Y; cup centre in front of the palm (+hook X) and a bit below the knuckles
    m = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))).transposed()  # columns: cup X,Y,Z in hook space
    root.matrix_basis = Matrix.Translation((0.058, 0.035, 0.012)) @ m.to_4x4() @ Matrix.Translation((0, 0, -0.07))
    s.cup = root
    s.objs.append(root)
    return root


def append_subject(name="SUBJECT", outfit="gary", cup=True, coll=None, colors=None):
    """Append Gary v2 (collection ASSET_gary_v2) into the scene and create the NLA tracks. The Subject IS Gary: outfit
    "gary" (default, owner 2026-10-10) keeps the exact gary_v2 outfit and materials (hard hat, overalls, bib, tool belt,
    tools); outfit "hoodie" = the earlier casual variant (not used). cup: coffee cup prop on the right hand (hide it with
    cup_show while the right hand holds the bird). Returns a Subject."""
    with bpy.data.libraries.load(GARY_BLEND, link=False) as (src, dst):
        dst.collections = [c for c in src.collections if c.startswith("ASSET_")][:1]
    c = dst.collections[0]
    (coll or bpy.context.scene.collection).children.link(c)
    s = Subject()
    s.coll = c
    s.objs = list(c.all_objects)
    s.root = next(o for o in s.objs if o.name.startswith("ROOT_"))
    s.arm = next(o for o in s.objs if o.type == "ARMATURE")
    s.aid = s.root["asset_id"]
    s.cup = None
    s.look_n = 0
    s.name = name
    s.root.rotation_mode = "XYZ"
    bpy.context.view_layer.update()
    if outfit == "hoodie":
        build_hoodie(s, colors)
    if cup:
        make_cup(s, colors)
    ad = s.arm.animation_data_create()
    ad.action = None
    s.tracks = {}
    for tn in TRACKS:
        tr = ad.nla_tracks.new()
        tr.name = "subj_" + tn
        s.tracks[tn] = tr
    return s


# ================================================================================================ actions / NLA
def action(s, short):
    """bpy Action ACT_<aid>_v2_<short> (appended from the motion_v2 library on first use, with its _face action)."""
    full = "ACT_%s_v2_%s" % (s.aid, short)
    if full not in bpy.data.actions:
        with bpy.data.libraries.load(MOTION_BLEND, link=False) as (src, dst):
            dst.actions = [n for n in (full, full + "_face") if n in src.actions]
        for a in dst.actions:
            a.use_fake_user = True
    return bpy.data.actions[full]


def _track_for(s, tn, f0, f1):
    """Track `tn`, or a new track right above it when [f0, f1] overlaps an existing strip."""
    tr = s.tracks[tn]
    for t in [tr] + getattr(s, "_extra", {}).get(tn, []):
        if all(st.frame_end < f0 or st.frame_start > f1 for st in t.strips):
            return t
    ad = s.arm.animation_data
    t = ad.nla_tracks.new(prev=([tr] + s._extra.get(tn, []))[-1] if hasattr(s, "_extra") else tr)
    t.name = "subj_%s_%d" % (tn, len(getattr(s, "_extra", {}).get(tn, [])) + 1)
    s.__dict__.setdefault("_extra", {}).setdefault(tn, []).append(t)
    return t


def strip(s, tn, act, f0, repeat=1.0, speed=1.0, hold=False, blend="REPLACE", blend_in=0, blend_out=0, influence=None,
          start_frame=None, end_frame=None, face=True):
    """Place an action strip on track tn at frame f0. Returns the strip."""
    a = act if isinstance(act, bpy.types.Action) else action(s, act)
    lenf = (a.frame_range[1] - a.frame_range[0]) * repeat / speed
    tr = _track_for(s, tn, f0, f0 + lenf)
    st = tr.strips.new(a.name, int(f0), a)
    if start_frame is not None:
        st.action_frame_start = float(start_frame)
    if end_frame is not None:
        st.action_frame_end = float(end_frame)
    st.scale = 1.0 / speed
    st.repeat = repeat
    st.blend_type = blend
    st.extrapolation = "HOLD_FORWARD" if hold else "NOTHING"
    st.blend_in, st.blend_out = blend_in, blend_out
    if influence is not None:
        st.use_animated_influence = True
        st.influence = influence
    fa = bpy.data.actions.get(a.name + "_face")
    if face and fa is not None:
        rad = s.root.animation_data_create()
        ftr = rad.nla_tracks.new()
        ftr.name = "subj_face_%d" % len(rad.nla_tracks)
        fs = ftr.strips.new(fa.name, int(f0), fa)
        fs.scale, fs.repeat = st.scale, st.repeat
        if start_frame is not None:
            fs.action_frame_start = float(start_frame)
        if end_frame is not None:
            fs.action_frame_end = float(end_frame)
        fs.extrapolation = "NOTHING"
    return st


def idle(s, f0, f1, name="idle_breathe"):
    """Idle on the base track over [f0, f1] (looping, held forward)."""
    a = action(s, name)
    n = (a.frame_range[1] - a.frame_range[0])
    return strip(s, "base", a, f0, repeat=max(1.0, (f1 - f0) / n), hold=True)


# ================================================================================================ placement / walking
def place(s, f, pos, yaw, interp="LINEAR"):
    """Key the Subject root at world pos (z = floor) with yaw (0 = facing world -Y)."""
    KY.key(s.root, "location", f, (pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0), interp)
    KY.key(s.root, "rotation_euler", f, (0.0, 0.0, yaw), interp)


def yaw_face(src, dst):
    return math.atan2(dst[0] - src[0], -(dst[1] - src[1]))


def _unwrap(prev, a):
    while a - prev > math.pi:
        a -= 2 * math.pi
    while a - prev < -math.pi:
        a += 2 * math.pi
    return a


def walk_path(s, f0, pts, gait="walk", speed=None, ease_in=0.35, ease_out=0.5, end_yaw=None, blend_out=8, step=2):
    """Walk the root along world points (Catmull-Rom through pts, z ignored) starting at frame f0.

    Root-motion safe: the gait strip's action time is driven by the travelled distance (strip_time = distance / baked
    root speed x 30, cyclic), so the feet stay locked through the ease-in / ease-out; the end is nudged (<= 6 % stride)
    to stop at a passing pose (feet together) and the strip blends out into the base idle. Returns the end frame."""
    try:
        from . import pgeom as G
    except ImportError:
        import pgeom as G
    a = action(s, gait)
    v_bake = float(a.get("root_speed_mps", 0.917))
    cyc = a.frame_range[1] - a.frame_range[0]
    v = speed or v_bake
    poly = G.catmull([(p[0], p[1], 0.0) for p in pts], 16)
    S = G.arclen(poly)
    L = S[-1]
    # speed profile: ease in / constant / ease out; distance(t) by numeric integration at 1/30 s
    T_c = max(0.0, (L - v * (ease_in + ease_out) / 2) / v)
    T = ease_in + T_c + ease_out

    def vel(t):
        if t < ease_in:
            return v * math.sin(0.5 * math.pi * t / ease_in) ** 2
        if t > ease_in + T_c:
            u = (T - t) / ease_out
            return v * math.sin(0.5 * math.pi * max(0.0, u)) ** 2
        return v
    nF = int(math.ceil(T * FPS))
    dist = [0.0]
    for i in range(nF):
        dist.append(dist[-1] + 0.5 * (vel(i / FPS) + vel((i + 1) / FPS)) / FPS)
    k = L / dist[-1]
    dist = [d * k for d in dist]
    # gait phase: stop on a passing pose (phase = cyc/4 or 3cyc/4)
    ph_end = dist[-1] / v_bake * FPS
    q = cyc / 2.0
    target = math.floor((ph_end - cyc / 4) / q + 0.5) * q + cyc / 4
    ps = target / ph_end if ph_end > 0 else 1.0
    yaw_prev = KY.value(s.root, "rotation_euler", 2, f0, s.root.rotation_euler[2])
    for i in range(nF + 1):
        if i % step and i != nF:
            continue
        p, d = G.sample_at(poly, S, dist[i])
        yaw = _unwrap(yaw_prev, math.atan2(d[0], -d[1]))
        yaw_prev = yaw
        place(s, f0 + i, p, yaw)
    if end_yaw is not None:
        ye = _unwrap(yaw_prev, end_yaw)
        place(s, f0 + nF + 12, poly[-1], ye, "BEZIER")
    st = strip(s, "loco", a, f0, repeat=max(1.0, (nF + 1) / cyc), blend_out=blend_out)
    st.use_animated_time = True
    st.use_animated_time_cyclic = True
    for i in range(0, nF + 1, step):
        st.strip_time = dist[i] / v_bake * FPS * ps
        st.keyframe_insert("strip_time", frame=f0 + i)
    st.strip_time = dist[-1] / v_bake * FPS * ps
    st.keyframe_insert("strip_time", frame=f0 + nF)
    _linear_strip_curves(s.arm, st)
    s.__dict__.setdefault("walks", []).append((f0, f0 + nF))
    return f0 + nF


def _linear_strip_curves(arm, st):
    for fc in st.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def turn(s, f0, side="R", deg=90):
    """Turn in place with the motion_v2 turn action (feet step; root bone rotates), then hand the yaw over to the root
    object at the strip end (CONSTANT key), as motion_v2.apply does. side 'L' = toward the character's left (+yaw).
    Returns the end frame."""
    name = "turn_%s_%d" % ("left" if side == "L" else "right", deg)
    a = action(s, name)
    st = strip(s, "act", a, f0)
    fe = int(round(st.frame_end))
    dy = float(a.get("root_yaw_delta_rad", (1 if side == "L" else -1) * deg * D2R))
    y0 = KY.value(s.root, "rotation_euler", 2, f0, s.root.rotation_euler[2])
    loc = KY.values(s.root, "location", f0)
    place(s, fe, loc, y0, "CONSTANT")
    place(s, fe + 1, loc, y0 + dy, "CONSTANT")
    return fe + 1


def stop(s, f, name="idle_breathe", f1=None):
    """Ensure the idle base covers f..f1 (walk strips blend out into it)."""
    return idle(s, f, f1 or f + 600, name)


# ================================================================================================ overlays: look, glance
def _root_matrix(s, f):
    loc = KY.values(s.root, "location", f)
    rot = KY.values(s.root, "rotation_euler", f)
    return Matrix.Translation(loc) @ Euler(rot).to_matrix().to_4x4()


HEAD_C = Vector((0.0, -0.03, 1.52))   # head centre, armature rest space (eyes at z 1.514)
LOOK_REST = Vector((0.0, -0.869, 1.574))


def _look_action(s, keys_):
    """keys_: list of (frame_rel, dict bone -> (rx, ry, rz)) euler deltas for XYZ bones, plus 'ctl_look' -> (dx,dy,dz) in
    root-bone local space. Returns a new action."""
    s.look_n += 1
    a = bpy.data.actions.new("SUBJ_look_%03d" % s.look_n)
    a.use_fake_user = True
    chans = {}
    for fr, d in keys_:
        for b, v in d.items():
            path = 'pose.bones["%s"].%s' % (b, "location" if b == "ctl_look" else "rotation_euler")
            for i in range(3):
                chans.setdefault((path, i, b), []).append((fr, v[i]))
    for (path, i, b), pts in chans.items():
        fc = a.fcurves.new(path, index=i, action_group=b)
        fc.keyframe_points.add(len(pts))
        for kp, (fr, val) in zip(fc.keyframe_points, pts):
            kp.co = (fr, val)
            kp.interpolation = "BEZIER"
        fc.update()
    return a


def _look_delta(s, f, target, w_chest=0.0, max_yaw=80 * D2R, max_pitch=35 * D2R):
    """Yaw/pitch (radians) from the facing direction to a world target at frame f (character space), clamped."""
    M = _root_matrix(s, f)
    tl = M.inverted() @ Vector(target)
    d = tl - HEAD_C
    yaw = math.atan2(d.x, -d.y)                 # + = toward character left (+X)
    pitch = math.atan2(d.z, math.hypot(d.x, d.y))  # + = up
    return max(-max_yaw, min(max_yaw, yaw)), max(-max_pitch, min(max_pitch, pitch)), d.normalized()


def _pose_keys(yaw, pitch, dirv, w_chest, w_neck=0.35):
    w_head = 1.0 - w_neck - w_chest
    out = {"neck": (-pitch * 0.35, yaw * w_neck, 0.0), "head": (-pitch * 0.65, yaw * w_head, 0.0)}
    if w_chest:
        out["chest"] = (0.0, yaw * w_chest, 0.0)
    # eyes: move ctl_look onto the gaze ray (root-bone local axes: x = X, y = Z, z = -Y)
    p = HEAD_C + dirv * 0.87
    dl = p - LOOK_REST
    out["ctl_look"] = (dl.x, dl.z, -dl.y)
    return out


def look_at(s, f0, f1, target, ramp=6, w_chest=0.0, track=True, step=6, max_yaw=80 * D2R, max_pitch=35 * D2R):
    """Head/neck (and optionally chest) turn plus eyes toward a world target (point or callable frame -> point) over
    [f0, f1] as an ADD overlay; eases in/out over `ramp` frames. Returns f1."""
    tf = target if callable(target) else (lambda f: target)
    kk = [(0.0, _pose_keys(0, 0, Vector((0, -1, 0)), w_chest))]
    kk[0][1]["ctl_look"] = (0.0, 0.0, 0.0)
    f = f0 + ramp
    while f <= f1 - ramp:
        y, p, dv = _look_delta(s, f, tf(f), max_yaw=max_yaw, max_pitch=max_pitch)
        kk.append((f - f0, _pose_keys(y, p, dv, w_chest)))
        if not track:
            kk.append((f1 - ramp - f0, _pose_keys(y, p, dv, w_chest)))
            break
        f += step
    z = _pose_keys(0, 0, Vector((0, -1, 0)), w_chest)
    z["ctl_look"] = (0.0, 0.0, 0.0)
    kk.append((f1 - f0, z))
    a = _look_action(s, kk)
    strip(s, "look", a, f0, blend="ADD", face=False)
    return f1


def glance_over_shoulder(s, f0, side="L", hold=18, ramp=7):
    """Quick look back over a shoulder (chest 25 deg, neck + head 85 deg, eyes lead), hold, return. Returns end frame."""
    sg = 1.0 if side == "L" else -1.0
    yaw = sg * 110 * D2R
    dv = Vector((math.sin(yaw), -math.cos(yaw), -0.1)).normalized()
    pk = {"chest": (0.0, sg * 25 * D2R, 0.0), "neck": (0.05, sg * 35 * D2R, 0.0), "head": (0.08, sg * 50 * D2R, 0.0)}
    p = HEAD_C + dv * 0.87
    dl = p - LOOK_REST
    pk["ctl_look"] = (dl.x, dl.z, -dl.y)
    zero = {k: (0.0, 0.0, 0.0) for k in pk}
    a = _look_action(s, [(0, zero), (ramp - 2, pk), (ramp + hold, pk), (2 * ramp + hold, zero)])
    strip(s, "look", a, f0, blend="ADD", face=False)
    return f0 + 2 * ramp + hold


# ================================================================================================ full-body acts
def startle(s, f0, then_freeze=0):
    """Startle hop (motion_v2 'startle', face shock); optional freeze afterwards (frames). Returns end frame."""
    st = strip(s, "act", "startle", f0, blend_in=2, blend_out=6 if not then_freeze else 0)
    fe = int(st.frame_end)
    if then_freeze:
        return freeze(s, fe, fe + then_freeze, pose_from=("startle", 30))
    return fe


def freeze(s, f0, f1, pose_from=("flinch_upper", 8)):
    """Hold a tense pose over [f0, f1] (a single action frame held: shoulders up, still)."""
    name, fr = pose_from
    st = strip(s, "act", name, f0, start_frame=fr, end_frame=fr + 1, blend_in=3, blend_out=6)
    st.repeat = 1.0
    st.scale = max(1.0, (f1 - f0))
    return f1


def gesture(s, f0, name, **kw):
    """Any motion_v2 action by short name on the act track (e.g. 'flinch_upper', 'shrug_upper', 'point')."""
    st = strip(s, "act", name, f0, **dict(dict(blend_in=3, blend_out=6), **kw))
    return int(st.frame_end)


# ================================================================================================ right-arm overlays
def _ik_basis(s, bone, M_arm):
    """Pose-bone basis that puts `bone` at armature-space matrix M_arm, assuming its parent (chest) at rest."""
    b = s.arm.data.bones[bone]
    return b.matrix_local.inverted() @ M_arm


def _hand_target(s, pos_arm, x_dir, y_dir, side="R"):
    """Armature-space matrix of the hand bone so that HOOK_hand_<side> lands at pos_arm with hook axes x_dir and y_dir
    (finger direction); z = x cross y is the palm normal."""
    hook = next(o for o in s.objs if o.name.startswith("HOOK_hand_" + side))
    b = s.arm.data.bones["hand_" + side]
    # bone-parented child: world = arm @ pose_bone.matrix @ T(0, length, 0) @ parent_inverse @ basis (pose independent)
    H_rel = Matrix.Translation((0.0, b.length, 0.0)) @ hook.matrix_parent_inverse @ hook.matrix_basis
    x = Vector(x_dir).normalized()
    y = (Vector(y_dir) - x * Vector(y_dir).dot(x)).normalized()
    z = x.cross(y)
    Mh = Matrix((x, y, z)).transposed().to_4x4()
    Mh.translation = Vector(pos_arm)
    return Mh @ H_rel.inverted()


def _arm_action(s, name, keys_, side="R"):
    """keys_: list of (frame_rel, hand matrix (armature space) or None = rest pose, elbow pole pos or None); side R / L."""
    ikh, ike = "ik_hand_" + side, "ik_elbow_" + side
    a = bpy.data.actions.new(name)
    a.use_fake_user = True
    chans = {}
    for fr, M, pole in keys_:
        if M is None:
            loc, q = Vector(), (1.0, 0.0, 0.0, 0.0)
        else:
            B = _ik_basis(s, ikh, M)
            loc, q = B.to_translation(), tuple(B.to_quaternion())
        for i in range(3):
            chans.setdefault(('pose.bones["%s"].location' % ikh, i), []).append((fr, loc[i]))
        for i in range(4):
            chans.setdefault(('pose.bones["%s"].rotation_quaternion' % ikh, i), []).append((fr, q[i]))
        pl = Vector() if pole is None else (s.arm.data.bones[ike].matrix_local.inverted() @ Matrix.Translation(pole)).to_translation()
        for i in range(3):
            chans.setdefault(('pose.bones["%s"].location' % ike, i), []).append((fr, pl[i]))
    for (path, i), pts in chans.items():
        fc = a.fcurves.new(path, index=i, action_group="arm")
        fc.keyframe_points.add(len(pts))
        for kp, (fr, val) in zip(fc.keyframe_points, pts):
            kp.co = (fr, val)
            kp.interpolation = "BEZIER"
        fc.update()
    return a


def sip_coffee(s, f0, raise_frames=14, hold=22, lower_frames=14, from_pose=None, from_pole=None):
    """Raise the cup to the mouth (cup tilted toward the face), hold, lower; head tips back slightly. Overlay on the
    right-arm IK bones (REPLACE, blended in/out). Returns the end frame."""
    top = Vector((0.0, -0.178, 1.432))            # cup top centre target (measured: IK lands ~2 cm low; rim then meets the lips)
    up = Vector((0.0, 0.5, 0.866))                # cup axis tilted toward the face
    x_dir = Vector((1.0, 0.0, 0.0))               # cup in front of the palm, toward the body centre line
    z_dir = x_dir.cross(-up)
    # make_cup geometry: cup top centre = hook + 0.058 X_h - 0.047 Y_h + 0.012 Z_h, with Y_h = -up
    hook_p = top - x_dir * 0.058 - up * 0.047 - z_dir * 0.012
    M_sip = _hand_target(s, hook_p, x_dir, -up)
    M_mid = _hand_target(s, hook_p + Vector((-0.06, -0.06, -0.22)), Vector((0.5, -0.86, 0)), Vector((0, 0.2, -1)))
    pole = Vector((-0.42, 0.05, 0.85))
    n, lo = raise_frames, lower_frames
    a = _arm_action(s, "SUBJ_sip_%03d" % len(bpy.data.actions),
                    [(0, from_pose, from_pole), (n * 0.55, M_mid, pole), (n, M_sip, pole), (n + hold, M_sip, pole),
                     (n + hold + lo * 0.5, M_mid, pole), (n + hold + lo, from_pose, from_pole)])
    strip(s, "arm", a, f0, blend_in=4, blend_out=4, face=False)
    hk = [(0, {"head": (0.0, 0.0, 0.0), "neck": (0.0, 0.0, 0.0)}), (n, {"head": (-0.12, 0.0, 0.0), "neck": (-0.05, 0.0, 0.0)}),
          (n + hold, {"head": (-0.12, 0.0, 0.0), "neck": (-0.05, 0.0, 0.0)}), (n + hold + lower_frames, {"head": (0.0, 0.0, 0.0), "neck": (0.0, 0.0, 0.0)})]
    strip(s, "look", _look_action(s, hk), f0, blend="ADD", face=False)
    return f0 + n + hold + lower_frames


def reach(s, f0, target, reach_frames=12, hold=20, back_frames=14):
    """Right hand reaches toward a world point (clamped to arm reach), palm down, fingers toward the target; hold; back.
    Returns the end frame."""
    M = _root_matrix(s, f0 + reach_frames)
    t = M.inverted() @ Vector(target)
    sh = Vector((-0.183, 0.0, 1.357))
    d = t - sh
    L = min(d.length, 0.50)
    hp = sh + d.normalized() * L
    fdir = d.normalized()
    xdir = fdir.cross(Vector((0, 0, 1))).normalized() if abs(fdir.z) < 0.95 else Vector((0, -1, 0))
    Mr = _hand_target(s, hp, xdir, fdir)
    pole = sh + Vector((-0.3, 0.25, -0.35))
    a = _arm_action(s, "SUBJ_reach_%03d" % len(bpy.data.actions), [(0, None, None), (reach_frames, Mr, pole),
                                                                    (reach_frames + hold, Mr, pole), (reach_frames + hold + back_frames, None, None)])
    strip(s, "arm", a, f0, blend_in=4, blend_out=6, face=False)
    look_at(s, f0, f0 + reach_frames + hold + back_frames, target, ramp=6, track=False)
    return f0 + reach_frames + hold + back_frames


# ================================================================================================ queries
def head_world(s, f):
    """World position of the Subject's head centre at frame f (evaluates the rig; use for bird observation targets)."""
    bpy.context.scene.frame_set(int(f))
    pb = s.arm.pose.bones["head"]
    return tuple(s.arm.matrix_world @ (pb.matrix @ Vector((0.0, 0.1, 0.0))))


def head_track(s, f0, f1, step=3):
    """[(frame, xyz)] samples of head_world over [f0, f1] (pass to behaviors.Track)."""
    return [(f, head_world(s, f)) for f in range(int(f0), int(f1) + 1, step)]


# ================================================================================================ v2: hold the bird (DevLog-010 s9 #4)
def palm_frame(s, hand="R"):
    """Palm-surface centre in the HOOK_hand_<hand> frame, measured once from the rest-pose hand vertices of the body mesh
    (hook axes: +Y finger direction, +Z palm normal). Returns Vector (x, y, z)."""
    key = "_palm_" + hand
    if key in s.__dict__:
        return s.__dict__[key]
    body = next(o for o in s.objs if o.name == "%s_body" % s.aid)
    hook = next(o for o in s.objs if o.name.startswith("HOOK_hand_" + hand))
    b = s.arm.data.bones["hand_" + hand]
    M = (b.matrix_local @ Matrix.Translation((0.0, b.length, 0.0)) @ hook.matrix_parent_inverse @ hook.matrix_basis).inverted()
    gi = body.vertex_groups["hand_" + hand].index
    pts = [M @ v.co for v in body.data.vertices if any(g.group == gi and g.weight > 0.5 for g in v.groups)]
    cx = sum(p.x for p in pts) / len(pts)
    cy = sum(p.y for p in pts) / len(pts)
    near = [p for p in pts if abs(p.x - cx) < 0.025 and abs(p.y - cy) < 0.03] or pts
    c = Vector((cx, cy, max(p.z for p in near)))
    s.__dict__[key] = c
    return c


def _palm_hand(s, P, normal, finger, side="R"):
    """Armature-space hand matrix that puts the palm-surface centre at P (armature space) with the given palm normal and
    finger direction."""
    z = Vector(normal).normalized()
    y = (Vector(finger) - z * Vector(finger).dot(z)).normalized()
    x = y.cross(z)
    R = Matrix((x, y, z)).transposed()
    hook_p = Vector(P) - R @ palm_frame(s, side)
    return _hand_target(s, hook_p, x, y, side)


# bird pose in the palm frame (x = finger x normal, y = fingers, z = palm normal); bird frame +X beak, +Z up
_PERCH = Matrix(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))   # beak toward the wrist (faces the Subject), upright
_CROSS = Matrix(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)))  # upright across the palm, beak to his left
_ONBACK = Matrix(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)))  # on its back across the palm (beak to his left), belly/feet up


def hold_bird(s, f0, f1, cell_pos, cell_yaw, rig=None, present_dir=None, end="cell", bird_scale=2.6, lip=0.13,
              look=True, hand="R"):
    """The Subject reaches to a cell, the bird hops onto his palm (feet planted), he raises it to chest height, the bird
    flops onto its back in the palm and the hand tilts toward present_dir (belly and feet up), then back upright; end
    'cell' = lowered to the cell and the bird hops back in (feet on the cell floor), 'hold' = stays perched at chest height.

    cell_pos / cell_yaw: world feet origin and beak yaw of the bird in its cell (R hook: +X = beak).
    rig: optional pigeon_rig.PigeonRig (hero): its root is keyed per frame from the carrier (do not key its root in
    [f0, f1] elsewhere); crouch / wing balance keys are added. present_dir: world vector toward the viewer (camera).
    Returns dict(carrier=<world empty, quaternion keys per frame>, frames={phase: frame}). The carrier transform is the
    bird root (feet origin, +X beak, +Z up, no scale); parent the bird root to it or use rig=."""
    if hand != "R":
        raise NotImplementedError("right hand only (IK overlay bones ik_hand_R / ik_elbow_R)")
    KY.flush()
    # ---- timeline (frames relative to f0)
    d = dict(reach=12, hop=9, raise_=16, flip=10, tilt=10, untilt=10, unflip=10, lower=14, hopback=9, retract=12)
    tail = d["untilt"] + d["unflip"] + (d["lower"] + d["hopback"] + d["retract"] if end == "cell" else 0)
    head = d["reach"] + 2 + d["hop"] + d["raise_"] + 2 + d["flip"] + d["tilt"]
    hold = (f1 - f0) - head - tail
    if hold < 6:
        raise ValueError("hold_bird needs at least %d frames (got %d)" % (head + tail + 6, f1 - f0))
    T = {}
    t = 0
    T["reach"] = t = t + d["reach"]
    T["hop0"] = t = t + 2
    T["on"] = t = t + d["hop"]
    T["raised"] = t = t + d["raise_"]
    T["flip0"] = t = t + 2
    T["back"] = t = t + d["flip"]
    T["present"] = t = t + d["tilt"]
    T["unpresent0"] = t = t + hold
    T["unpresent"] = t = t + d["untilt"]
    T["upright"] = t = t + d["unflip"]
    if end == "cell":
        T["lowered"] = t = t + d["lower"]
        T["in"] = t = t + d["hopback"]
        T["end"] = t = t + d["retract"]
    else:
        T["end"] = t
    # ---- hand poses (character space at the reach frame)
    Mroot = _root_matrix(s, f0 + T["reach"])
    Minv = Mroot.inverted()
    cp = Vector(cell_pos)
    beak = Vector((math.cos(cell_yaw), math.sin(cell_yaw), 0.0))
    P_cell = Minv @ (cp + beak * lip)
    sh = Vector((-0.183, 0.0, 1.357))
    if (P_cell - sh).length > 0.5:
        P_cell = sh + (P_cell - sh).normalized() * 0.5
    fing_cell = (Minv.to_3x3() @ -beak).normalized()                 # fingers point into the cell
    up = Vector((0.0, 0.0, 1.0))
    P_chest = Vector((-0.08, -0.46, 1.08))           # forward of the bib and below the hat brim
    fing_chest = Vector((0.15, -1.0, 0.0))
    if present_dir is not None:
        pdl = (Minv.to_3x3() @ Vector(present_dir)).normalized()
    else:
        pdl = Vector((0.0, -1.0, 0.0))
    n_pres = (up * 0.45 + pdl * 0.55).normalized()
    P_pres = P_chest + Vector((0.03, -0.05, 0.10))
    f_pres = (fing_chest - n_pres * fing_chest.dot(n_pres)).normalized()
    H = {"cell": _palm_hand(s, P_cell, up, fing_cell), "cell_dip": _palm_hand(s, P_cell + Vector((0, 0, -0.015)), up, fing_cell),
         "cell_low": _palm_hand(s, P_cell + (P_chest - P_cell) * 0.25 + Vector((0, 0, -0.07)), up, fing_cell),
         "chest": _palm_hand(s, P_chest, up, fing_chest), "pres": _palm_hand(s, P_pres, n_pres, f_pres)}
    pole = Vector((-0.5, 0.15, 0.9))
    K = [(0, None, None), (T["reach"], H["cell"], pole), (T["on"] - 1, H["cell"], pole), (T["on"] + 2, H["cell_dip"], pole),
         (T["on"] + 6, H["cell_low"], pole), (T["raised"], H["chest"], pole), (T["back"], H["chest"], pole), (T["present"], H["pres"], pole),
         (T["unpresent0"], H["pres"], pole), (T["unpresent"], H["chest"], pole), (T["upright"], H["chest"], pole)]
    if end == "cell":
        K += [(T["lowered"] - 6, H["cell_low"], pole), (T["lowered"], H["cell"], pole), (T["in"] + 2, H["cell"], pole),
              (T["end"], None, None)]
    a = _arm_action(s, "SUBJ_hold_%03d" % len(bpy.data.actions), K)
    st = strip(s, "arm", a, f0, blend_in=4, blend_out=5 if end == "cell" else 0, face=False)
    if end != "cell":
        st.extrapolation = "HOLD_FORWARD"
    KY.flush()
    # ---- bake the carrier (world) from the evaluated hand
    sc = bpy.context.scene
    hook = next(o for o in s.objs if o.name.startswith("HOOK_hand_R"))
    pf = palm_frame(s)
    h_back = 0.066 * bird_scale                                      # back-of-body height when lying on its back
    cell_M = Matrix.Translation(cp) @ Matrix.Rotation(cell_yaw, 4, "Z")

    def palm_world(f):
        sc.frame_set(int(f))
        Mh = hook.matrix_world.copy()
        return Mh @ Matrix.Translation(pf)

    def blend(Ma, Mb, u, lift=0.0):
        u = max(0.0, min(1.0, u))
        e = u * u * (3 - 2 * u)
        q = Ma.to_quaternion().slerp(Mb.to_quaternion(), e)
        p = Ma.to_translation().lerp(Mb.to_translation(), e) + Vector((0, 0, 4 * lift * u * (1 - u)))
        return Matrix.Translation(p) @ q.to_matrix().to_4x4()
    perch4 = _PERCH.to_4x4()
    cross4 = _CROSS.to_4x4()
    back4 = Matrix.Translation((0.0, 0.0, h_back)) @ _ONBACK.to_4x4()
    carrier = bpy.data.objects.new("SUBJ_bird_carrier", None)
    carrier.empty_display_size = 0.1
    s.coll.objects.link(carrier)
    carrier.rotation_mode = "QUATERNION"
    mats = {}
    for i in range(0, T["end"] + 1):
        f = f0 + i
        if i < T["hop0"]:
            M = cell_M
        elif i <= T["on"]:
            M = blend(cell_M, palm_world(f) @ perch4, (i - T["hop0"]) / d["hop"], 0.06)
        elif i < T["flip0"]:                                          # turns on the palm while raised (crosswise)
            M = blend(palm_world(f) @ perch4, palm_world(f) @ cross4, (i - T["on"] - 2) / (T["raised"] - T["on"] - 2))
        elif i <= T["back"]:                                          # rolls onto its back about its own long axis
            Pw = palm_world(f)
            M = blend(Pw @ cross4, Pw @ back4, (i - T["flip0"]) / d["flip"], 0.03)
        elif i < T["unpresent"]:
            M = palm_world(f) @ back4
        elif i <= T["upright"]:
            Pw = palm_world(f)
            M = blend(Pw @ back4, Pw @ cross4, (i - T["unpresent"]) / d["unflip"], 0.03)
        elif end == "cell" and i < T["lowered"]:                      # turns back to face the Subject while lowered
            Pw = palm_world(f)
            M = blend(Pw @ cross4, Pw @ perch4, (i - T["upright"]) / (T["lowered"] - T["upright"] - 2))
        elif end == "cell" and i <= T["in"]:
            M = blend(palm_world(f) @ perch4, cell_M, (i - T["lowered"]) / d["hopback"], 0.05)
        elif end == "cell":
            M = cell_M
        else:
            M = palm_world(f) @ cross4
        mats[f] = M
        KY.key(carrier, "location", f, tuple(M.to_translation()), "LINEAR")
        KY.key(carrier, "rotation_quaternion", f, tuple(M.to_quaternion()), "LINEAR")
    # quaternion sign continuity
    fcs = [carrier.animation_data.action.fcurves.find("rotation_quaternion", index=k) for k in range(4)]
    prev = None
    for j in range(len(fcs[0].keyframe_points)):
        q = [fc.keyframe_points[j].co[1] for fc in fcs]
        if prev is not None and sum(a * b for a, b in zip(q, prev)) < 0:
            for fc, v in zip(fcs, q):
                fc.keyframe_points[j].co[1] = -v
            q = [-v for v in q]
        prev = q
    KY.flush()
    frames = {k: f0 + v for k, v in T.items()}
    if rig is not None:
        try:
            from . import pigeon_rig as PR
        except ImportError:
            import pigeon_rig as PR
        prev_e = None
        for f in sorted(mats):
            e = mats[f].to_euler("XYZ", prev_e) if prev_e is not None else mats[f].to_euler("XYZ")
            prev_e = e
            KY.key(rig.root, "location", f, tuple(mats[f].to_translation()), "LINEAR")
            KY.key(rig.root, "rotation_euler", f, tuple(e), "LINEAR")
        h0, hn = frames["hop0"], frames["on"]
        PR.pose_body(rig, h0 - 3, 0.0, 0.0)
        PR.pose_body(rig, h0, -8 * D2R, 1.0)
        PR.pose_body(rig, h0 + 2, 6 * D2R, 0.0, grounded=False)
        PR.pose_body(rig, hn, 0.0, 0.8)
        PR.pose_body(rig, hn + 4, 0.0, 0.0)
        for fa, fb in ((h0, hn), (frames["flip0"], frames["back"]), (frames["unpresent"], frames["upright"])):
            PR.wings(rig, fa, 0.0)
            PR.wings(rig, fa + 2, 0.55, 35 * D2R)
            PR.wings(rig, (fa + fb) / 2, 0.45, -20 * D2R)
            PR.wings(rig, fb + 2, 0.3, 10 * D2R)
            PR.wings(rig, fb + 6, 0.0)
        if end == "cell":
            fa, fb = frames["lowered"], frames["in"]
            PR.pose_body(rig, fa, -6 * D2R, 0.8)
            PR.pose_body(rig, fa + 2, 4 * D2R, 0.0, grounded=False)
            PR.pose_body(rig, fb, 0.0, 0.8)
            PR.pose_body(rig, fb + 4, 0.0, 0.0)
            PR.wings(rig, fa, 0.0)
            PR.wings(rig, fa + 2, 0.5, 30 * D2R)
            PR.wings(rig, fb + 2, 0.0)
        PR.tail(rig, frames["flip0"], 0.0, 0.0)
        PR.tail(rig, frames["back"], 0.15, 0.6)
        PR.tail(rig, frames["upright"], 0.0, 0.0)
        KY.flush()
    if look:
        def bird_pt(f):
            M = mats.get(int(f)) or mats[min(mats, key=lambda k: abs(k - f))]
            return M @ Vector((0.0, 0.0, 0.06 * bird_scale))
        look_at(s, f0 + 2, frames["end"] - 2, bird_pt, ramp=6, step=4, max_pitch=18 * D2R)   # brim stays clear of the bird
    return dict(carrier=carrier, frames=frames, matrices=mats)


# ================================================================================================ v2: lens, S6 helpers
def _pos_fn(obj):
    """frame -> world position of obj (F-curves if unparented and keyed, else evaluated matrix_world)."""
    if obj.parent is None and obj.animation_data and obj.animation_data.action:
        return lambda f: Vector(KY.values(obj, "location", f))
    cache = {}

    def fn(f):
        f = int(f)
        if f not in cache:
            bpy.context.scene.frame_set(f)
            cache[f] = obj.matrix_world.translation.copy()
        return cache[f]
    return fn


def look_at_lens(s, f0, f1, cam, ramp=4, w_chest=0.12, step=4):
    """Head + eyes onto the camera (object; follows its motion) over [f0, f1] (ADD overlay). Returns f1."""
    return look_at(s, f0, f1, _pos_fn(cam), ramp=ramp, w_chest=w_chest, track=True, step=step)


def glance_at_camera(s, f, cam, hold=24, ramp=4):
    """Quick glance into the lens ('It sees you'): eyes and head snap to the camera, hold, release. Returns end frame."""
    return look_at_lens(s, f, f + hold + 2 * ramp, cam, ramp=ramp)


def walk_to_mark(s, f0, mark, via=(), face=None, gait="walk", speed=None, turn_min_deg=55.0):
    """Walk from the current root position to a mark (world xy) and stop; if face (world point, e.g. the camera) is
    given and the needed body turn exceeds turn_min_deg, add a stepping 90/180 deg turn toward it (smaller residuals are
    left to look_at). Returns the frame the Subject is settled."""
    p0 = KY.values(s.root, "location", f0)
    pts = [(p0[0], p0[1])] + [tuple(v[:2]) for v in via] + [tuple(mark[:2])]
    fe = walk_path(s, f0, pts, gait=gait, speed=speed)
    if face is not None:
        y_end = KY.value(s.root, "rotation_euler", 2, fe)
        dy = (yaw_face(mark, face) - y_end + math.pi) % (2 * math.pi) - math.pi
        if abs(dy) > turn_min_deg * D2R:
            deg = 180 if abs(dy) > 135 * D2R else 90
            fe = turn(s, fe + 1, "L" if dy > 0 else "R", deg)
    return fe


def look_around_slow(s, f0, f1, center, span_deg=40.0, w_chest=0.1):
    """Slow wary look-around centred on a world point (e.g. the camera): center -> left -> hold -> right -> hold ->
    center, eased; yaw stays within +-span around the centre so the face stays readable. Returns f1."""
    n = f1 - f0
    y_c, p_c, _ = _look_delta(s, f0, center)
    sp = span_deg * D2R
    seq = [(0.0, 0.0), (0.12, 0.0), (0.32, sp), (0.45, sp), (0.68, -sp), (0.82, -sp), (1.0, 0.0)]

    def keyset(dy, u):
        y = max(-80 * D2R, min(80 * D2R, y_c + dy))
        dv = Vector((math.sin(y), -math.cos(y), math.tan(p_c))).normalized()
        return (u * n, _pose_keys(y, p_c, dv, w_chest))
    kk = [(0.0, _pose_keys(0, 0, Vector((0, -1, 0)), w_chest))]
    kk[0][1]["ctl_look"] = (0.0, 0.0, 0.0)
    for u, dy in seq[1:-1]:
        kk.append(keyset(dy, u))
    z = _pose_keys(0, 0, Vector((0, -1, 0)), w_chest)
    z["ctl_look"] = (0.0, 0.0, 0.0)
    kk.append((float(n), z))
    strip(s, "look", _look_action(s, kk), f0, blend="ADD", face=False)
    return f1


def startle_freeze(s, f0, f_end, pose_frame=18, look=None):
    """DEPRECATED for the film (owner v3: "gary should be unaware and goes abt his day"); use stroll_unaware / idle_hum.
    Startle hop (motion_v2 'startle', shock face), then freeze on startle frame `pose_frame` (arms up, wide eyes) until
    f_end; look: optional world point / camera object to stare at during the freeze (keeps the face readable)."""
    st = strip(s, "act", "startle", f0, end_frame=pose_frame, blend_in=2)
    fe = int(round(st.frame_end))
    freeze(s, fe, f_end, pose_from=("startle", pose_frame))
    if look is not None:
        tgt = _pos_fn(look) if hasattr(look, "matrix_world") else look
        look_at(s, fe - 4, f_end, tgt, ramp=5, track=hasattr(look, "matrix_world"))
    return f_end


def face_angle(s, f, cam_pos):
    """Angle (deg) between the Subject's face direction (head bone) and the direction to cam_pos at frame f
    (0 = looking into the lens; > 100 means the face is turned away)."""
    bpy.context.scene.frame_set(int(f))
    M = s.arm.matrix_world @ s.arm.pose.bones["head"].matrix
    fwd = M.col[2].xyz.normalized()
    to = (Vector(cam_pos) - M.translation).normalized()
    return math.degrees(math.acos(max(-1.0, min(1.0, fwd.dot(to)))))


def unreadable_frames(s, f0, f1, cam, max_deg=100.0, step=3):
    """Frames in [f0, f1] where the face is turned more than max_deg away from the camera object."""
    pf = _pos_fn(cam)
    return [f for f in range(int(f0), int(f1) + 1, step) if face_angle(s, f, pf(f)) > max_deg]


def cup_show(s, f, show):
    """Key the coffee cup visible / hidden from frame f (CONSTANT), e.g. hidden while the right hand holds the bird."""
    if s.cup is None:
        return
    for o in [s.cup] + list(s.cup.children):
        o.hide_render = not show
        o.keyframe_insert("hide_render", frame=int(f))
        o.hide_viewport = not show
        o.keyframe_insert("hide_viewport", frame=int(f))
        for fc in o.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


# ================================================================================================ v3: Gary goes about his day
OUTFIT.update({"phone": (0.03, 0.03, 0.035), "screen": (0.35, 0.55, 0.85), "watch_band": (0.05, 0.05, 0.05), "watch_face": (0.8, 0.82, 0.85)})
CARRY_HOOK = (Vector((-0.20, -0.22, 0.95)), Vector((0.0, -1.0, 0.0)), Vector((0.0, 0.0, -1.0)))   # right hook pos, x, y (cup upright)
CARRY_POLE = Vector((-0.38, 0.25, 1.0))


def _simple_mat(name, color, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = 0.4
    if emit:
        b.inputs["Emission Color"].default_value = (*color, 1.0)
        b.inputs["Emission Strength"].default_value = emit
    m.diffuse_color = (*color, 1.0)
    return m


def _box_obj(name, size, mat, coll):
    import bmesh
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=size, verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    return o


def make_phone(s):
    """Phone prop lying on the left palm (screen toward the palm normal); hidden until phone_show."""
    if getattr(s, "phone", None) is not None:
        return s.phone
    hook = next(o for o in s.objs if o.name.startswith("HOOK_hand_L"))
    root = bpy.data.objects.new("subj_phone", None)
    s.coll.objects.link(root)
    root.parent = hook
    root.matrix_parent_inverse.identity()
    pf = palm_frame(s, "L")
    root.location = pf + Vector((0.0, 0.02, 0.007))
    body = _box_obj("subj_phone_body", (0.072, 0.148, 0.009), _simple_mat("MAT_subject_phone", OUTFIT["phone"]), s.coll)
    scr = _box_obj("subj_phone_screen", (0.064, 0.136, 0.001), _simple_mat("MAT_subject_screen", OUTFIT["screen"], 2.5), s.coll)
    body.parent = root
    scr.parent = root
    scr.location = (0.0, 0.0, 0.0048)
    s.phone = root
    s.objs += [root, body, scr]
    phone_show(s, 1, False)
    return root


def phone_show(s, f, show):
    """Key the phone visible / hidden from frame f (CONSTANT)."""
    for o in [s.phone] + list(s.phone.children):
        o.hide_render = o.hide_viewport = not show
        o.keyframe_insert("hide_render", frame=int(f))
        o.keyframe_insert("hide_viewport", frame=int(f))
        for fc in o.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"


def make_watch(s):
    """Wristwatch on the left forearm (bone-parented), face on the back of the wrist."""
    if getattr(s, "watch", None) is not None:
        return s.watch
    h, t = _bone_pts(s.arm, "forearm_L")
    ax = (t - h).normalized()
    c = t - ax * 0.035
    band = _torus("subj_watch_band", c, 0.056, 0.008, _simple_mat("MAT_subject_watch_band", OUTFIT["watch_band"]), s.coll,
                  rot=tuple(ax.to_track_quat("Z", "Y").to_euler()))
    out = Vector((1.0, 0.0, 0.0))
    out = (out - ax * out.dot(ax)).normalized()
    face = _tube("subj_watch_face", c + out * 0.054, c + out * 0.064, 0.021, 0.021,
                 _simple_mat("MAT_subject_watch_face", OUTFIT["watch_face"], 0.3), s.coll, segs=20)
    for o in (band, face):
        _bone_parent(o, s.arm, "forearm_L")
        s.objs.append(o)
    s.watch = face
    return face


def _overlay(s, name, keys_, f0, side, blend=4):
    a = _arm_action(s, "SUBJ_%s_%03d" % (name, len(bpy.data.actions)), keys_, side)
    return strip(s, "arm", a, f0, blend_in=blend, blend_out=blend, face=False)


def carry_pose(s):
    p, x, y = CARRY_HOOK
    return _hand_target(s, p, x, y, "R")


def carry_cup(s, f0, f1, ramp=8):
    """Right hand carries the cup upright in front of the hip (instead of swinging it) over [f0, f1]."""
    M = carry_pose(s)
    n = f1 - f0
    _overlay(s, "carry", [(0, None, None), (ramp, M, CARRY_POLE), (n - ramp, M, CARRY_POLE), (n, None, None)], f0, "R")
    return f1


def walk_sipping(s, f0, pts, sips=1, gait="walk", speed=0.85, **kw):
    """Relaxed walk carrying the coffee, with `sips` sips spread along the path (the legs keep walking). Returns the stop
    frame."""
    fe = walk_path(s, f0, pts, gait=gait, speed=speed, **kw)
    carry_cup(s, f0, fe + 14)
    M = carry_pose(s)
    for i in range(sips):
        fs = int(f0 + (i + 1) * (fe - f0) / (sips + 1) - 19)
        if fs > f0 + 8:
            sip_coffee(s, fs, raise_frames=12, hold=14, lower_frames=12, from_pose=M, from_pole=CARRY_POLE)
    return fe


def glance_phone(s, f0, dur=60, look=True, ramp=10):
    """Left hand brings the phone up in front of the chest, screen toward his face; he reads it, then pockets it."""
    make_phone(s)
    P = Vector((0.09, -0.31, 1.20))
    eye = Vector((0.0, -0.12, 1.51))
    n_ = (eye - P).normalized()
    M = _palm_hand(s, P, n_, Vector((-0.15, -0.25, 1.0)), "L")
    pole = Vector((0.38, 0.12, 0.95))
    _overlay(s, "phone", [(0, None, None), (ramp, M, pole), (dur - ramp, M, pole), (dur, None, None)], f0, "L")
    phone_show(s, f0 + 3, True)
    phone_show(s, f0 + dur - 3, False)
    if look:
        look_at(s, f0 + 4, f0 + dur - 2, lambda f: _root_matrix(s, f) @ (P + n_ * 0.01), ramp=6, step=8)
    return f0 + dur


def check_watch(s, f0, hold=24, ramp=9):
    """Left forearm comes across the chest, back of the wrist up; he looks at the watch, then lets the arm drop."""
    make_watch(s)
    P = Vector((0.02, -0.33, 1.12))
    M = _palm_hand(s, P, Vector((0.0, 0.0, -1.0)), Vector((-1.0, -0.25, 0.0)), "L")
    pole = Vector((0.45, 0.0, 1.0))
    n = 2 * ramp + hold
    _overlay(s, "watch", [(0, None, None), (ramp, M, pole), (ramp + hold, M, pole), (n, None, None)], f0, "L")
    wpos = P + Vector((0.11, 0.0, 0.05))
    look_at(s, f0 + 3, f0 + n, lambda f: _root_matrix(s, f) @ wpos, ramp=6, track=False)
    return f0 + n


def tap_rack(s, f0, point, side="L", taps=2, ramp=10, period=6, look_max_yaw=40 * D2R):
    """Casual knock on a rack/door with the palm: hand to a world point, `taps` quick taps, back. The head only glances
    toward the point (yaw clamped to look_max_yaw) so the face stays readable."""
    Mr = _root_matrix(s, f0 + ramp)
    t = Mr.inverted() @ Vector(point)
    sh = Vector((0.183 if side == "L" else -0.183, 0.0, 1.357))
    if (t - sh).length > 0.52:
        t = sh + (t - sh).normalized() * 0.52
    dirv = (t - sh).normalized()
    P0 = t - dirv * 0.02
    P1 = t - dirv * 0.055
    fing = Vector((0.0, 0.0, 1.0))
    pole = sh + Vector((0.35 if side == "L" else -0.35, 0.2, -0.4))
    K = [(0, None, None), (ramp, _palm_hand(s, P1, dirv, fing, side), pole)]
    f = ramp
    for _ in range(taps):
        K.append((f + period / 2, _palm_hand(s, P0, dirv, fing, side), pole))
        K.append((f + period, _palm_hand(s, P1, dirv, fing, side), pole))
        f += period
    K.append((f + ramp, None, None))
    _overlay(s, "tap", K, f0, side)
    look_at(s, f0, f0 + int(f + ramp), tuple(point), ramp=5, track=False, max_yaw=look_max_yaw)
    return f0 + int(f + ramp)


def stretch_yawn(s, f0, hold=20, ramp=12):
    """Left arm stretches overhead, chest arches, head tips back, jaw opens (yawn); release."""
    P = Vector((0.20, -0.05, 1.86))
    M = _palm_hand(s, P, Vector((0.0, -1.0, 0.0)), Vector((0.0, 0.0, 1.0)), "L")
    pole = Vector((0.55, 0.15, 1.45))
    n = 2 * ramp + hold
    _overlay(s, "stretch", [(0, None, None), (ramp, M, pole), (ramp + hold, M, pole), (n, None, None)], f0, "L", blend=6)
    zero = {"chest": (0.0, 0.0, 0.0), "neck": (0.0, 0.0, 0.0), "head": (0.0, 0.0, 0.0), "jaw": (0.0, 0.0, 0.0)}
    pk = {"chest": (-0.12, 0.0, 0.05), "neck": (-0.10, 0.0, 0.0), "head": (-0.25, 0.0, 0.08), "jaw": (0.38, 0.0, 0.0)}
    strip(s, "look", _look_action(s, [(0, zero), (ramp, pk), (ramp + hold, pk), (n, zero)]), f0, blend="ADD", face=False)
    return f0 + n


def idle_hum(s, f0, f1, hz=0.75, amp=1.0):
    """Relaxed humming sway on top of the idle or a walk: small head roll/yaw and chest sway, eased at both ends."""
    per = FPS / hz
    kk = []
    f = 0.0
    i = 0
    n = f1 - f0
    while f <= n:
        e = max(0.0, min(1.0, f / 15.0, (n - f) / 15.0))
        sgn = (0.0, 1.0, 0.0, -1.0)[i % 4]
        kk.append((f, {"head": (0.0, 0.05 * sgn * amp * e, 0.07 * sgn * amp * e), "chest": (0.0, 0.0, 0.025 * sgn * amp * e)}))
        f += per / 4
        i += 1
    kk.append((float(n), {"head": (0.0, 0.0, 0.0), "chest": (0.0, 0.0, 0.0)}))
    strip(s, "look", _look_action(s, kk), f0, blend="ADD", face=False)
    return f1


def stroll_unaware(s, f0, pts, phone=True, sips=1, hum=False, speed=0.85, **kw):
    """S6 (owner v3): Gary strolls, unaware of the birds: carries and sips his coffee, optionally reads his phone part way
    (left hand) and hums. Keep the path heading roughly toward the camera so the face stays readable (check with
    unreadable_frames). Returns the stop frame."""
    fe = walk_sipping(s, f0, pts, sips=sips, speed=speed, **kw)
    if phone:
        span = fe - f0
        glance_phone(s, int(f0 + 0.55 * span), dur=int(min(70, 0.4 * span)))
    if hum:
        idle_hum(s, f0, fe)
    return fe
