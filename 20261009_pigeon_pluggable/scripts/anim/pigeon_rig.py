"""Pigeon animation rig for the hero GLB and a light crowd proxy derived from it (Blender 4.2, headless-safe, no bpy.ops
except the glTF importer).

Rig = hierarchy of empties with pivots (bird frame and units: see pgeom.py; root scale = world size factor, film 2.6):

    <name>               root: world loc/rot (euler XYZ = roll, -pitch_up, yaw); hops, flight, turns
      <name>_body        hip-line pivot: pitch, crouch (z drop)
        <name>_breath    chest pivot: breathing scale (body meshes)
        <name>_head      neck/head pivot: yaw / nod / roll (euler XYZ = roll, nod, yaw), bob = local x offset
          <name>_lid_tx / _lid_rx   lid pivots (scale z 0 = open, 1 = closed); TX lid closed = TX disable
        <name>_wflap_L/R shoulder: euler x = flap (stroke, + = up), z = sweep (+ = tip forward)
          <name>_wopen_L/R  quaternion: folded (identity) -> open (wing spread, chord behind the leading edge)
        <name>_tail      tail pivot: euler y = tail up (+), scale y = fan spread
        <name>_leg_L/R   hip pivot: euler y = -swing (swing + = foot forward), scale z = crouch compensation

Hero: `build_hero_rig()` imports the hero GLB and reparents its top-level parts to the pivots (head parts by centroid,
wings, tail + leash, legs + feet by name). Proxy: `build_proxy_library()` builds decimated part meshes once (and can save
them to assets/components/characters/pigeon_proxy.blend); `make_proxy_rig()` instantiates a rig sharing those meshes;
mode="head" makes a 5-object rack bird (static rest mesh + head + lids) for head-only crowds.

All motion functions take absolute scene frames, write keys straight into the empties (deterministic; seeds where
random), and return the end frame. Call keys.flush() once after keying (recalculates handles).
"""
import math
import os
import random

import bpy
from mathutils import Matrix, Quaternion, Vector

try:
    from . import keys as KY
    from . import pgeom as G
except ImportError:  # imported as a top-level module (sys.path has scripts/anim)
    import keys as KY
    import pgeom as G

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(HERE))
HERO_GLB = os.path.join(ROOT_DIR, "assets", "hero", "pigeon_osfp_dr8.glb")
PROXY_BLEND = os.path.join(ROOT_DIR, "assets", "components", "characters", "pigeon_proxy.blend")
S_BIRD = 2.6
D2R = math.pi / 180.0
PARTS = ("body", "head", "wing_L", "wing_R", "tail", "leg_L", "leg_R")

# open-wing orientation (left wing): folded tip -X -> +Y (outward), top edge +Z -> +X (leading edge forward), outer face +Y -> -Z
_RO_L = Matrix(((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)))
_MIR = Matrix(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)))
Q_OPEN = {"L": _RO_L.to_quaternion(), "R": (_MIR @ _RO_L @ _MIR).to_quaternion()}
# open wing reshaping in the folded-wing frame (applied before the open rotation): longer span (x), flattened wrap (y),
# broader chord (z); the folded wing is narrow and wraps over the back, so a rigid rotation alone reads as a thin blade
WING_OPEN_SCALE = (1.35, 0.35, 1.6)


# ================================================================================================ classification
def classify(name, c_mm):
    """Part for a top-level hero object from its name and centroid (bird frame, mm)."""
    n = name.lower()
    if n.startswith(("leash", "hook_leash", "tail_fan")):
        return "tail"
    if n.startswith("wing_left"):
        return "wing_L"
    if n.startswith("wing_right"):
        return "wing_R"
    if n.startswith(("leg_rx", "foot_rx", "hook_foot_rx")):
        return "leg_L"
    if n.startswith(("leg_tx", "foot_tx", "hook_foot_tx")):
        return "leg_R"
    if ("clip" in n or "flex" in n) and "leg" in n and c_mm.z < 15.0:
        return "leg_L" if c_mm.y > 0 else "leg_R"
    if c_mm.x > 30.0 and c_mm.z > 68.0:
        return "head"
    return "body"


PROXY_KEEP = ("skin_back_body", "skin_belly_body", "skin_back_neck", "skin_front_neck", "skin_head_cranium", "skin_head_jaw",
              "wing_left_flight", "wing_right_flight", "wing_left_covert", "wing_right_covert", "tail_fan", "eye_", "beak_",
              "leg_", "foot_", "leash_")
PROXY_DROP = ("band_text", "edge_pads", "seams", "screw")
PROXY_RATIO = {"body": 0.2, "head": 0.5, "wing_L": 0.55, "wing_R": 0.55, "tail": 0.7, "leg_L": 0.5, "leg_R": 0.5}


def _centroid_mm(o):
    if o.type == "MESH":
        bb = [o.matrix_world @ Vector(v) for v in o.bound_box]
        return sum(bb, Vector()) / 8.0 / 0.001
    return o.matrix_world.translation / 0.001


def _import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    hroot = next(o for o in new if o.name.startswith("HERO_ROOT"))
    hroot.rotation_mode = "XYZ"
    hroot.rotation_euler = (0.0, 0.0, math.pi / 2)    # GLB -Y beak -> bird +X
    hroot.location = (0, 0, 0)
    hroot.scale = (1, 1, 1)
    bpy.context.view_layer.update()
    return hroot, new


def _hooks(new):
    out = {}
    for o in new:
        if o.name.startswith("HOOK_eye_tx"):
            out["tx"] = tuple(o.matrix_world.translation)
        elif o.name.startswith("HOOK_eye_rx"):
            out["rx"] = tuple(o.matrix_world.translation)
    return out


# ================================================================================================ rig object
class PigeonRig:
    """Handles to the rig empties. Attributes: name, kind ('hero' | 'proxy' | 'proxy_head'), root, body, breath, head,
    lid {'tx','rx'}, wflap {'L','R'}, wopen {'L','R'}, tail, leg {'L','R'}, scale, eye {'tx','rx'} (bird frame, hero m)."""

    def __init__(self, name, kind, scale):
        self.name, self.kind, self.scale = name, kind, scale
        self.lid, self.wflap, self.wopen, self.leg = {}, {}, {}, {}
        self.body = self.breath = self.head = self.tail = None
        self.eye = dict(G.EYE)
        self.head_pivot = G.HEAD_PIVOT          # bird frame, hero m (shifted by a neck extension)
        self.neck_arm = None                    # neck deform armature (driven by self.head)
        self.objects = []

    @property
    def head_rest(self):
        return _rel(self.head_pivot, G.BODY_PIVOT)

    @property
    def full(self):
        return self.kind in ("hero", "proxy")

    def eye_world(self, k, f=None):
        """World position of eye k at frame f (evaluates the keyed root/body/head; f None = current state)."""
        if f is not None:
            bpy.context.scene.frame_set(int(f))
        bpy.context.view_layer.update()
        piv = self.head.matrix_world
        return piv @ (Vector(self.eye[k]) - Vector(self.head_pivot))

    @classmethod
    def from_root(cls, root):
        """Rebuild handles from an existing rig root (e.g. after reopening a blend)."""
        r = cls(root.name, root.get("pg_kind", "proxy"), root.get("pg_scale", S_BIRD))
        r.root = root
        kids = {}

        def walk(o):
            for c in o.children:
                kids[c.name] = c
                walk(c)
        walk(root)
        n = root.name
        r.body = kids.get(n + "_body")
        r.breath = kids.get(n + "_breath")
        r.head = kids.get(n + "_head")
        r.tail = kids.get(n + "_tail")
        r.neck_arm = kids.get(n + "_neckrig")
        for s in ("L", "R"):
            r.wflap[s], r.wopen[s], r.leg[s] = kids.get(n + "_wflap_" + s), kids.get(n + "_wopen_" + s), kids.get(n + "_leg_" + s)
        for k in ("tx", "rx"):
            r.lid[k] = kids.get(n + "_lid_" + k)
            if ("pg_eye_" + k) in root:
                r.eye[k] = tuple(root["pg_eye_" + k])
        if "pg_head_pivot" in root:
            r.head_pivot = tuple(root["pg_head_pivot"])
        return r


def _empty(name, loc=(0, 0, 0), parent=None, coll=None, size=0.01):
    o = bpy.data.objects.new(name, None)
    o.empty_display_size = size
    o.empty_display_type = "PLAIN_AXES"
    (coll or bpy.context.scene.collection).objects.link(o)
    if parent is not None:
        o.parent = parent
    o.location = loc
    o.rotation_mode = "XYZ"
    return o


def _rel(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _skeleton(name, kind, scale, coll, loc=(0, 0, 0), yaw=0.0, wings=True, legs=True, tail=True):
    """Create the pivot empties (rest pose = bird frame, hero metres under a scaled root)."""
    r = PigeonRig(name, kind, scale)
    r.root = _empty(name, loc, coll=coll, size=0.05)
    r.root.rotation_euler = (0.0, 0.0, yaw)
    r.root.scale = (scale,) * 3
    r.root["pg_kind"] = kind
    r.root["pg_scale"] = scale
    r.body = _empty(name + "_body", G.BODY_PIVOT, r.root, coll)
    r.breath = _empty(name + "_breath", _rel(G.BREATH_PIVOT, G.BODY_PIVOT), r.body, coll)
    r.head = _empty(name + "_head", _rel(G.HEAD_PIVOT, G.BODY_PIVOT), r.body, coll)
    if wings:
        for s in ("L", "R"):
            r.wflap[s] = _empty(name + "_wflap_" + s, _rel(G.WING_PIVOT[s], G.BODY_PIVOT), r.body, coll)
            r.wopen[s] = _empty(name + "_wopen_" + s, (0, 0, 0), r.wflap[s], coll)
            r.wopen[s].rotation_mode = "QUATERNION"
    if tail:
        r.tail = _empty(name + "_tail", _rel(G.TAIL_PIVOT, G.BODY_PIVOT), r.body, coll)
    if legs:
        for s in ("L", "R"):
            r.leg[s] = _empty(name + "_leg_" + s, _rel(G.LEG_PIVOT[s], G.BODY_PIVOT), r.body, coll)
    r.objects = [o for o in [r.root, r.body, r.breath, r.head, r.tail] + list(r.wflap.values()) + list(r.wopen.values())
                 + list(r.leg.values()) if o is not None]
    return r


def pivot_of(r, part):
    return {"body": r.breath, "head": r.head, "tail": r.tail, "wing_L": r.wopen.get("L"), "wing_R": r.wopen.get("R"),
            "leg_L": r.leg.get("L"), "leg_R": r.leg.get("R")}[part]


def pivot_rest(part):
    return {"body": G.BREATH_PIVOT, "head": G.HEAD_PIVOT, "tail": G.TAIL_PIVOT, "wing_L": G.WING_PIVOT["L"],
            "wing_R": G.WING_PIVOT["R"], "leg_L": G.LEG_PIVOT["L"], "leg_R": G.LEG_PIVOT["R"]}[part]


def _lid_mesh(name, k, mat, eye=None):
    """Disc lid (hero metres) relative to its pivot at the top of the eye; scale z 0..1 closes it downward."""
    import bmesh
    s = G.EYE_SIGN[k]
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    rr = G.EYE_R
    bmesh.ops.create_cone(bm, cap_ends=True, segments=20, radius1=rr, radius2=rr * 0.96, depth=0.0006)
    # cone axis is Z: rotate onto the lateral eye axis and drop by the radius so the pivot sits at the top of the eye
    rot = Matrix.Rotation(-s * math.pi / 2, 4, "X")
    bmesh.ops.transform(bm, matrix=rot, verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector((0.0, 0.0, -rr)), verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    if mat is not None:
        me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = True
    return me


def _add_lids(r, coll, mat, mesh_cache=None):
    for k in ("tx", "rx"):
        e = r.eye[k]
        s = G.EYE_SIGN[k]
        top = (e[0], e[1] + s * 0.0016, e[2] + G.EYE_R)
        piv = _empty(r.name + "_lid_" + k, _rel(top, r.head_pivot), r.head, coll, size=0.003)
        me = (mesh_cache or {}).get(k) or _lid_mesh("PGN_lid_" + k, k, mat, e)
        if mesh_cache is not None:
            mesh_cache[k] = me
        ob = bpy.data.objects.new(r.name + "_lidmesh_" + k, me)
        coll.objects.link(ob)
        ob.parent = piv
        piv.scale = (1.0, 1.0, 0.0)
        r.lid[k] = piv
        r.objects += [piv, ob]


# ================================================================================================ neck deform (owner F1)
LIB_VERSION = 2          # proxy library layout: 2 = separate skinned 'neck' part (body/rest exclude the neck skins)
_NECK_ARM = {}
NECK_RIGID = ("skin_back_body", "skin_belly_body", "skin_head_", "eye_", "beak_", "wing_", "tail_", "leg_", "foot_", "leash",
              "HOOK", "tx_", "rx_", "dsp_")
NECK_SKIN = ("skin_back_neck", "skin_front_neck")


def neck_armature_data():
    """Shared armature data: NECK_BONES all at the head pivot, along +Y (identity orientation, so bone-local euler = the
    head empty's euler in the bird frame). Bones carry no animation; per-rig drivers set their pose."""
    d = _NECK_ARM.get("data")
    if d is not None and d.name in bpy.data.armatures:
        return d
    d = bpy.data.armatures.new("PGN_neck_arm")
    d.use_fake_user = True
    tmp = bpy.data.objects.new("_tmp_neck_arm", d)
    bpy.context.scene.collection.objects.link(tmp)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = tmp
    tmp.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for name, _, _, _ in G.NECK_BONES:
        eb = d.edit_bones.new(name)
        eb.head = G.HEAD_PIVOT
        eb.tail = (G.HEAD_PIVOT[0], G.HEAD_PIVOT[1] + 0.004, G.HEAD_PIVOT[2])
        eb.roll = 0.0
        eb.use_deform = True
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(tmp, do_unlink=True)
    _NECK_ARM["data"] = d
    return d


def _drv(pb, path, k, target, tpath, expr):
    fc = pb.driver_add(path, k)
    dr = fc.driver
    dr.type = "SCRIPTED"
    v = dr.variables.new()
    v.name = "v"
    v.type = "SINGLE_PROP"
    v.targets[0].id = target
    v.targets[0].data_path = tpath
    dr.expression = expr


def add_neck_rig(r, coll):
    """Neck armature object for rig r (child of the body, armature space = bird frame). Each neck bone rotates about the
    head pivot by its fraction of the head empty's rotation and moves by its fraction of the head's offset from the rest
    pivot (bob, neck extension): simple-expression drivers, so any keys on r.head (head_pose, head_snap, behaviours, S's
    own keys) drive the neck."""
    arm = bpy.data.objects.new(r.name + "_neckrig", neck_armature_data())
    coll.objects.link(arm)
    arm.parent = r.body
    arm.location = tuple(-x for x in G.BODY_PIVOT)
    arm.hide_render = True
    if arm.pose is None:
        bpy.context.view_layer.update()          # the pose is created on the first depsgraph evaluation
    h0 = _rel(G.HEAD_PIVOT, G.BODY_PIVOT)
    for name, _, fr, fl in G.NECK_BONES[1:]:
        pb = arm.pose.bones[name]
        pb.rotation_mode = "XYZ"
        for k in range(3):
            _drv(pb, "rotation_euler", k, r.head, "rotation_euler[%d]" % k, "v*%.4f" % fr)
            _drv(pb, "location", k, r.head, "location[%d]" % k, "(v-%.6f)*%.4f" % (h0[k], fl))
    r.neck_arm = arm
    r.objects.append(arm)
    return arm


def skin_neck(mesh, positions, pure_neck=True):
    """Vertex groups (NECK_BONES) on a mesh from bird-frame vertex positions (hero m)."""
    groups = {}
    for name, _, _, _ in G.NECK_BONES:
        groups[name] = mesh.vertex_groups.get(name) or mesh.vertex_groups.new(name=name)
    buckets = {}
    for i, p in enumerate(positions):
        for b, w in G.neck_weights(p, pure_neck).items():
            buckets.setdefault((b, round(w, 3)), []).append(i)
    for (b, w), idx in buckets.items():
        groups[b].add(idx, w, "REPLACE")


def _armature_mod(o, arm):
    m = o.modifiers.new("neck", "ARMATURE")
    m.object = arm
    m.use_vertex_groups = True
    m.use_deform_preserve_volume = True       # dual-quaternion blending: no candy-wrapper shrink between neck bones
    if len(o.modifiers) > 1:                  # imported GLB meshes carry no modifiers; keep the armature first anyway
        o.modifiers.move(len(o.modifiers) - 1, 0)
    return m


# ================================================================================================ hero
def build_hero_rig(name="HERO", glb=HERO_GLB, scale=S_BIRD, loc=(0, 0, 0), yaw=0.0, coll=None, hide_leash=False, log=print,
                   neck_deform=True):
    """Import the hero GLB and rig it. Returns PigeonRig (r.glb = imported objects, r.by(sub) finds one by name)."""
    coll = coll or bpy.context.scene.collection
    hroot, new = _import_glb(glb)
    hk = _hooks(new)
    r = _skeleton(name, "hero", 1.0, coll)
    for k in ("tx", "rx"):
        if k in hk:
            r.eye[k] = hk[k]
            r.root["pg_eye_" + k] = hk[k]
    bpy.context.view_layer.update()
    top = [o for o in new if o.parent == hroot]
    counts = {}
    for o in top:
        part = classify(o.name, _centroid_mm(o))
        counts[part] = counts.get(part, 0) + 1
        mw = o.matrix_world.copy()
        o.parent = pivot_of(r, part)
        o.matrix_world = mw
        if hide_leash and o.name.startswith(("leash", "HOOK_leash")):
            o.hide_render = o.hide_viewport = True
    hroot.parent = r.root
    hroot.matrix_parent_inverse.identity()
    log("HERO_RIG parts", counts)
    if neck_deform:
        bpy.context.view_layer.update()
        arm = add_neck_rig(r, coll)
        skinned = []
        for o in top:
            if o.type != "MESH" or o.name.startswith(NECK_RIGID) or o.parent not in (r.breath, r.head):
                continue
            pos = [tuple(o.matrix_world @ v.co) for v in o.data.vertices]
            pure = o.name.startswith(NECK_SKIN)
            if not pure and not any(0.02 < G.neck_t(q) < 0.98 and G.neck_gate(q) > 0.5 for q in pos):
                continue
            if o.parent == r.head:                       # neck-crossing part that was riding the head: back to the body
                mw = o.matrix_world.copy()
                o.parent = r.breath
                o.matrix_world = mw
            skin_neck(o, pos, pure)
            _armature_mod(o, arm)
            skinned.append(o.name)
        r.neck_skinned = skinned
        log("HERO_RIG neck skinned", len(skinned), skinned)
    cran = next((o for o in new if o.name.startswith("skin_head_cranium")), None)
    _add_lids(r, coll, cran.active_material if cran else None)
    r.root.scale = (scale,) * 3
    r.root.location = loc
    r.root.rotation_euler = (0.0, 0.0, yaw)
    r.scale = scale
    r.root["pg_scale"] = scale
    r.glb = new
    r.by = lambda sub: next((o for o in new if sub in o.name), None)
    r.objects += new
    return r


# ================================================================================================ proxy
def _merge(objs, name, offset, ratio, log=print):
    """Join meshes (world transforms baked, then translated by -offset) into one new mesh; decimate; return mesh."""
    import bmesh
    bm = bmesh.new()
    mats = []
    for o in objs:
        me = o.data.copy()
        me.transform(o.matrix_world)
        me.transform(Matrix.Translation(-Vector(offset)))
        remap = []
        for m in me.materials:
            if m not in mats:
                mats.append(m)
            remap.append(mats.index(m))
        if remap:
            idx = [0] * len(me.polygons)
            me.polygons.foreach_get("material_index", idx)
            me.polygons.foreach_set("material_index", [remap[i] if i < len(remap) else 0 for i in idx])
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    out = bpy.data.meshes.new(name + "_full")
    bm.to_mesh(out)
    bm.free()
    for m in mats:
        out.materials.append(m)
    n0 = len(out.polygons)
    if ratio < 0.999:
        tmp = bpy.data.objects.new("_tmp_dec", out)
        bpy.context.scene.collection.objects.link(tmp)
        mod = tmp.modifiers.new("dec", "DECIMATE")
        mod.ratio = ratio
        dg = bpy.context.evaluated_depsgraph_get()
        res = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
        bpy.data.objects.remove(tmp, do_unlink=True)
        bpy.data.meshes.remove(out)
        out = res
    out.name = name
    for p in out.polygons:
        p.use_smooth = True
    log("PROXY %-16s objs %3d faces %6d -> %6d" % (name, len(objs), n0, len(out.polygons)))
    return out


def _lenslet_mesh(name, src, mat, offset):
    """2x4 low-poly lenslet domes over the bounding box of a GLB lenslet array (bird frame), relative to offset."""
    import bmesh
    bb = [src.matrix_world @ Vector(v) for v in src.bound_box]
    mn = Vector([min(v[i] for v in bb) for i in range(3)])
    mx = Vector([max(v[i] for v in bb) for i in range(3)])
    c = (mn + mx) / 2
    ext = mx - mn
    outer_y = mx.y if c.y > 0 else mn.y
    lx, lz = ext.x, ext.z
    cols, rows = (4, 2) if lx >= lz else (2, 4)
    r = 0.42 * min(lx / cols, lz / rows)
    bm = bmesh.new()
    for i in range(cols):
        for j in range(rows):
            p = Vector((mn.x + (i + 0.5) * lx / cols, outer_y, mn.z + (j + 0.5) * lz / rows)) - Vector(offset)
            m = Matrix.Translation(p) @ Matrix.Diagonal((1.0, 0.45, 1.0, 1.0))
            bmesh.ops.create_uvsphere(bm, u_segments=10, v_segments=6, radius=r, matrix=m)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if mat is not None:
        me.materials.append(mat)
    for p in me.polygons:
        p.use_smooth = True
    return me


def build_proxy_library(glb=HERO_GLB, save=None, ratios=None, include_leash=True, log=print):
    """Build the proxy part meshes from the hero GLB (outer shell, wings, tail fan, legs/feet, beak, eye rims; internals
    dropped; decimated). Each part mesh is relative to its pivot (hero metres). Also builds 'rest' = all body-side parts
    joined in the bird frame (for head-only rack birds) and 'head_full' (head + lenslets). Returns {part: mesh}.
    save: path of a .blend to write the meshes into (fake users), e.g. PROXY_BLEND."""
    ratios = dict(PROXY_RATIO, **(ratios or {}))
    hroot, new = _import_glb(glb)
    hk = _hooks(new)
    groups = {p: [] for p in PARTS}
    groups["neck"] = []
    lens = {}
    for o in new:
        if o.type != "MESH":
            continue
        n = o.name
        if "lenslet" in n and ("eye" in n or "lenslet_array" in n):
            lens["rx" if _centroid_mm(o).y > 0 else "tx"] = o
            continue
        if not n.startswith(PROXY_KEEP) or any(d in n for d in PROXY_DROP):
            continue
        if n.startswith("leash_") and not include_leash:
            continue
        groups["neck" if n.startswith(NECK_SKIN) else classify(n, _centroid_mm(o))].append(o)
    meshes = {}
    if groups["neck"]:
        nm = _merge(groups["neck"], "PGN_proxy_neck", (0.0, 0.0, 0.0), ratios.get("neck", 0.6), log)
        meshes["neck"] = nm
        tmpo = bpy.data.objects.new("_tmp_neck", nm)
        skin_neck(tmpo, [tuple(v.co) for v in nm.vertices], True)
        bpy.data.objects.remove(tmpo, do_unlink=True)
    for p in PARTS:
        if groups[p]:
            meshes[p] = _merge(groups[p], "PGN_proxy_" + p, pivot_rest(p), ratios[p], log)
    lmat = {}
    for k, o in lens.items():
        mat = o.active_material
        lm = _lenslet_mesh("PGN_proxy_lens_" + k, o, mat, G.HEAD_PIVOT)
        lmat[k] = lm
    # head with lenslets joined (one object per head)
    if lmat:
        import bmesh
        bm = bmesh.new()
        mats = list(meshes["head"].materials)
        bm.from_mesh(meshes["head"])
        for k, lm in lmat.items():
            m = lm.materials[0] if len(lm.materials) else None
            if m is not None and m not in mats:
                mats.append(m)
            tmp = lm.copy()
            if m is not None:
                tmp.polygons.foreach_set("material_index", [mats.index(m)] * len(tmp.polygons))
            bm.from_mesh(tmp)
            bpy.data.meshes.remove(tmp)
        hm = bpy.data.meshes.new("PGN_proxy_head_full")
        bm.to_mesh(hm)
        bm.free()
        for m in mats:
            hm.materials.append(m)
        for pp in hm.polygons:
            pp.use_smooth = True
        meshes["head_full"] = hm
        for lm in lmat.values():
            bpy.data.meshes.remove(lm)
    else:
        meshes["head_full"] = meshes["head"]
    # rest mesh for head-only birds: all non-head parts at their rest pose, bird frame
    import bmesh
    bm = bmesh.new()
    mats = []
    for p in PARTS:
        if p == "head" or p not in meshes:
            continue
        tmp = meshes[p].copy()
        tmp.transform(Matrix.Translation(Vector(pivot_rest(p))))
        remap = []
        for m in tmp.materials:
            if m not in mats:
                mats.append(m)
            remap.append(mats.index(m))
        if remap:
            idx = [0] * len(tmp.polygons)
            tmp.polygons.foreach_get("material_index", idx)
            tmp.polygons.foreach_set("material_index", [remap[i] for i in idx])
        bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
    rest = bpy.data.meshes.new("PGN_proxy_rest")
    bm.to_mesh(rest)
    bm.free()
    for m in mats:
        rest.materials.append(m)
    for pp in rest.polygons:
        pp.use_smooth = True
    meshes["rest"] = rest
    cran = next((o for o in new if o.name.startswith("skin_head_cranium")), None)
    lid_mat = cran.active_material if cran else None
    for k in ("tx", "rx"):
        meshes["lid_" + k] = _lid_mesh("PGN_proxy_lid_" + k, k, lid_mat, hk.get(k))
    for m in meshes.values():
        m["pg_lib_version"] = LIB_VERSION
        m["pg_eye_tx"] = hk.get("tx", G.EYE["tx"])
        m["pg_eye_rx"] = hk.get("rx", G.EYE["rx"])
    # drop the imported objects (meshes we merged from are freed when orphaned)
    for o in new:
        bpy.data.objects.remove(o, do_unlink=True)
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        bpy.data.libraries.write(save, set(meshes.values()), fake_user=True, compress=True)
        log("PROXY saved", save, "%.1f MB" % (os.path.getsize(save) / 1e6))
    return meshes


def load_proxy_library(path=PROXY_BLEND):
    """Append the proxy meshes saved by build_proxy_library(save=...). Returns {part: mesh}."""
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.meshes = [n for n in src.meshes if n.startswith("PGN_proxy_")]
    out = {}
    for m in dst.meshes:
        out[m.name.replace("PGN_proxy_", "").split(".")[0]] = m
    return out


def proxy_library(path=PROXY_BLEND, rebuild=False, log=print):
    """Load the saved proxy library, building (and saving) it from the hero GLB if absent or older than the GLB."""
    if (not rebuild and os.path.exists(path) and os.path.exists(HERO_GLB)
            and os.path.getmtime(path) >= os.path.getmtime(HERO_GLB)):
        lib = load_proxy_library(path)
        if lib.get("body") is not None and lib["body"].get("pg_lib_version", 1) >= LIB_VERSION:
            return lib
        for m in lib.values():
            bpy.data.meshes.remove(m)
    return build_proxy_library(save=path, log=log)


_NECK_MESH = {}


def _neck_mesh(lib, length):
    """Neck filler (tapered tube along +X from the neck base, hero m) for a neck extension; iridescent neck material."""
    import bmesh
    key = round(length, 4)
    if key in _NECK_MESH:
        return _NECK_MESH[key]
    me = bpy.data.meshes.new("PGN_neck_%03d" % int(key * 1e4))
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=False, segments=16, radius1=0.0105, radius2=0.0085, depth=length,
                          matrix=Matrix.Translation((0.0, 0.0, length / 2)))
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.pi / 2, 3, "Y"))
    bm.to_mesh(me)
    bm.free()
    m = next((mm for mm in lib["body"].materials if mm and "neck" in mm.name), None) or (lib["body"].materials[0] if lib["body"].materials else None)
    if m is not None:
        me.materials.append(m)
    for p in me.polygons:
        p.use_smooth = True
    _NECK_MESH[key] = me
    return me


NECK_BASE = (0.026, 0.0, 0.066)    # where an extended neck leaves the body (bird frame, hero m)


def make_proxy_rig(name, lib, loc=(0, 0, 0), yaw=0.0, scale=S_BIRD, mode="full", lids=True, coll=None, neck_ext=0.0,
                   neck_deform=True):
    """Instantiate a proxy rig sharing lib meshes. mode 'full' = all parts on pivots (street birds, flyers);
    'head' = root + static rest mesh + head (+ lids): 5-7 objects, for head-only rack crowds.
    neck_ext: forward head offset in WORLD metres (hall_layout cell 'neck_extension_m'); the head pivot moves forward and a
    neck filler tube bridges the gap (pass the same value to behaviors.BirdSpec.neck_ext)."""
    coll = coll or bpy.context.scene.collection
    full = mode == "full"
    r = _skeleton(name, "proxy" if full else "proxy_head", scale, coll, loc, yaw, wings=full, legs=full, tail=full)
    eye_tx = tuple(lib["head"].get("pg_eye_tx", G.EYE["tx"]))
    eye_rx = tuple(lib["head"].get("pg_eye_rx", G.EYE["rx"]))
    d = neck_ext / scale
    r.head_pivot = (G.HEAD_PIVOT[0] + d, G.HEAD_PIVOT[1], G.HEAD_PIVOT[2])
    r.eye = {"tx": (eye_tx[0] + d, eye_tx[1], eye_tx[2]), "rx": (eye_rx[0] + d, eye_rx[1], eye_rx[2])}
    r.root["pg_eye_tx"], r.root["pg_eye_rx"] = r.eye["tx"], r.eye["rx"]
    r.root["pg_head_pivot"] = r.head_pivot
    r.root["pg_neck_ext"] = neck_ext
    r.head.location = r.head_rest

    def ob(nm, me, parent):
        o = bpy.data.objects.new(nm, me)
        coll.objects.link(o)
        o.parent = parent
        r.objects.append(o)
        return o
    if full:
        for p in PARTS:
            if p in lib and p != "head":
                ob(name + "_m_" + p, lib[p], pivot_of(r, p))
    else:
        o = ob(name + "_m_rest", lib["rest"], r.body)
        o.location = tuple(-x for x in G.BODY_PIVOT)
    ob(name + "_m_head", lib.get("head_full", lib["head"]), r.head)
    if "neck" in lib:                                     # skinned neck (library v2): deforms with the head
        nk = ob(name + "_m_neck", lib["neck"], r.body)
        nk.location = tuple(-x for x in G.BODY_PIVOT)
        if neck_deform:
            arm = add_neck_rig(r, coll)
            if not nk.vertex_groups:                      # names live on the mesh since Blender 3.0; guard anyway
                for bn, _, _, _ in G.NECK_BONES:
                    nk.vertex_groups.new(name=bn)
            nk.parent = arm
            nk.location = (0.0, 0.0, 0.0)
            _armature_mod(nk, arm)
    elif d > 0.002:
        L = (r.head_pivot[0] - NECK_BASE[0]) + 0.004
        nk = ob(name + "_m_neck", _neck_mesh(lib, L), r.body)
        nk.location = _rel(NECK_BASE, G.BODY_PIVOT)
        nk.rotation_euler = (0.0, -math.atan2(r.head_pivot[2] - NECK_BASE[2], r.head_pivot[0] - NECK_BASE[0]), 0.0)
    if lids:
        _add_lids(r, coll, None, mesh_cache={"tx": lib.get("lid_tx"), "rx": lib.get("lid_rx")})
    return r


# ================================================================================================ primitives
def _ease(a, b, u):
    u = max(0.0, min(1.0, u))
    return a + (b - a) * (u * u * (3 - 2 * u))


def place(r, f, pos, yaw=0.0, pitch_up=0.0, roll=0.0, interp="BEZIER"):
    """Key the root world location and orientation (radians)."""
    KY.key(r.root, "location", f, tuple(pos), interp)
    KY.key(r.root, "rotation_euler", f, (roll, -pitch_up, yaw), interp)


def root_state(r, f):
    loc = KY.values(r.root, "location", f)
    rot = KY.values(r.root, "rotation_euler", f)
    return Vector(loc), rot[2], -rot[1], rot[0]


def pose_body(r, f, pitch_up=0.0, crouch=0.0, grounded=True, interp="BEZIER"):
    """Body pitch about the hip line and crouch 0..1 (body drops 2.4 mm hero; legs shorten so the feet stay planted).
    grounded: legs counter-rotate the pitch so the feet do not move."""
    drop = 0.0024 * crouch
    KY.key(r.body, "location", f, (G.BODY_PIVOT[0], G.BODY_PIVOT[1], G.BODY_PIVOT[2] - drop), interp)
    KY.key(r.body, "rotation_euler", f, (0.0, -pitch_up, 0.0), interp)
    if r.leg and grounded:
        hip = G.LEG_PIVOT["L"][2] - drop
        for s in ("L", "R"):
            KY.key(r.leg[s], "rotation_euler", f, (0.0, pitch_up, 0.0), interp)
            KY.key(r.leg[s], "scale", f, (1.0, 1.0, hip / G.LEG_PIVOT["L"][2]), interp)


def legs(r, f, swing_L=0.0, swing_R=None, tuck=0.0, interp="BEZIER"):
    """Airborne legs: swing (rad, + = feet forward), tuck 0..1 (feet back and up under the tail)."""
    if not r.leg:
        return
    swing_R = swing_L if swing_R is None else swing_R
    for s, sw in (("L", swing_L), ("R", swing_R)):
        a = sw - tuck * 75 * D2R
        KY.key(r.leg[s], "rotation_euler", f, (0.0, -a, 0.0), interp)
        KY.key(r.leg[s], "scale", f, (1.0, 1.0, 1.0 - 0.25 * tuck), interp)


def head_pose(r, f, yaw=0.0, roll=0.0, nod=0.0, bob=0.0, interp="BEZIER"):
    """Key the head (radians; bob in hero metres along the bird +X)."""
    hp = r.head_rest
    KY.key(r.head, "rotation_euler", f, (roll, nod, yaw), interp)
    KY.key(r.head, "location", f, (hp[0] + bob, hp[1], hp[2]), interp)


def head_state(r, f):
    rot = KY.values(r.head, "rotation_euler", f)
    hp = r.head_rest
    bob = KY.value(r.head, "location", 0, f, None) - hp[0] if r.head.animation_data else 0.0
    return rot[2], rot[0], rot[1], bob


def head_snap(r, f, yaw, roll=0.0, nod=0.0, bob=0.0, dur=3, overshoot=0.12):
    """Pigeon head saccade: hold the current pose until f, snap to the target in `dur` frames with a small overshoot,
    settle 2 frames later. Returns the settle frame."""
    y0, r0, n0, b0 = head_state(r, f)
    head_pose(r, f, y0, r0, n0, b0)
    o = overshoot
    head_pose(r, f + dur, yaw + o * (yaw - y0), roll + o * (roll - r0), nod + o * (nod - n0), bob)
    head_pose(r, f + dur + 2, yaw, roll, nod, bob)
    return f + dur + 2


def head_look_at(r, f, target, eye="auto", dur=3, nod=0.0, yaw_lim=150 * D2R, roll_lim=60 * D2R, overshoot=0.12):
    """Snap the head so eye ('tx' | 'rx' | 'auto' = smaller turn) points at world target. Uses the root pose keyed at f
    (body pitch ignored). Returns (settle_frame, eye, yaw, roll, exact)."""
    pos, byaw, _, _ = root_state(r, f)
    k, y, ro, exact = G.stare(tuple(pos), byaw, r.root.scale[0], tuple(target), yaw_lim, roll_lim,
                              prefer=None if eye == "auto" else eye, eyes=r.eye, pivot=r.head_pivot)
    fe = head_snap(r, f, y, ro, nod, 0.0, dur, overshoot)
    return fe, k, y, ro, exact


def head_track(r, f0, f1, target_fn, eye="auto", step=(6, 12), seed=0, yaw_lim=150 * D2R, roll_lim=60 * D2R):
    """Pigeon-like stepwise tracking of a moving target: re-aim with a saccade every 6-12 frames. target_fn(frame) -> xyz."""
    rng = random.Random(seed)
    f = f0
    k = None if eye == "auto" else eye
    while f < f1 - 3:
        fe, k, _, _, _ = head_look_at(r, f, target_fn(f + 3), k or "auto", dur=rng.choice((2, 3, 3)), yaw_lim=yaw_lim,
                                      roll_lim=roll_lim, overshoot=0.08)
        f = max(fe, f + rng.randint(*step))
    return f


def head_tilt(r, f, roll, dur=4, hold=12):
    """Curious head tilt (roll, radians) and back."""
    y, r0, n, b = head_state(r, f)
    head_snap(r, f, y, roll, n, b, dur, 0.1)
    return head_snap(r, f + dur + 2 + hold, y, r0, n, b, dur, 0.05)


def head_bob(r, f, amp=0.004, n=1, period=6):
    """Quick forward thrust and return (n times)."""
    y, ro, nd, b = head_state(r, f)
    head_pose(r, f, y, ro, nd, b)
    for i in range(n):
        f0 = f + i * period
        head_pose(r, f0 + 2, y, ro, nd + 0.05, b + amp, "BEZIER")
        head_pose(r, f0 + period, y, ro, nd, b, "BEZIER")
    return f + n * period


def breathe(r, f0, f1, rate_hz=0.5, amp=0.012, seed=0, fps=30):
    """Idle breathing: chest scale y/z +-amp, keyed every quarter cycle (smooth Bezier). Deterministic phase."""
    rng = random.Random(seed)
    per = fps / rate_hz
    ph = rng.random() * per
    q = per / 4.0
    f = f0 - (ph % q)
    i = int(ph // q)
    while f <= f1 + q:
        s = (0.0, 1.0, 0.0, -1.0)[i % 4]
        KY.key(r.breath, "scale", f, (1.0, 1.0 + amp * s, 1.0 + amp * 0.8 * s), "BEZIER")
        f += q
        i += 1


def wings(r, f, open=0.0, flap=0.0, sweep=0.0, interp="BEZIER", sides=("L", "R")):
    """Wing state: open 0 (folded) .. 1 (spread); flap (rad, + = up stroke); sweep (rad, + = tips forward)."""
    if not r.wflap:
        return
    for s in sides:
        q = Quaternion().slerp(Q_OPEN[s], max(0.0, min(1.0, open)))
        sg = 1.0 if s == "L" else -1.0
        KY.key(r.wopen[s], "rotation_quaternion", f, tuple(q), interp)
        o = max(0.0, min(1.0, open))
        KY.key(r.wopen[s], "scale", f, tuple(1.0 + (w - 1.0) * o for w in WING_OPEN_SCALE), interp)
        KY.key(r.wflap[s], "rotation_euler", f, (sg * flap, 0.0, -sg * sweep), interp)


def wings_fold(r, f, interp="BEZIER"):
    wings(r, f, 0.0, 0.0, 0.0, interp)


def wings_half(r, f, interp="BEZIER"):
    wings(r, f, 0.5, 10 * D2R, 0.0, interp)


def wings_open(r, f, interp="BEZIER"):
    wings(r, f, 1.0, 0.0, 0.0, interp)


def flap_cycle(r, f0, f1, hz=4.0, up=55 * D2R, down=40 * D2R, open=1.0, sweep=8 * D2R, fps=30, phase=0.0):
    """Flapping between f0 and f1: up-stroke peak at phase 0, down-stroke peak at phase 0.5 (sweep forward on the down
    stroke). Ends at f1 on a mid-stroke key. Returns f1."""
    per = fps / hz
    f = f0 - phase * per
    i = 0
    while f < f1:
        if f >= f0:
            if i % 2 == 0:
                wings(r, f, open, up, -0.4 * sweep)
            else:
                wings(r, f, open, -down, sweep)
        f += per / 2
        i += 1
    wings(r, f1, open, 0.0, 0.0)
    return f1


def glide(r, f0, f1, dihedral=8 * D2R, tail_spread=0.6):
    """Wings spread with a slight V, tail fanned; small slow wing wobble."""
    wings(r, f0, 1.0, dihedral, 0.0)
    n = max(1, int((f1 - f0) / 18))
    for i in range(1, n + 1):
        wings(r, f0 + i * (f1 - f0) / n, 1.0, dihedral + (2 * D2R if i % 2 else -2 * D2R), 0.0)
    tail(r, f0, 0.0, tail_spread)
    tail(r, f1, 0.0, tail_spread)
    return f1


def tail(r, f, up=0.0, spread=0.0, interp="BEZIER"):
    """Tail pitch (rad, + = up) and fan spread 0..1 (scale y 1..1.7)."""
    if r.tail is None:
        return
    KY.key(r.tail, "rotation_euler", f, (0.0, up, 0.0), interp)
    KY.key(r.tail, "scale", f, (1.0, 1.0 + 0.7 * spread, 1.0), interp)


def blink(r, f, eye="both", dur=5):
    """Lid close-open. eye 'tx' | 'rx' | 'both'."""
    for k in (("tx", "rx") if eye == "both" else (eye,)):
        lid = r.lid.get(k)
        if lid is None:
            continue
        KY.key(lid, "scale", f, (1.0, 1.0, 0.0), "BEZIER")
        KY.key(lid, "scale", f + dur * 0.4, (1.0, 1.0, 1.0), "BEZIER")
        KY.key(lid, "scale", f + dur * 0.6, (1.0, 1.0, 1.0), "BEZIER")
        KY.key(lid, "scale", f + dur, (1.0, 1.0, 0.0), "BEZIER")
    return f + dur


def tx_disable(r, f0, f1, ramp=3):
    """TX disable: the TX (right) lid closes over [f0, f1] (ramp frames to close and open)."""
    lid = r.lid.get("tx")
    if lid is None:
        return f1
    for f, v in ((f0, 0.0), (f0 + ramp, 1.0), (f1 - ramp, 1.0), (f1, 0.0)):
        KY.key(lid, "scale", f, (1.0, 1.0, v), "BEZIER")
    return f1


def blinks(r, f0, f1, seed=0, every=(60, 150)):
    """Random natural blinks (both lids) in [f0, f1]."""
    rng = random.Random(seed)
    f = f0 + rng.randint(0, every[0])
    while f < f1 - 6:
        blink(r, f, "both", rng.choice((4, 5, 5, 6)))
        f += rng.randint(*every)


# ================================================================================================ ground motion
def _yaw_unwrap(prev, a):
    while a - prev > math.pi:
        a -= 2 * math.pi
    while a - prev < -math.pi:
        a += 2 * math.pi
    return a


def hop(r, f0, p1, yaw1=None, height=None, air=8, crouch=3):
    """Hop from the current root position to p1 (world): crouch, jump (parabola), land with an absorb crouch.
    Wings half-open for one beat. Returns the frame when settled."""
    p0, y0, _, _ = root_state(r, f0)
    p1 = Vector(p1)
    yaw1 = y0 if yaw1 is None else _yaw_unwrap(y0, yaw1)
    h = height if height is not None else 0.035 * r.root.scale[0]
    place(r, f0, p0, y0)
    pose_body(r, f0, 0.0, 0.0)
    pose_body(r, f0 + crouch, -8 * D2R, 1.0)
    place(r, f0 + crouch, p0, y0)
    fl = f0 + crouch
    for i in range(1, air + 1):
        u = i / air
        p = p0.lerp(p1, u)
        p.z += 4 * h * u * (1 - u)
        place(r, fl + i, p, y0 + (yaw1 - y0) * _ease(0, 1, u), 6 * D2R * math.sin(math.pi * u), 0.0, "LINEAR")
    if r.leg:
        legs(r, fl + 2, 10 * D2R, tuck=0.0)
        legs(r, fl + air - 2, 20 * D2R)
    pose_body(r, fl + 1, 6 * D2R, 0.0, grounded=False)
    pose_body(r, fl + air, 0.0, 0.9)
    pose_body(r, fl + air + 4, 0.0, 0.0)
    wings(r, fl, 0.0, 0.0)
    wings(r, fl + 2, 0.5, 30 * D2R)
    wings(r, fl + 5, 0.4, -15 * D2R)
    wings(r, fl + air + 2, 0.0, 0.0)
    tail(r, fl, 0.0, 0.0)
    tail(r, fl + 3, 8 * D2R, 0.5)
    tail(r, fl + air + 3, 0.0, 0.0)
    y, ro, n, b = head_state(r, f0)
    head_pose(r, fl + air, y, ro, n + 0.08, b + 0.002)
    head_pose(r, fl + air + 4, y, ro, n, b)
    return fl + air + 4


def perch_shuffle(r, f0, steps=2, step=None, direction=1, frames=7):
    """Sidestep along a perch: `steps` small lateral steps (direction +1 = bird left / -1 = right), body sway.
    Returns the end frame."""
    p, y, _, _ = root_state(r, f0)
    st = (step if step is not None else 0.010 * r.root.scale[0]) * direction
    lat = Vector((-math.sin(y), math.cos(y), 0.0))
    place(r, f0, p, y)
    f = f0
    for i in range(steps):
        q = p + lat * st * (i + 1)
        mid = (p + lat * st * (i + 0.5))
        mid.z += 0.002 * r.root.scale[0]
        sway = (5 * D2R) * direction
        KY.key(r.root, "rotation_euler", f + 2, (sway, 0.0, y))
        KY.key(r.root, "location", f + frames * 0.5, tuple(mid))
        place(r, f + frames, q, y, 0.0, 0.0)
        if r.leg:
            lead = "L" if direction > 0 else "R"
            KY.key(r.leg[lead], "rotation_euler", f + 2, (0.0, -12 * D2R, 0.0))
            KY.key(r.leg[lead], "rotation_euler", f + frames, (0.0, 0.0, 0.0))
        f += frames
    return f


def turn_in_place(r, f0, yaw1, frames=8, hop_turn=True):
    """Turn the body to yaw1 (world). hop_turn: a small hop (pigeons often hop-turn); else shuffling steps."""
    p, y0, _, _ = root_state(r, f0)
    if hop_turn:
        return hop(r, f0, p, yaw1, height=0.012 * r.root.scale[0], air=max(4, frames - 3), crouch=2)
    y1 = _yaw_unwrap(y0, yaw1)
    place(r, f0, p, y0)
    n = 3
    for i in range(1, n + 1):
        place(r, f0 + i * frames / n, p, y0 + (y1 - y0) * i / n)
    return f0 + frames


def walk(r, f0, pts, speed=None, step_frames=7):
    """Ground walk along world points with the pigeon head bob (thrust, then the head holds still in space while the body
    advances). speed (m/s world) defaults to 0.45 m/s x scale/2.6. Returns the end frame."""
    sc = r.root.scale[0]
    speed = speed or 0.45 * sc / 2.6
    poly = G.catmull([tuple(p) for p in pts], 12)
    S = G.arclen(poly)
    L = S[-1]
    n_steps = max(1, int(round(L / (speed * step_frames / 30.0))))
    T = n_steps * step_frames
    p0, y_prev, _, _ = root_state(r, f0)
    yh, ro, nd, _ = head_state(r, f0)
    step_len = L / n_steps / sc          # hero metres per step
    for i in range(T + 1):
        s = L * i / T
        p, d = G.sample_at(poly, S, s)
        yaw = _yaw_unwrap(y_prev, math.atan2(d[1], d[0]))
        y_prev = yaw
        k = i % step_frames
        sway = 3 * D2R * (1 if (i // step_frames) % 2 else -1) * math.sin(math.pi * k / step_frames)
        place(r, f0 + i, p, yaw, 0.0, sway, "LINEAR")
        # head: thrust over the first 2 frames of a step, then hold still in space (bob decreases as the body advances)
        if k == 0:
            head_pose(r, f0 + i, yh, ro, nd, -0.5 * step_len, "LINEAR")
        elif k == 2:
            head_pose(r, f0 + i, yh, ro, nd, 0.5 * step_len, "LINEAR")
        if r.leg and k in (0, step_frames // 2):
            sgn = 1 if (i // step_frames) % 2 else -1
            for s_, a in (("L", sgn), ("R", -sgn)):
                KY.key(r.leg[s_], "rotation_euler", f0 + i, (0.0, -a * 18 * D2R if k == 0 else 0.0, 0.0), "BEZIER")
    head_pose(r, f0 + T + 2, yh, ro, nd, 0.0)
    return f0 + T


# ================================================================================================ flight
def flight(r, f0, pts, speed=None, takeoff=True, land=True, flap_hz=4.0, glide_frac=0.35, end_yaw=None, fps=30,
           crouch_frames=5, flare_frames=10, settle_frames=12, seed=0):
    """Take-off -> flight -> landing along world points pts (feet-origin positions; pts[0] = perch, pts[-1] = landing
    spot). takeoff False: starts already flying at pts[0] (wings open, flapping); land False: ends flying at pts[-1].
    speed: mean air speed (world m/s; default 2.2 m/s x scale/2.6). Wings flap at flap_hz (4 Hz default: readable at
    30 fps; real pigeons flap faster), fast power flaps after take-off, a glide in the middle (glide_frac of the path),
    braking flaps and flare before touchdown, legs reach forward, absorb crouch, wings fold.
    Returns dict(f_launch, f_touch, f_end)."""
    rng = random.Random(seed)
    sc = r.root.scale[0]
    speed = speed or 2.2 * sc / 2.6
    poly = G.catmull([tuple(p) for p in pts], 24)
    S = G.arclen(poly)
    L = S[-1]
    T = max(12, int(round(L / speed * fps * (1.25 if (takeoff or land) else 1.0))))
    p_start, y_start, _, _ = root_state(r, f0)
    f = f0
    if takeoff:
        place(r, f0, poly[0], y_start)
        pose_body(r, f0, 0.0, 0.0)
        wings_fold(r, f0)
        tail(r, f0)
        head = head_state(r, f0)
        head_pose(r, f0, *head)
        head_pose(r, f0 + 3, 0.0, 0.0, -0.15, 0.0)               # head up, look ahead
        pose_body(r, f0 + crouch_frames - 1, 12 * D2R, 1.0)       # crouch, chest up
        wings(r, f0 + crouch_frames - 2, 0.6, 40 * D2R)           # wings start to open
        place(r, f0 + crouch_frames - 1, poly[0], y_start)
        f = f0 + crouch_frames
    f_launch = f
    # time -> arc length profile
    if takeoff and land:
        prog = lambda u: (1 - math.cos(math.pi * u)) / 2       # noqa: E731
    elif takeoff:
        prog = lambda u: 1 - math.cos(math.pi * u / 2)          # noqa: E731
    elif land:
        prog = lambda u: math.sin(math.pi * u / 2)              # noqa: E731
    else:
        prog = lambda u: u                                      # noqa: E731
    yaw_prev = y_start
    samples = []
    for i in range(T + 1):
        u = i / T
        s = L * prog(u)
        p, d = G.sample_at(poly, S, s)
        samples.append((p, d, s))
    for i, (p, d, s) in enumerate(samples):
        u = i / T
        hd = math.atan2(d[1], d[0]) if (abs(d[0]) + abs(d[1])) > 1e-6 else yaw_prev
        if takeoff and i < 6:
            hd = y_start + (_yaw_unwrap(y_start, hd) - y_start) * (i / 6)
        yaw = _yaw_unwrap(yaw_prev, hd)
        climb = math.asin(max(-1, min(1, d[2])))
        # bank from yaw rate
        if 0 < i < T:
            d0, d1 = samples[i - 1][1], samples[i + 1][1]
            dy = G.wrap(math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0]))
            v = (samples[i + 1][2] - samples[i - 1][2]) * fps / 2
            bank = max(-35 * D2R, min(35 * D2R, math.atan2(v * dy * fps / 2, 9.81) * 0.8))
        else:
            bank = 0.0
        pitch = 0.55 * climb
        if takeoff and u < 0.15:
            pitch += 25 * D2R * (1 - u / 0.15)
        if land and u > 1 - flare_frames / T:
            pitch += 35 * D2R * (u - (1 - flare_frames / T)) / (flare_frames / T)
        if i % 2 == 0 or i == T:
            place(r, f + i, p, yaw, pitch, bank, "LINEAR")
        yaw_prev = yaw
    f_touch = f + T
    if takeoff:
        pose_body(r, f + 1, 0.0, 0.0, grounded=False)
        legs(r, f + 2, -10 * D2R)
        legs(r, f + 8, 0.0, tuck=1.0)
    else:
        pose_body(r, f, 0.0, 0.0, grounded=False)
        legs(r, f, 0.0, tuck=1.0)
    # wings: power flaps, cruise flaps, glide, braking flaps
    fp_end = f + (int(0.45 * fps) if takeoff else 0)
    g0 = f + int(T * (0.5 - glide_frac / 2))
    g1 = f + int(T * (0.5 + glide_frac / 2))
    fl0 = f_touch - flare_frames if land else f_touch
    if takeoff:
        flap_cycle(r, f, fp_end, hz=flap_hz * 1.4, up=70 * D2R, down=50 * D2R)
    if glide_frac > 0 and g1 - g0 > 10:
        flap_cycle(r, fp_end, max(fp_end + 1, g0), hz=flap_hz, phase=rng.random() * 0.2)
        glide(r, g0 + 3, g1 - 3)
        flap_cycle(r, g1, fl0, hz=flap_hz) if fl0 - g1 > 4 else None
    else:
        flap_cycle(r, fp_end, fl0, hz=flap_hz)
    tail(r, f, 5 * D2R, 0.3)
    if land:
        flap_cycle(r, fl0, f_touch - 1, hz=flap_hz * 1.3, up=60 * D2R, down=35 * D2R, sweep=22 * D2R)
        wings(r, f_touch, 0.9, 35 * D2R, 18 * D2R)
        tail(r, fl0, -10 * D2R, 1.0)
        legs(r, fl0 - 4, 0.0, tuck=1.0)
        legs(r, fl0 + 2, 40 * D2R, tuck=0.0)                      # feet reach forward
        legs(r, f_touch - 1, 30 * D2R)
        yaw_end = yaw_prev if end_yaw is None else _yaw_unwrap(yaw_prev, end_yaw)
        place(r, f_touch, poly[-1], yaw_prev, 0.0, 0.0, "BEZIER")
        place(r, f_touch + 6, poly[-1], yaw_end, 0.0, 0.0, "BEZIER")
        pose_body(r, f_touch, 15 * D2R, 0.6)
        pose_body(r, f_touch + 3, -6 * D2R, 1.0)                  # absorb
        pose_body(r, f_touch + 8, 0.0, 0.0)
        wings(r, f_touch + 3, 0.6, 25 * D2R, 5 * D2R)
        wings(r, f_touch + 7, 0.25, 8 * D2R)
        wings_fold(r, f_touch + settle_frames)
        tail(r, f_touch + 2, 12 * D2R, 0.8)
        tail(r, f_touch + settle_frames, 0.0, 0.0)
        hy = 0.0
        head_pose(r, f_touch, hy, 0.0, -0.1, 0.0)
        head_pose(r, f_touch + 3, hy, 0.0, 0.12, 0.003)
        head_pose(r, f_touch + 8, hy, 0.0, 0.0, 0.0)
        return dict(f_launch=f_launch, f_touch=f_touch, f_end=f_touch + settle_frames)
    head_pose(r, f + 4, 0.0, 0.0, -0.05, 0.0)
    return dict(f_launch=f_launch, f_touch=None, f_end=f_touch)


def take_off(r, f0, pts, **kw):
    """Perch -> airborne (ends flying at pts[-1]). See flight()."""
    return flight(r, f0, pts, land=False, **kw)


def landing(r, f0, pts, **kw):
    """Airborne at pts[0] -> touchdown at pts[-1] (flare, feet reach, absorb, wings fold). See flight()."""
    return flight(r, f0, pts, takeoff=False, **kw)


def rest_pose(r, f):
    """Key the neutral pose on every channel at frame f (use once at the start of a shot)."""
    pose_body(r, f, 0.0, 0.0)
    head_pose(r, f)
    if r.wflap:
        wings_fold(r, f)
    tail(r, f)
    for k in r.lid:
        KY.key(r.lid[k], "scale", f, (1.0, 1.0, 0.0))


# ================================================================================================ feet contact (owner v3)
_SOLE = {}
_BVH = {}


def sole_points(r, z_max=0.0012):
    """Toe-sole points (bird frame, hero m): feet vertices within z_max of the lowest one, at the rest pose. Proxies: from
    the shared rest mesh (head-only) or the leg part meshes (full); hero: its foot/leg GLB meshes."""
    key = ("hero", r.name) if r.kind == "hero" else r.kind
    if key in _SOLE:
        return _SOLE[key]
    pts = []
    if r.kind == "hero":
        bpy.context.view_layer.update()
        inv = r.root.matrix_world.inverted()
        for o in r.glb:
            if o.type == "MESH" and o.name.startswith(("foot_", "leg_")):
                M = inv @ o.matrix_world
                pts += [M @ v.co for v in o.data.vertices]
    else:
        for o in r.objects:
            if o.type != "MESH":
                continue
            if o.name.endswith("_m_rest"):
                pts += [v.co.copy() for v in o.data.vertices]          # rest mesh is in the bird frame
            elif o.name.endswith(("_m_leg_L", "_m_leg_R")):
                piv = Vector(G.LEG_PIVOT[o.name[-1]])
                pts += [v.co + piv for v in o.data.vertices]
    zmin = min(p.z for p in pts)
    sole = [p.copy() for p in pts if p.z < zmin + z_max]
    _SOLE[key] = sole
    return sole


def surfaces_bvh(surfaces):
    """World-space BVH of mesh objects and collection-instance empties (instanced meshes expanded). Cached per list."""
    from mathutils.bvhtree import BVHTree
    key = tuple(sorted(o.name for o in surfaces))
    if key in _BVH:
        return _BVH[key]
    verts, polys = [], []
    dg = bpy.context.evaluated_depsgraph_get()

    def add(o, M):
        if o.type != "MESH":
            return
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        base = len(verts)
        verts.extend(M @ v.co for v in me.vertices)
        polys.extend([base + i for i in p.vertices] for p in me.polygons)
        ev.to_mesh_clear()

    def walk_coll(c, M):
        for o in c.all_objects:
            if o.instance_type == "COLLECTION" and o.instance_collection:
                walk_coll(o.instance_collection, M @ o.matrix_world)
            else:
                add(o, M @ o.matrix_world)
    for o in surfaces:
        if o.instance_type == "COLLECTION" and o.instance_collection is not None:
            off = Matrix.Translation(-o.instance_collection.instance_offset)
            walk_coll(o.instance_collection, o.matrix_world @ off)
        else:
            add(o, o.matrix_world)
    t = BVHTree.FromPolygons(verts, polys)
    _BVH[key] = t
    return t


def _set_neck_ext(r, ext):
    """Change a proxy rig's forward neck extension (world m) after creation; shifts existing head x keys."""
    d_old = r.head_pivot[0] - G.HEAD_PIVOT[0]
    d_new = ext / r.root.scale[0]
    dd = d_new - d_old
    r.head_pivot = (G.HEAD_PIVOT[0] + d_new, G.HEAD_PIVOT[1], G.HEAD_PIVOT[2])
    r.eye = {k: (v[0] + dd, v[1], v[2]) for k, v in r.eye.items()}
    r.root["pg_eye_tx"], r.root["pg_eye_rx"] = r.eye["tx"], r.eye["rx"]
    r.root["pg_head_pivot"] = r.head_pivot
    r.root["pg_neck_ext"] = ext
    r.head.location.x += dd
    ad = r.head.animation_data
    if ad and ad.action:
        fc = ad.action.fcurves.find("location", index=0)
        if fc:
            for kp in fc.keyframe_points:
                kp.co[1] += dd
                kp.handle_left[1] += dd
                kp.handle_right[1] += dd


def place_on_floor(r, f=1, surfaces=None, bvh=None, max_shift=0.10, step=0.002, support=0.98, keep_head=True, probe=0.03,
                   reach=0.08):
    """Rest a perched bird's toes on its floor: ray-cast down from every toe-sole point onto `surfaces` (mesh objects or
    collection-instance empties, e.g. the bird racks) or a prepared `bvh`; if less than `support` of the sole has floor
    under it (toes over the tread edge), slide the bird straight back (along -beak) in `step` increments up to
    `max_shift`; then set the root height so the lowest sole point touches the highest floor point under the sole
    (no floating, no sinking). keep_head: add the back-shift to the neck extension (proxies) so the head stays where it
    was. Keys the root at frame f (location only; yaw unchanged). Returns dict(shift, dz, support, ext, pos)."""
    bvh = bvh or surfaces_bvh(surfaces)
    pos, yaw, _, _ = root_state(r, f)
    s = r.root.scale[0]
    sole = sole_points(r)
    fwd = Vector((math.cos(yaw), math.sin(yaw), 0.0))
    R = Matrix.Rotation(yaw, 3, "Z")

    def cast(p0):
        hits, miss = [], 0
        for q in sole:
            w = p0 + R @ (q * s)
            h = bvh.ray_cast(w + Vector((0, 0, probe)), Vector((0, 0, -1)), probe + reach)
            if h[0] is None:
                miss += 1
            else:
                hits.append((h[0].z, w.z))
        return hits, 1.0 - miss / len(sole)
    shift = 0.0
    hits, sup = cast(pos)
    while sup < support and shift < max_shift - 1e-9:
        shift += step
        hits, sup = cast(pos - fwd * shift)
    newpos = pos - fwd * shift
    dz = 0.0
    if hits:
        dz = max(hz - wz for hz, wz in hits)            # raise (or lower) so the lowest-penetrating point is on the floor
        newpos = newpos + Vector((0, 0, dz))
    KY.key(r.root, "location", f, tuple(newpos), "BEZIER")
    r.root.location = newpos
    ext = float(r.root.get("pg_neck_ext", 0.0))
    if keep_head and shift > 0 and r.kind.startswith("proxy"):
        ext += shift
        _set_neck_ext(r, ext)
    return dict(shift=shift, dz=dz, support=sup, ext=ext, pos=tuple(newpos))


def plant_on_wire(r, f, p0, p1, radius=0.004, grip_x=0.012, yaw=None):
    """Perch on a wire / thin rail from p0 to p1 (world): the bird stands across it (body perpendicular, the side nearest
    its current yaw, or `yaw`), the wire under the middle of the toes (bird-frame x = grip_x hero m) and the toe bottoms
    on the wire top. Keys the root at frame f. Returns dict(pos, yaw)."""
    pos, y0, _, _ = root_state(r, f)
    a, b = Vector(p0), Vector(p1)
    d = (b - a).normalized()
    u = max(0.0, min((b - a).length, (pos - a).dot(d)))
    c = a + d * u
    if yaw is None:
        n = math.atan2(d.y, d.x) + math.pi / 2
        yaw = n if math.cos(n - y0) >= 0 else n + math.pi
    s = r.root.scale[0]
    zmin = min(q.z for q in sole_points(r))
    fwd = Vector((math.cos(yaw), math.sin(yaw), 0.0))
    newpos = c - fwd * (grip_x * s) + Vector((0, 0, radius - zmin * s))
    # correction: the sole may have no vertex exactly over the wire; drop until the closest sole point touches it
    R = Matrix.Rotation(yaw, 3, "Z")
    for _ in range(3):
        g = []
        for q in sole_points(r):
            w = newpos + R @ (q * s)
            u_ = max(0.0, min((b - a).length, (w - a).dot(d)))
            g.append((w - (a + d * u_)).length - radius)
        newpos = newpos - Vector((0, 0, min(g)))
    place(r, f, newpos, yaw)
    return dict(pos=tuple(newpos), yaw=yaw)


def plant_on_surface(r, f, surfaces=None, bvh=None, allow_shift=False, **kw):
    """Perch on a flat-ish surface (post top, ledge, railing tube, sign, AC unit): toes rest on the highest surface point
    under the sole; allow_shift moves the bird back if its toes overhang (see place_on_floor)."""
    kw.setdefault("probe", 0.25)
    kw.setdefault("reach", 0.5)
    return place_on_floor(r, f, surfaces, bvh, max_shift=0.10 if allow_shift else 0.0, keep_head=False, **kw)
