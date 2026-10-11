"""Report frames where the camera is inside or within a margin of any visible mesh (bounding-box pre-test, then exact
surface distance / inside test via a BVH for meshes; curves stay bounding-box only). Haze volumes are ignored.

Usage: blender -b <file.blend> --python scripts/check_camera_clearance.py -- [step] [margin_m]
Collection instances (crowd birds) are checked against the instanced objects' boxes when the camera is near them.
Overlay objects parented to the camera are ignored. Prints one line per hit run: frames, object name.
"""
import sys

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
step = int(argv[0]) if argv else 2
margin = float(argv[1]) if len(argv) > 1 else 0.003
scn = bpy.context.scene
camo = scn.camera


def under_cam(o):
    p = o.parent
    while p is not None:
        if p == camo:
            return True
        p = p.parent
    return False


SKIP = ("haze_", "sky_dome", "_cone", "haze_card")   # volumes, sky dome, fake light cones (camera may be inside)
meshes = [o for o in scn.objects if o.type in ("MESH", "CURVE") and not under_cam(o) and not any(k in o.name for k in SKIP)]
_bvh = {}


def near_surface(ob, mw, p):
    """Exact test after a bounding-box hit: world distance from p to the mesh surface <= margin (meshes only)."""
    if ob.type == "CURVE":     # animated curves (bevel/trim reveals): test the evaluated geometry of this frame
        ev = ob.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        try:
            if len(me.vertices) == 0:
                return False
            bm = bmesh.new()
            bm.from_mesh(me)
            t = BVHTree.FromBMesh(bm)
            bm.free()
        finally:
            ev.to_mesh_clear()
        hit = t.find_nearest(mw.inverted_safe() @ p)
        return hit[0] is not None and (mw @ hit[0] - p).length <= max(margin * 3, 0.01)
    if ob.type != "MESH":
        return True
    if ob.data.name not in _bvh:
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        _bvh[ob.data.name] = BVHTree.FromBMesh(bm)
        bm.free()
    q = mw.inverted_safe() @ p
    hit = _bvh[ob.data.name].find_nearest(q)
    if hit[0] is None:
        return False
    return (mw @ hit[0] - p).length <= max(margin * 3, 0.01)   # surface distance only (parity tests fail on open meshes)


def _inside_closed(ob, q):
    """Ray parity test (camera inside a closed mesh)."""
    t = _bvh[ob.data.name]
    n, o, d = 0, q.copy(), Vector((0.577, 0.577, 0.577))
    for _ in range(64):
        h = t.ray_cast(o, d)
        if h[0] is None:
            break
        n += 1
        o = h[0] + d * 1e-6
    return n % 2 == 1
insts = [o for o in scn.objects if o.instance_type == "COLLECTION" and o.instance_collection is not None]
inst_parts = {}
inst_radius = {}
for o in insts:
    c = o.instance_collection
    if c.name not in inst_parts:
        hidden = set()

        def walk(cc, off):
            off = off or cc.hide_render
            if off:
                hidden.update(cc.objects)
            for ch in cc.children:
                walk(ch, off)
        walk(c, False)     # child collections switched off for render (e.g. R's VARIANT_pulled_trays) are not checked
        inst_parts[c.name] = [x for x in c.all_objects if x.type == "MESH" and x not in hidden and not x.hide_render]


def basis_chain(x):
    """Local matrix from basis transforms (instanced collection objects are not evaluated in the scene)."""
    m = x.matrix_basis.copy()
    p = x.parent
    while p is not None:
        m = p.matrix_basis @ m
        p = p.parent
    return m


def coll_radius(name):
    """Bounding radius of an instanced collection about its origin (basis transforms), for the near-test."""
    if name not in inst_radius:
        r = 0.0
        for x in inst_parts[name]:
            m = basis_chain(x)
            for v in x.bound_box:
                r = max(r, (m @ Vector(v)).length)
        inst_radius[name] = r
    return inst_radius[name]


def inside(mw, ob_bbox, p):
    """Point p inside the local bounding box grown by `margin` metres (margin converted to local units per axis)."""
    q = mw.inverted_safe() @ p
    ml = [margin / max(mw.col[k].xyz.length, 1e-9) for k in range(3)]
    lo = Vector((min(v[0] for v in ob_bbox), min(v[1] for v in ob_bbox), min(v[2] for v in ob_bbox)))
    hi = Vector((max(v[0] for v in ob_bbox), max(v[1] for v in ob_bbox), max(v[2] for v in ob_bbox)))
    return all(lo[k] - ml[k] <= q[k] <= hi[k] + ml[k] for k in range(3))


hits = {}
for f in range(scn.frame_start, scn.frame_end + 1, step):
    scn.frame_set(f)
    p = camo.matrix_world.translation.copy()
    for o in meshes:
        if o.hide_render:
            continue
        # skip quick if far: bbox radius test in world
        mw = o.matrix_world
        bb = [mw @ Vector(v) for v in o.bound_box]
        c = sum(bb, Vector()) / 8.0
        r = max((v - c).length for v in bb)
        if (p - c).length > r + margin:
            continue
        if inside(mw, o.bound_box, p) and near_surface(o, mw, p):
            hits.setdefault(o.name, []).append(f)
    for o in insts:
        if o.hide_render:
            continue
        if (p - o.matrix_world.translation).length > coll_radius(o.instance_collection.name) * max(o.matrix_world.to_scale()) + margin:
            continue
        for x in inst_parts[o.instance_collection.name]:
            if any(k in x.name for k in SKIP):
                continue
            mw = o.matrix_world @ basis_chain(x)
            if inside(mw, x.bound_box, p) and near_surface(x, mw, p):
                hits.setdefault(o.name + "/" + x.name, []).append(f)


def runs(fr):
    out, s, prev = [], fr[0], fr[0]
    for x in fr[1:]:
        if x - prev > step:
            out.append((s, prev))
            s = x
        prev = x
    out.append((s, prev))
    return out


print("CLEARANCE step=%d margin=%.4f" % (step, margin))
for name, fr in sorted(hits.items(), key=lambda kv: kv[1][0]):
    for a, b in runs(fr):
        print("HIT frames %d-%d  S%d %.1fs  %s" % (a, b, (a - 1) // 300 + 1, ((a - 1) % 300) / 30.0, name))
print("CLEARANCE DONE, objects hit:", len(hits))
