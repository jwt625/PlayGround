"""Category helpers for the `props` assets (Blender 4.2, headless). Built on scripts/assets/_common/common.py.

Provides: bmesh primitives (rounded box, lathe, loft, tube along a path), clay / metal / wood materials, armature and
bone-parenting helpers, drivers on root custom properties, text to mesh, multi-asset blend writer with previews.
All lengths in metres in the API unless a name ends in _mm.
"""
import json
import math
import os
import random
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import common as C  # noqa: E402

mm = C.mm
V = Vector
PI = math.pi
CAT = "props"
SCRATCH = "/private/tmp/claude-501/-Users-wentaojiang-Documents-GitHub-PlayGround/36a0a473-d33f-4dfa-9d16-9644ea7bb16f/scratchpad/agents/props"


# ------------------------------------------------------------------ mesh building
def obj_from_bm(name, bm, mat=None, smooth=True, recalc=True, sharp_deg=None):
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    if sharp_deg is not None:
        mark_sharp(bm, sharp_deg)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    if mat is not None:
        o.data.materials.append(mat)
    return o


def mark_sharp(bm, deg):
    """Mark edges sharp where the dihedral angle exceeds deg (faces stay smooth-shaded: split normals)."""
    lim = math.radians(deg)
    bm.normal_update()
    for e in bm.edges:
        if len(e.link_faces) == 2:
            a = e.link_faces[0].normal.angle(e.link_faces[1].normal, 0.0)
            e.smooth = a < lim
    for f in bm.faces:
        f.smooth = True


def _vs(bm, pts):
    return [bm.verts.new(p) for p in pts]


def ring_frame(center, u, v, pts2d):
    """Ring of 3D points from 2D points in the (u, v) plane at center. u x v must equal the path direction."""
    c = Vector(center)
    return [c + u * a + v * b for a, b in pts2d]


def circle2d(r, n, rx=None, ry=None, phase=0.0):
    rx = r if rx is None else rx
    ry = r if ry is None else ry
    return [(rx * math.cos(phase + 2 * PI * i / n), ry * math.sin(phase + 2 * PI * i / n)) for i in range(n)]


def superellipse2d(w, h, n=24, p=3.0, cx=0.0, cz=0.0):
    pts = []
    for i in range(n):
        t = 2 * PI * i / n
        c, s = math.cos(t), math.sin(t)
        x = (w / 2) * math.copysign(abs(c) ** (2 / p), c)
        z = (h / 2) * math.copysign(abs(s) ** (2 / p), s)
        pts.append((cx + x, cz + z))
    return pts


def loft_bm(bm, rings, cap_start=True, cap_end=True, closed=True):
    """Loft equal-length rings (lists of Vector). Rings CCW seen from the leading end (see ring_frame). Returns vert rings."""
    vr = [_vs(bm, r) for r in rings]
    n = len(rings[0])
    for k in range(len(vr) - 1):
        a, b = vr[k], vr[k + 1]
        rng = range(n) if closed else range(n - 1)
        for i in rng:
            j = (i + 1) % n
            try:
                bm.faces.new((a[i], a[j], b[j], b[i]))
            except ValueError:
                pass
    if cap_start:
        c = bm.verts.new(sum((v.co for v in vr[0]), Vector()) / n)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((c, vr[0][j], vr[0][i]))
    if cap_end:
        c = bm.verts.new(sum((v.co for v in vr[-1]), Vector()) / n)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((c, vr[-1][i], vr[-1][j]))
    return vr


def lathe_bm(bm, profile, segs=48, xf=None, closed_profile=False, phase=0.0):
    """Surface of revolution about local Z. profile = [(r, h), ...]; r == 0 collapses to one vertex.

    xf: optional Matrix applied to every vertex (orient/locate). closed_profile joins last to first (solid)."""
    xf = xf or Matrix.Identity(4)
    rings = []
    for r, h in profile:
        if abs(r) < 1e-9:
            rings.append(bm.verts.new(xf @ Vector((0, 0, h))))
        else:
            rings.append([bm.verts.new(xf @ Vector((r * math.cos(phase + 2 * PI * i / segs), r * math.sin(phase + 2 * PI * i / segs), h))) for i in range(segs)])
    pairs = list(range(len(rings) - 1)) + ([len(rings) - 1] if closed_profile else [])
    for k in pairs:
        a, b = rings[k], rings[(k + 1) % len(rings)]
        for i in range(segs):
            j = (i + 1) % segs
            try:
                if isinstance(a, list) and isinstance(b, list):
                    bm.faces.new((a[i], a[j], b[j], b[i]))
                elif isinstance(a, list):
                    bm.faces.new((a[i], a[j], b))
                elif isinstance(b, list):
                    bm.faces.new((a, b[j], b[i]))
            except ValueError:
                pass
    return rings


def lathe(name, profile, segs=48, xf=None, closed_profile=False, mat=None, smooth=True, recalc=True, sharp_deg=None, phase=0.0):
    bm = bmesh.new()
    lathe_bm(bm, profile, segs, xf, closed_profile, phase)
    return obj_from_bm(name, bm, mat, smooth, recalc, sharp_deg)


def box_bm(bm, size, loc=(0, 0, 0), bevel=0.0, seg=2, rot=None):
    """Box (size = (sx, sy, sz)) with optional rounded edges; rot: Matrix about the box centre; loc = centre."""
    res = bmesh.ops.create_cube(bm, size=1.0)
    verts = res["verts"]
    for v in verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    m = Matrix.Translation(loc)
    if rot is not None:
        m = m @ (rot.to_4x4() if len(rot) == 3 else rot)
    bmesh.ops.transform(bm, matrix=m, verts=verts)
    if bevel > 0:
        edges = list({e for v in verts for e in v.link_edges})
        bmesh.ops.bevel(bm, geom=edges, offset=min(bevel, min(size) * 0.49), segments=seg, affect="EDGES", profile=0.5)


def rbox(name, size, loc=(0, 0, 0), bevel=0.0, seg=2, mat=None, rot=None, sharp_deg=35):
    """Rounded box object centred at loc."""
    bm = bmesh.new()
    box_bm(bm, size, loc, bevel, seg, rot)
    return obj_from_bm(name, bm, mat, True, True, sharp_deg)


def tube_path_bm(bm, pts, prof, binormal=Vector((1, 0, 0)), cap=True, scale_fn=None):
    """Sweep a 2D profile (list of (a, b) in the (binormal, normal) frame) along 3D polyline pts (smoothed by caller)."""
    rings = []
    n = len(pts)
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized()
        u = binormal - t * binormal.dot(t)
        u.normalize()
        v = t.cross(u)  # u x v = ? want u x v = t -> v = t x u
        s = scale_fn(i / max(n - 1, 1)) if scale_fn else 1.0
        rings.append(ring_frame(p, u, v, [(a * s, b * s) for a, b in prof]))
    return loft_bm(bm, rings, cap, cap)


def catmull(points, n_per=8):
    """Catmull-Rom interpolation through points (list of Vector), returns dense list."""
    P = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(n_per):
            t = k / n_per
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(points[-1])
    return out


def smooth_curve(xs, ys, n=60, win=5, it=2):
    """Dense monotone-x interpolation of (xs, ys) with moving-average smoothing. Returns (X, Y)."""
    X = [xs[0] + (xs[-1] - xs[0]) * i / (n - 1) for i in range(n)]
    Y = []
    for x in X:
        for k in range(len(xs) - 1):
            if xs[k] <= x <= xs[k + 1]:
                t = (x - xs[k]) / (xs[k + 1] - xs[k]) if xs[k + 1] != xs[k] else 0
                Y.append(ys[k] * (1 - t) + ys[k + 1] * t)
                break
        else:
            Y.append(ys[-1])
    for _ in range(it):
        Z = Y[:]
        for i in range(1, n - 1):
            lo, hi = max(0, i - win), min(n, i + win + 1)
            Z[i] = sum(Y[lo:hi]) / (hi - lo)
        Y = Z
    return X, Y


def apply_xf(obj, matrix):
    """Bake a Matrix into mesh data (object stays at identity)."""
    obj.data.transform(matrix)
    return obj


# ------------------------------------------------------------------ text
def text_obj(name, body, size, loc=(0, 0, 0), rot=(0, 0, 0), extrude=0.0, mat=None, align="CENTER", align_y="CENTER", space_line=1.0):
    """Text converted to a mesh object (identity object transform; loc/rot baked into the mesh)."""
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = align_y
    cu.extrude = extrude
    cu.space_line = space_line
    t = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(t)
    bpy.ops.object.select_all(action="DESELECT")
    t.select_set(True)
    bpy.context.view_layer.objects.active = t
    bpy.ops.object.convert(target="MESH")
    o = bpy.context.view_layer.objects.active
    o.name = name
    if mat is not None:
        o.data.materials.clear()
        o.data.materials.append(mat)
    e = Matrix.Translation(loc) @ Matrix.Rotation(rot[2], 4, "Z") @ Matrix.Rotation(rot[1], 4, "Y") @ Matrix.Rotation(rot[0], 4, "X")
    o.data.transform(e)
    for p in o.data.polygons:
        p.use_smooth = False
    return o


_PITCH = {}


def line_pitch(size):
    """Line pitch (m) of multi-line text of a given size at space_line = 1 (measured)."""
    if size not in _PITCH:
        a = text_obj("_tp1", "I", size, align_y="TOP")
        b = text_obj("_tp2", "I\nI", size, align_y="TOP")
        ha = max(v.co.y for v in a.data.vertices) - min(v.co.y for v in a.data.vertices)
        hb = max(v.co.y for v in b.data.vertices) - min(v.co.y for v in b.data.vertices)
        _PITCH[size] = hb - ha
        bpy.data.objects.remove(a, do_unlink=True)
        bpy.data.objects.remove(b, do_unlink=True)
    return _PITCH[size]


# ------------------------------------------------------------------ materials
def _bump_noise(m, scale, strength, dist, ring_scale=None):
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n1 = nt.nodes.new("ShaderNodeTexNoise")
    n1.inputs["Scale"].default_value = scale
    n1.inputs["Detail"].default_value = 5.0
    n1.inputs["Roughness"].default_value = 0.55
    nt.links.new(tc.outputs["Object"], n1.inputs["Vector"])
    last = n1.outputs["Fac"]
    if ring_scale:
        # thumbprint-like concentric ridges, distorted by noise
        w = nt.nodes.new("ShaderNodeTexWave")
        w.wave_type = "RINGS"
        w.inputs["Scale"].default_value = ring_scale
        w.inputs["Distortion"].default_value = 6.0
        w.inputs["Detail"].default_value = 2.0
        nt.links.new(tc.outputs["Object"], w.inputs["Vector"])
        mix = nt.nodes.new("ShaderNodeMath")
        mix.operation = "MULTIPLY_ADD"
        mix.inputs[1].default_value = 0.5
        nt.links.new(w.outputs["Fac"], mix.inputs[0])
        nt.links.new(n1.outputs["Fac"], mix.inputs[2])
        last = mix.outputs["Value"]
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = dist
    nt.links.new(last, bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])


def clay(name, color, bump=0.35, scale=90.0, rough=0.62, ring=None):
    """Matte clay: soft roughness, fine noise bump plus faint thumbprint ridges (object-space texture, scale per metre)."""
    m = C.principled(name, color, rough=rough)
    b = m.node_tree.nodes["Principled BSDF"]
    try:
        b.inputs["Specular IOR Level"].default_value = 0.25
        b.inputs["Subsurface Weight"].default_value = 0.04
        b.inputs["Subsurface Radius"].default_value = (0.02, 0.01, 0.01)
        b.inputs["Subsurface Scale"].default_value = 0.01
        b.inputs["Subsurface Weight"].default_value = 0.0
    except Exception:
        pass
    _bump_noise(m, scale, bump, 0.0008, ring_scale=(ring if ring is not None else scale * 0.18))
    m["clay"] = True
    return m


def metal(name, color=(0.6, 0.6, 0.62), rough=0.3):
    return C.principled(name, color, metallic=1.0, rough=rough)


def wood_mat(name, dark=(0.07, 0.035, 0.018), light=(0.2, 0.1, 0.045), axis_scale=(14.0, 14.0, 1.1), rough=0.38):
    """Procedural walnut-like wood: wave bands along local Y distorted by noise, clear-coat lacquer."""
    m = C.principled(name, light, rough=rough, coat=0.4)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (axis_scale[0], axis_scale[1] * 0.12, axis_scale[0])
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    w = nt.nodes.new("ShaderNodeTexWave")
    w.wave_type = "BANDS"
    w.bands_direction = "X"
    w.inputs["Scale"].default_value = 18.0
    w.inputs["Distortion"].default_value = 8.0
    w.inputs["Detail"].default_value = 3.0
    w.inputs["Detail Scale"].default_value = 1.5
    nt.links.new(mp.outputs["Vector"], w.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*dark, 1)
    ramp.color_ramp.elements[1].color = (*light, 1)
    ramp.color_ramp.elements[0].position = 0.25
    ramp.color_ramp.elements[1].position = 0.8
    nt.links.new(w.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.0004
    nt.links.new(w.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def paper_mat(name, color=(0.93, 0.92, 0.88), rough=0.85):
    return C.principled(name, color, rough=rough)


# ------------------------------------------------------------------ armature / rigging
def make_armature(name, bones, coll, parent, display="OCTAHEDRAL", z_axis=(0, 0, 1)):
    """bones: list of dicts(name, head, tail, parent=None, roll_axis=None). Returns armature object (parented to `parent`)."""
    ad = bpy.data.armatures.new(name)
    ao = bpy.data.objects.new(name, ad)
    bpy.context.scene.collection.objects.link(ao)
    bpy.context.view_layer.objects.active = ao
    ao.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for b in bones:
        eb = ad.edit_bones.new(b["name"])
        eb.head = Vector(b["head"])
        eb.tail = Vector(b["tail"])
        if b.get("parent"):
            eb.parent = ad.edit_bones[b["parent"]]
        eb.align_roll(Vector(b.get("roll_axis", z_axis)))
        eb.use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    ad.display_type = display
    C.add(ao, coll, parent)
    for pb in ao.pose.bones:
        pb.rotation_mode = "XYZ"
    return ao


def bone_parent(obj, arm, bone_name):
    """Parent obj to a bone, keeping obj's current world transform at rest pose."""
    obj.parent = arm
    obj.parent_type = "BONE"
    obj.parent_bone = bone_name
    b = arm.data.bones[bone_name]
    pm = arm.matrix_world @ b.matrix_local @ Matrix.Translation((0, b.length, 0))
    obj.matrix_parent_inverse = pm.inverted()
    return obj


def add_prop(root, name, value, vmin=0.0, vmax=1.0, desc=""):
    root[name] = value
    try:
        root.id_properties_ui(name).update(min=vmin, max=vmax, soft_min=vmin, soft_max=vmax, description=desc)
    except Exception:
        pass


def driver(owner, path, index, root, props, expr):
    """Scripted driver on owner.path[index] reading root custom props `props` (list) as v0, v1, ..."""
    fc = owner.driver_add(path, index) if index is not None else owner.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    for k, pn in enumerate(props):
        var = d.variables.new()
        var.name = "v%d" % k
        var.type = "SINGLE_PROP"
        var.targets[0].id = root
        var.targets[0].data_path = '["%s"]' % pn
    d.expression = expr
    return fc


# ------------------------------------------------------------------ misc
def hook_at(name, coll, parent, loc, rot=(0, 0, 0), size=0.03):
    h = C.hook(name, coll, parent, loc=loc, rot=rot)
    h.empty_display_size = size
    return h


def parent_keep(obj, parent):
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted()
    return obj


def link_children(coll, root, objs):
    for o in objs:
        C.add(o, coll, root if o.parent is None else None)


def set_prop(root, name, value):
    """Set a root custom property and force driver re-evaluation (needed headless)."""
    root[name] = value
    root.update_tag()
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            o.update_tag()
    bpy.context.view_layer.update()


def world_pos(obj):
    return obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.translation.copy()


def reopen(blend_path):
    """Re-open the saved blend (proves it loads standalone and refreshes the depsgraph/drivers)."""
    bpy.ops.wm.open_mainfile(filepath=blend_path)


def ensure_object_mode():
    if bpy.context.object and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


# ------------------------------------------------------------------ previews
def _studio(center, radius, floor_z, floor=True):
    w = bpy.data.worlds.new("PREVIEW_world")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.57, 0.6, 1)
    bg.inputs["Strength"].default_value = 0.55
    bpy.context.scene.world = w
    objs = []
    sun = bpy.data.lights.new("PREVIEW_sun", "SUN")
    sun.energy = 1.6
    sun.angle = 0.15
    so = bpy.data.objects.new("PREVIEW_sun", sun)
    so.rotation_euler = (0.85, 0.2, 0.6)
    bpy.context.scene.collection.objects.link(so)
    objs.append(so)
    fill = bpy.data.lights.new("PREVIEW_fill", "AREA")
    d = max(radius, 1e-3) * 4.0
    fill.energy = 10 * d * d
    fill.size = max(radius, 1e-3) * 3
    fo = bpy.data.objects.new("PREVIEW_fill", fill)
    fo.location = (center[0] - radius * 2, center[1] - radius * 3, center[2] + radius * 2)
    fo.rotation_euler = (1.1, 0, -0.5)
    bpy.context.scene.collection.objects.link(fo)
    objs.append(fo)
    if floor:
        bpy.ops.mesh.primitive_plane_add(size=max(radius, 1e-3) * 40, location=(center[0], center[1], floor_z))
        fl = bpy.context.active_object
        fl.name = "PREVIEW_floor"
        fl.data.materials.append(C.principled("PREVIEW_floor_mat", (0.5, 0.52, 0.55), rough=0.95))
        objs.append(fl)
    return objs


def _items_bbox(colls):
    pts = []
    for c in colls:
        bb = C.bbox_mm(c)
        if bb:
            pts += [Vector(bb[0]) / 1000.0, Vector(bb[1]) / 1000.0]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


STD_DIRS = {
    "front": (0, -1, 0.12),
    "three_quarter": (0.8, -0.9, 0.6),
    "top": (0, -0.04, 1),
    "side": (1, 0, 0.12),
    "back": (0, 1, 0.3),
    "low": (0.6, -0.8, -0.1),
}


def render_views(colls, out_dir, prefix, views, res=(900, 675), samples=16, floor=True, fit=1.0, keep_visible=None, lens=50):
    """Render preview views of the given collections.

    views: list of dicts: name; either dir (key of STD_DIRS or Vector) [+ fit multiplier], or explicit
    target (m, Vector), cam_dir (Vector) and dist (m). Objects outside `colls` (+ keep_visible) are hidden from render.
    """
    os.makedirs(out_dir, exist_ok=True)
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = samples
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.compression = 90
    scn.view_settings.view_transform = "Standard"
    keep = set()
    for c in list(colls) + list(keep_visible or []):
        for cc in [c] + list(c.children_recursive):
            for o in cc.objects:
                keep.add(o)
    for o in scn.objects:
        if not o.name.startswith("PREVIEW_"):
            o.hide_render = o not in keep
    lo, hi = _items_bbox(colls)
    center = (lo + hi) / 2
    size = hi - lo
    radius = max(size.length / 2, 0.005)
    objs = _studio(center, radius, lo.z - 1e-4, floor)
    cam_d = bpy.data.cameras.new("PREVIEW_cam")
    cam_d.lens = lens
    cam_d.sensor_width = 36
    cam_d.clip_start = 0.0005
    cam_d.clip_end = max(radius * 200, 20)
    cam = bpy.data.objects.new("PREVIEW_cam", cam_d)
    scn.collection.objects.link(cam)
    scn.camera = cam
    out = []
    for v in views:
        if "target" in v:
            tgt = Vector(v["target"])
            d = Vector(v["cam_dir"]).normalized()
            dist = v["dist"]
        else:
            d = Vector(STD_DIRS[v["dir"]] if isinstance(v["dir"], str) else v["dir"]).normalized()
            tgt = center + Vector(v.get("offset", (0, 0, 0)))
            rr = v.get("radius", radius)
            dist = rr * 3.0 / (fit * v.get("fit", 1.0))
        cam_d.lens = v.get("lens", lens)
        cam.location = tgt + d * dist
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        path = os.path.join(out_dir, "%s_%s.png" % (prefix, v["name"]))
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        out.append(path)
    for o in objs + [cam]:
        bpy.data.objects.remove(o, do_unlink=True)
    for o in scn.objects:
        o.hide_render = False
    return out


def std_views(close=()):
    """front/three_quarter/top plus extra close-up dicts."""
    return [{"name": "front", "dir": "front"}, {"name": "three_quarter", "dir": "three_quarter"}, {"name": "top", "dir": "top"}] + list(close)


def write_family(blend_path, items, meta, preview_dir=None, previews=True):
    """Save the .blend (several ASSET_ collections allowed) and write the JSON metadata.

    items: list of dicts(asset_id, coll, root, views=[...], meta={...}). Per-item bbox and triangle counts are added.
    """
    ensure_object_mode()
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    meta = dict(meta)
    meta["items"] = {}
    for it in items:
        bb = C.bbox_mm(it["coll"])
        d = dict(it.get("meta", {}))
        d["collection"] = it["coll"].name
        d["root"] = it["root"].name
        d["bbox_mm"] = bb
        if bb:
            d["size_mm"] = [round(bb[1][k] - bb[0][k], 2) for k in range(3)]
        d["triangles"] = C.count_tris(it["coll"])
        meta["items"][it["asset_id"]] = d
    meta["triangles_total"] = sum(v["triangles"] for v in meta["items"].values())
    meta["blend_file"] = os.path.basename(blend_path)
    C.write_meta(os.path.splitext(blend_path)[0] + ".json", meta)
    pngs = []
    if previews and preview_dir:
        for it in items:
            if it.get("views"):
                pngs += render_views([it["coll"]] + it.get("extra_colls", []), preview_dir, it.get("prefix", it["asset_id"]), it["views"],
                                     floor=it.get("floor", True), fit=it.get("fit", 1.0), keep_visible=it.get("keep_visible"))
    print("FAMILY", os.path.basename(blend_path), "tris", meta["triangles_total"], "previews", len(pngs))
    return meta, pngs


def args():
    a = C.argv_after_dashes()
    return a[0] if a else "assets/components/props"


def outdirs():
    out = os.path.abspath(args())
    return out, os.path.join(out, "previews")


def contact_sheet(paths, out_path, cols=3, cell=(450, 338)):
    """Tile PNGs (downscaled) into one sheet for quick review (written outside the asset folder, e.g. scratchpad)."""
    import numpy as np
    rows = (len(paths) + cols - 1) // cols
    W, H = cell
    sheet = np.zeros((rows * H, cols * W, 4), dtype=np.float32)
    sheet[..., 3] = 1.0
    for k, p in enumerate(paths):
        im = bpy.data.images.load(p)
        im.scale(W, H)
        a = np.array(im.pixels[:], dtype=np.float32).reshape(H, W, 4)
        r, c = divmod(k, cols)
        sheet[(rows - 1 - r) * H:(rows - r) * H, c * W:(c + 1) * W] = a
        bpy.data.images.remove(im)
    img = bpy.data.images.new("sheet", cols * W, rows * H)
    img.pixels = sheet.ravel().tolist()
    img.filepath_raw = out_path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return out_path
