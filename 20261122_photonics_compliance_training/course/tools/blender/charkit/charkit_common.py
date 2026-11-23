"""Shared helpers for asset build scripts (Blender 4.2, headless). Import with:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
    import common as C

Conventions are in assets/ASSET_SPEC.md. 1 Blender unit = 1 m; use C.mm() for millimetres.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector


def mm(x):
    return x * 0.001


def um(x):
    return x * 1e-6


def argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


# ---------------------------------------------------------------- scene / asset root
def reset():
    """Empty factory scene (no default cube/camera/light)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def new_asset(asset_id, accuracy="B", version="1.0.0"):
    """Create the root collection ASSET_<id> (linked to the scene) and a root empty at the origin.

    Put every object of the asset in the returned collection (use C.add(obj, coll)).
    accuracy: A = from spec/drawing, B = from photo/teardown/public dims, C = plausible generic.
    """
    coll = bpy.data.collections.new("ASSET_" + asset_id)
    bpy.context.scene.collection.children.link(coll)
    root = bpy.data.objects.new("ROOT_" + asset_id, None)
    root.empty_display_type = "PLAIN_AXES"
    coll.objects.link(root)
    root["asset_id"] = asset_id
    root["accuracy_level"] = accuracy
    root["asset_version"] = version
    return coll, root


def add(obj, coll, parent=None):
    """Move/link an object into the asset collection (unlinking it from other collections)."""
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


def sub_collection(coll, name):
    c = bpy.data.collections.new(name)
    coll.children.link(c)
    return c


def hook(name, coll, parent, loc=(0, 0, 0), rot=(0, 0, 0)):
    """Animation hook empty named HOOK_<name> (documented in the asset meta)."""
    h = bpy.data.objects.new("HOOK_" + name, None)
    h.empty_display_type = "ARROWS"
    h.empty_display_size = 0.05
    h.location = loc
    h.rotation_euler = rot
    coll.objects.link(h)
    h.parent = parent
    return h


# ---------------------------------------------------------------- materials
def principled(name, base=(0.8, 0.8, 0.8), metallic=0.0, rough=0.5, emit=None, emit_strength=0.0, alpha=1.0,
               coat=0.0, transmission=0.0, ior=1.45):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base[:3], 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    b.inputs["IOR"].default_value = ior
    if coat:
        b.inputs["Coat Weight"].default_value = coat
    if transmission:
        b.inputs["Transmission Weight"].default_value = transmission
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit[:3], 1)
        b.inputs["Emission Strength"].default_value = emit_strength
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        try:
            m.surface_render_method = "BLENDED"
        except Exception:
            pass
    return m


def assign(obj, mat):
    if obj.data is None or not hasattr(obj.data, "materials"):
        return obj
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj


# ---------------------------------------------------------------- mesh primitives (return objects, not yet in the asset collection)
def _active():
    return bpy.context.active_object


def cube(name, size, loc=(0, 0, 0), mat=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    o = _active()
    o.name = name
    o.scale = size
    bpy.ops.object.transform_apply(scale=True)
    if mat:
        assign(o, mat)
    return o


def cylinder(name, radius, depth, loc=(0, 0, 0), axis="Z", verts=32, mat=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc)
    o = _active()
    o.name = name
    if axis == "X":
        o.rotation_euler = (0, math.pi / 2, 0)
    elif axis == "Y":
        o.rotation_euler = (math.pi / 2, 0, 0)
    bpy.ops.object.transform_apply(rotation=True)
    if mat:
        assign(o, mat)
    return o


def sphere(name, radius, loc=(0, 0, 0), scale=(1, 1, 1), seg=32, rings=16, smooth=True, mat=None):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, segments=seg, ring_count=rings, location=loc)
    o = _active()
    o.name = name
    o.scale = scale
    bpy.ops.object.transform_apply(scale=True)
    if smooth:
        for p in o.data.polygons:
            p.use_smooth = True
    if mat:
        assign(o, mat)
    return o


def bevel(obj, width, segments=2, limit_angle=0.52):
    m = obj.modifiers.new("Bevel", "BEVEL")
    m.width = width
    m.segments = segments
    m.limit_method = "ANGLE"
    m.angle_limit = limit_angle
    return m


def shade_smooth(obj, auto_angle=0.52):
    for p in obj.data.polygons:
        p.use_smooth = True
    try:
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = auto_angle
    except Exception:
        pass


def join(objs, name):
    """Join objects into the first one (all must be meshes); returns the joined object."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.active_object
    o.name = name
    return o


def text_mesh(name, body, size, loc=(0, 0, 0), rot=(0, 0, 0), extrude=0.0, mat=None, align="CENTER"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.extrude = extrude
    o = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = rot
    if mat:
        o.data.materials.append(mat)
    return o


# ---------------------------------------------------------------- bounding box / metadata
def bbox_mm(coll):
    """World-space bounding box of all mesh objects in a collection tree, in mm: ((x0,y0,z0),(x1,y1,z1))."""
    pts = []

    def walk(c):
        for o in c.objects:
            if o.type in {"MESH", "CURVE", "FONT"}:
                for v in o.bound_box:
                    pts.append(o.matrix_world @ Vector(v))
        for ch in c.children:
            walk(ch)
    walk(coll)
    if not pts:
        return None
    xs, ys, zs = [p.x for p in pts], [p.y for p in pts], [p.z for p in pts]
    return ((min(xs) * 1000, min(ys) * 1000, min(zs) * 1000), (max(xs) * 1000, max(ys) * 1000, max(zs) * 1000))


def count_tris(coll):
    n = 0

    def walk(c):
        nonlocal n
        for o in c.objects:
            if o.type == "MESH":
                n += sum(len(p.vertices) - 2 for p in o.data.polygons)
        for ch in c.children:
            walk(ch)
    walk(coll)
    return n


def write_meta(path, meta):
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------- save + preview
def save(blend_path):
    os.makedirs(os.path.dirname(blend_path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    return blend_path


def _studio(center, radius):
    """Temporary studio: floor, sun, fill, world. Returns list of created objects."""
    objs = []
    w = bpy.data.worlds.new("PREVIEW_world")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.7, 0.78, 0.9, 1)
    bg.inputs["Strength"].default_value = 0.6
    bpy.context.scene.world = w
    sun = bpy.data.lights.new("PREVIEW_sun", "SUN")
    sun.energy = 2.5
    so = bpy.data.objects.new("PREVIEW_sun", sun)
    so.rotation_euler = (0.9, 0.3, 0.7)
    bpy.context.scene.collection.objects.link(so)
    objs.append(so)
    fill = bpy.data.lights.new("PREVIEW_fill", "AREA")
    _d = max(radius, 1e-3) * 4.0
    fill.energy = 25 * _d * _d
    fill.size = max(radius, 1e-3) * 3
    fo = bpy.data.objects.new("PREVIEW_fill", fill)
    fo.location = (center[0] - radius * 2, center[1] - radius * 3, center[2] + radius * 2)
    fo.rotation_euler = (1.1, 0, -0.5)
    bpy.context.scene.collection.objects.link(fo)
    objs.append(fo)
    bpy.ops.mesh.primitive_plane_add(size=max(radius, 1e-3) * 20, location=(center[0], center[1], center[2] - radius))
    fl = bpy.context.active_object
    fl.name = "PREVIEW_floor"
    m = principled("PREVIEW_floor_mat", (0.55, 0.57, 0.52), rough=0.9)
    fl.data.materials.append(m)
    objs.append(fl)
    return objs


def preview(coll, out_dir, asset_id, views=("front", "three_quarter", "top"), res=(900, 675), samples=16,
            fit=1.0, floor=True):
    """Render small preview PNGs of the asset collection (does not modify the saved .blend: call after save()).

    Objects outside the asset collection that were created for building are hidden from render.
    """
    os.makedirs(out_dir, exist_ok=True)
    bb = bbox_mm(coll)
    if bb is None:
        return []
    lo, hi = Vector(bb[0]) / 1000.0, Vector(bb[1]) / 1000.0
    center = (lo + hi) / 2
    size = (hi - lo)
    radius = max(size.length / 2, 0.01)
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = samples
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.image_settings.file_format = "PNG"
    scn.view_settings.view_transform = "Standard"
    keep = {o for c in [coll] + list(coll.children_recursive) for o in c.objects}
    for o in bpy.context.scene.objects:
        if o not in keep:
            o.hide_render = True
    st = _studio(center, radius) if floor else []
    cam_d = bpy.data.cameras.new("PREVIEW_cam")
    cam_d.sensor_fit = "AUTO"
    cam_d.clip_end = max(radius * 200, 10)
    cam_d.clip_start = max(radius * 0.01, 1e-4)
    cam = bpy.data.objects.new("PREVIEW_cam", cam_d)
    scn.collection.objects.link(cam)
    scn.camera = cam
    cons = cam.constraints.new("TRACK_TO")
    cons.track_axis = "TRACK_NEGATIVE_Z"
    cons.up_axis = "UP_Y"
    tgt = bpy.data.objects.new("PREVIEW_tgt", None)
    tgt.location = center
    scn.collection.objects.link(tgt)
    cons.target = tgt
    dist = radius * 2.4 / fit
    dirs = {"front": Vector((0, -1, 0.15)), "three_quarter": Vector((0.8, -0.9, 0.6)), "top": Vector((0, -0.05, 1)),
            "back": Vector((0, 1, 0.3)), "side": Vector((1, 0, 0.2)), "low": Vector((0.6, -0.8, -0.1))}
    cam_d.lens = 50
    cam_d.sensor_width = 36
    out = []
    for v in views:
        d = dirs[v].normalized()
        cam.location = center + d * radius * 3.2 / fit
        path = os.path.join(out_dir, "%s_%s.png" % (asset_id, v))
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        out.append(path)
    return out


def finish(asset_id, blend_path, coll, meta, preview_dir=None, views=("front", "three_quarter", "top"), res=(900, 675)):
    """Common ending: save .blend, write JSON meta next to it (with bbox and tri count), render previews."""
    bb = bbox_mm(coll)
    meta = dict(meta)
    meta["asset_id"] = asset_id
    meta["bbox_mm"] = bb
    if bb:
        meta["size_mm"] = [round(bb[1][k] - bb[0][k], 3) for k in range(3)]
    meta["triangles"] = count_tris(coll)
    save(blend_path)
    write_meta(os.path.splitext(blend_path)[0] + ".json", meta)
    pngs = []
    if preview_dir:
        pngs = preview(coll, preview_dir, asset_id, views=views, res=res)
    print("ASSET", asset_id, "tris", meta["triangles"], "size_mm", meta.get("size_mm"), "previews", len(pngs))
    return meta
