"""lab_office category helpers (build on scripts/assets/_common/common.py; do not edit the shared module).

All geometry helpers take MILLIMETRES and write metres into Blender (1 BU = 1 m). Every helper creates a mesh
object (rotation and scale baked into the mesh, object location = the part's pivot), links it into the current asset
collection `S.coll` and parents it to `S.root` (or the given parent). Names get the prefix `<asset_id>_` unless
`raw=True`.
"""
import math
import os
import sys
import types

import bmesh
import bpy
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import common as C  # noqa: E402

CAT = "lab_office"
S = types.SimpleNamespace(coll=None, root=None, aid=None, top_coll=None, top_root=None, dims=[], hooks=[], slots=set())
PI = math.pi
MM = 0.001


# ---------------------------------------------------------------- context
def begin(asset_id, accuracy="B"):
    C.reset()
    coll, root = C.new_asset(asset_id, accuracy)
    S.coll, S.root, S.aid = coll, root, asset_id
    S.top_coll, S.top_root = coll, root
    S.dims, S.hooks, S.slots = [], [], set()
    _MAT_CACHE.clear()
    return coll, root


def push_group(name, loc_mm=(0, 0, 0), rot_deg=(0, 0, 0), parent=None, sub=True, prefix=True):
    """Start a sub-assembly: an empty (at loc) parented to the current root, plus an optional sub-collection.
    Returns (coll, empty); S.coll/S.root are switched until pop_group()."""
    nm = (S.aid + "_" + name) if prefix else name
    e = bpy.data.objects.new(nm, None)
    e.empty_display_type = "PLAIN_AXES"
    e.empty_display_size = 0.05
    e.location = Vector(loc_mm) * MM
    e.rotation_euler = tuple(math.radians(a) for a in rot_deg)
    S.coll.objects.link(e)
    e.parent = parent or S.root
    c = S.coll
    if sub:
        c = bpy.data.collections.new("ITEM_" + name)
        S.coll.children.link(c)
        c.objects.link(e)
        S.coll.objects.unlink(e)
    S._stack = getattr(S, "_stack", [])
    S._stack.append((S.coll, S.root))
    S.coll, S.root = c, e
    return c, e


def pop_group():
    S.coll, S.root = S._stack.pop()


def _name(n, raw=False):
    if raw or n.startswith(S.aid):
        return n
    return S.aid + "_" + n


def _register(ob, parent=None, raw=False):
    S.coll.objects.link(ob)
    ob.parent = parent if parent is not None else S.root
    return ob


def dim(name, value, unit, source, level):
    """Record a dimension-table row for the metadata."""
    S.dims.append(dict(item=name, value=value, unit=unit, source=source, accuracy=level))


def custom_prop(obj, key, value, doc=None):
    obj[key] = value
    return obj


# ---------------------------------------------------------------- materials
_MAT_CACHE = {}


def _mix_bump(m, scale, strength, detail=6.0, rough_node=None, dist=0.0):
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    bm = nt.nodes.new("ShaderNodeBump")
    bm.inputs["Strength"].default_value = strength
    bm.inputs["Distance"].default_value = dist if dist else 0.002
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    nt.links.new(n.outputs["Fac"], bm.inputs["Height"])
    nt.links.new(bm.outputs["Normal"], b.inputs["Normal"])
    return n


def _color_noise(m, scale, c1, c2, mix=1.0, detail=4.0):
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].color = (*c1, 1)
    cr.color_ramp.elements[1].color = (*c2, 1)
    nt.links.new(tc.outputs["Object"], n.inputs["Vector"])
    nt.links.new(n.outputs["Fac"], cr.inputs["Fac"])
    nt.links.new(cr.outputs["Color"], b.inputs["Base Color"])
    return n


def M(key):
    """Shared lab_office material by short key (MAT_lab_office_<key>); created once per build."""
    full = "MAT_%s_%s" % (CAT, key)
    if full in _MAT_CACHE:
        return _MAT_CACHE[full]
    P = C.principled
    spec = {
        # instrument plastics and metals
        "abs_dark": lambda: P(full, (0.03, 0.032, 0.036), rough=0.42),
        "abs_black": lambda: P(full, (0.012, 0.012, 0.014), rough=0.5),
        "panel_dark": lambda: P(full, (0.045, 0.048, 0.055), rough=0.5),
        "case_grey": lambda: P(full, (0.20, 0.215, 0.235), rough=0.5),
        "case_light": lambda: P(full, (0.62, 0.64, 0.65), rough=0.5),
        "panel_light": lambda: P(full, (0.70, 0.71, 0.70), rough=0.45),
        "rubber": lambda: P(full, (0.015, 0.015, 0.017), rough=0.75),
        "knob_black": lambda: P(full, (0.02, 0.02, 0.024), rough=0.55),
        "alu": lambda: P(full, (0.72, 0.73, 0.75), metallic=1.0, rough=0.32),
        "alu_dark": lambda: P(full, (0.22, 0.23, 0.25), metallic=1.0, rough=0.38),
        "chrome": lambda: P(full, (0.86, 0.87, 0.9), metallic=1.0, rough=0.2),
        "steel": lambda: P(full, (0.5, 0.52, 0.55), metallic=1.0, rough=0.4),
        "zinc": lambda: P(full, (0.58, 0.6, 0.62), metallic=1.0, rough=0.5),
        "gold": lambda: P(full, (1.0, 0.77, 0.33), metallic=1.0, rough=0.28),
        "copper": lambda: P(full, (0.85, 0.45, 0.28), metallic=1.0, rough=0.3),
        "ptfe": lambda: P(full, (0.9, 0.88, 0.8), rough=0.45),
        "ink_white": lambda: P(full, (0.88, 0.88, 0.86), rough=0.6),
        "ink_grey": lambda: P(full, (0.45, 0.46, 0.48), rough=0.6),
        "led_green": lambda: P(full, (0.1, 0.9, 0.2), rough=0.3, emit=(0.1, 1.0, 0.2), emit_strength=2.0),
        "led_red": lambda: P(full, (0.9, 0.1, 0.05), rough=0.3, emit=(1.0, 0.1, 0.05), emit_strength=2.0),
        "led_amber": lambda: P(full, (0.95, 0.55, 0.05), rough=0.3, emit=(1.0, 0.55, 0.05), emit_strength=2.0),
        "led_blue": lambda: P(full, (0.1, 0.3, 0.95), rough=0.3, emit=(0.1, 0.35, 1.0), emit_strength=2.0),
        "key_yellow": lambda: P(full, (0.95, 0.75, 0.1), rough=0.45),
        "key_green": lambda: P(full, (0.1, 0.55, 0.15), rough=0.45),
        "key_red": lambda: P(full, (0.75, 0.08, 0.06), rough=0.45),
        "key_blue": lambda: P(full, (0.1, 0.25, 0.8), rough=0.45),
        "key_grey": lambda: P(full, (0.3, 0.31, 0.33), rough=0.5),
        "ch1": lambda: P(full, (0.9, 0.78, 0.1), rough=0.45),
        "ch2": lambda: P(full, (0.1, 0.75, 0.25), rough=0.45),
        "ch3": lambda: P(full, (0.2, 0.45, 0.95), rough=0.45),
        "ch4": lambda: P(full, (0.9, 0.2, 0.55), rough=0.45),
        "glass_dark": lambda: P(full, (0.01, 0.012, 0.015), rough=0.05, coat=0.5),
        "glass_clear": lambda: P(full, (0.75, 0.88, 0.92), rough=0.04, alpha=0.18),
        "glass_tint": lambda: P(full, (0.55, 0.75, 0.8), rough=0.04, alpha=0.3),
        "plastic_white": lambda: P(full, (0.85, 0.85, 0.82), rough=0.45),
        "plastic_cream": lambda: P(full, (0.78, 0.74, 0.62), rough=0.5),
        "plastic_red": lambda: P(full, (0.7, 0.05, 0.05), rough=0.4),
        "plastic_blue": lambda: P(full, (0.08, 0.2, 0.6), rough=0.4),
        "plastic_yellow": lambda: P(full, (0.9, 0.72, 0.08), rough=0.4),
        "plastic_green": lambda: P(full, (0.1, 0.5, 0.2), rough=0.4),
        "plastic_clear": lambda: P(full, (0.8, 0.85, 0.88), rough=0.1, alpha=0.35),
        "ceramic_white": lambda: P(full, (0.92, 0.92, 0.9), rough=0.12, coat=0.6),
        "coffee": lambda: P(full, (0.06, 0.03, 0.015), rough=0.15),
        "paper": lambda: P(full, (0.9, 0.9, 0.86), rough=0.9),
        "cover_blue": lambda: P(full, (0.08, 0.16, 0.38), rough=0.7),
        "pcb_green": lambda: P(full, (0.02, 0.2, 0.08), rough=0.35),
        "fabric_grey": lambda: P(full, (0.12, 0.13, 0.15), rough=0.95),
        "fabric_blue": lambda: P(full, (0.09, 0.17, 0.33), rough=0.95),
        "fabric_green": lambda: P(full, (0.12, 0.28, 0.16), rough=0.95),
        "wall_paint": lambda: P(full, (0.78, 0.77, 0.74), rough=0.9),
        "wall_accent": lambda: P(full, (0.18, 0.3, 0.4), rough=0.9),
        "ceiling_white": lambda: P(full, (0.88, 0.88, 0.86), rough=0.95),
        "light_panel": lambda: P(full, (0.95, 0.95, 0.9), rough=0.3, emit=(1.0, 0.97, 0.9), emit_strength=6.0),
        "plant_green": lambda: P(full, (0.08, 0.3, 0.07), rough=0.55),
        "soil": lambda: P(full, (0.07, 0.045, 0.03), rough=1.0),
        "terracotta": lambda: P(full, (0.55, 0.25, 0.13), rough=0.85),
        "exit_green": lambda: P(full, (0.02, 0.55, 0.12), rough=0.4, emit=(0.02, 0.8, 0.12), emit_strength=2.0),
        "water_blue": lambda: P(full, (0.4, 0.65, 0.95), rough=0.05, alpha=0.4),
        "laminate_white": lambda: P(full, (0.75, 0.76, 0.77), rough=0.5),
    }
    if key in spec:
        m = spec[key]()
    elif key == "laminate_esd":
        m = P(full, (0.5, 0.52, 0.52), rough=0.42)
        _color_noise(m, 400.0, (0.47, 0.49, 0.5), (0.54, 0.56, 0.56), detail=3.0)
        _mix_bump(m, 900.0, 0.03)
    elif key == "esd_mat":
        m = P(full, (0.1, 0.27, 0.19), rough=0.55)
        _color_noise(m, 60.0, (0.075, 0.22, 0.15), (0.12, 0.3, 0.21), detail=4.0)
        _mix_bump(m, 1200.0, 0.25, detail=2.0)
    elif key == "frame_powder":
        m = P(full, (0.07, 0.17, 0.32), rough=0.5)
        _mix_bump(m, 1500.0, 0.05, detail=2.0)
    elif key == "frame_grey":
        m = P(full, (0.3, 0.32, 0.35), rough=0.5)
        _mix_bump(m, 1500.0, 0.05, detail=2.0)
    elif key == "abs_textured":
        m = P(full, (0.03, 0.032, 0.036), rough=0.5)
        _mix_bump(m, 3000.0, 0.15, detail=1.0)
    elif key == "wood_table":
        m = P(full, (0.35, 0.18, 0.08), rough=0.4, coat=0.15)
        nt = m.node_tree
        b = nt.nodes["Principled BSDF"]
        tc = nt.nodes.new("ShaderNodeTexCoord")
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (1.0, 22.0, 22.0)
        w = nt.nodes.new("ShaderNodeTexWave")
        w.wave_type = "BANDS"
        w.bands_direction = "Y"
        w.inputs["Scale"].default_value = 3.0
        w.inputs["Distortion"].default_value = 4.0
        w.inputs["Detail"].default_value = 3.0
        w.inputs["Detail Scale"].default_value = 1.5
        cr = nt.nodes.new("ShaderNodeValToRGB")
        cr.color_ramp.elements[0].color = (0.30, 0.15, 0.065, 1)
        cr.color_ramp.elements[1].color = (0.46, 0.25, 0.12, 1)
        nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], w.inputs["Vector"])
        nt.links.new(w.outputs["Fac"], cr.inputs["Fac"])
        nt.links.new(cr.outputs["Color"], b.inputs["Base Color"])
    elif key == "wood_light":
        m = P(full, (0.6, 0.45, 0.28), rough=0.5)
        _color_noise(m, 25.0, (0.55, 0.4, 0.24), (0.68, 0.52, 0.33), detail=2.0)
    elif key == "carpet":
        m = P(full, (0.1, 0.12, 0.16), rough=1.0)
        _color_noise(m, 30.0, (0.085, 0.1, 0.14), (0.13, 0.15, 0.2), detail=4.0)
        _mix_bump(m, 1500.0, 0.5, detail=2.0)
    elif key == "ceiling_tile":
        m = P(full, (0.85, 0.85, 0.82), rough=0.95)
        _mix_bump(m, 800.0, 0.2, detail=3.0)
    elif key == "cubicle_fabric":
        m = P(full, (0.30, 0.34, 0.38), rough=1.0)
        _mix_bump(m, 1800.0, 0.6, detail=1.0)
    else:
        raise KeyError(key)
    _MAT_CACHE[full] = m
    S.slots.add(full)
    return m


def screen_material(name="MAT_lab_office_screen", idle=(0.004, 0.012, 0.018)):
    """Emission screen with an EMPTY image texture node 'SCREEN_IMAGE' (the assembler plugs the sequence).
    Graph: TexCoord(UV) -> Mapping -> Image Texture -> Mix(Fac = Value 'SCREEN_FAC') with a dark idle colour -> Emission
    -> Add with a glossy dark glass BSDF.  SCREEN_FAC = 0 shows the idle (powered, dark) screen so the empty slot does not
    render magenta; the assembler sets SCREEN_FAC = 1 after plugging the image."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (900, 0)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    tc.location = (-800, 0)
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.name = "SCREEN_MAPPING"
    mp.location = (-600, 0)
    im = nt.nodes.new("ShaderNodeTexImage")
    im.name = "SCREEN_IMAGE"
    im.label = "SCREEN_IMAGE (empty: assembler plugs the sequence)"
    im.interpolation = "Linear"
    im.extension = "EXTEND"
    im.location = (-380, 0)
    fac = nt.nodes.new("ShaderNodeValue")
    fac.name = "SCREEN_FAC"
    fac.label = "SCREEN_FAC (0 idle, 1 image)"
    fac.outputs[0].default_value = 0.0
    fac.location = (-380, 280)
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    mx.name = "SCREEN_MIX"
    mx.inputs["A"].default_value = (*idle, 1)
    mx.location = (-120, 100)
    em = nt.nodes.new("ShaderNodeEmission")
    em.name = "SCREEN_EMISSION"
    em.inputs["Strength"].default_value = 1.0
    em.location = (150, 120)
    gl = nt.nodes.new("ShaderNodeBsdfPrincipled")
    gl.name = "SCREEN_GLASS"
    gl.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1)
    gl.inputs["Roughness"].default_value = 0.08
    gl.inputs["Specular IOR Level"].default_value = 0.35
    gl.location = (150, -160)
    ad = nt.nodes.new("ShaderNodeAddShader")
    ad.location = (600, 0)
    nt.links.new(tc.outputs["UV"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], im.inputs["Vector"])
    nt.links.new(fac.outputs[0], mx.inputs["Factor"])
    nt.links.new(im.outputs["Color"], mx.inputs["B"])
    nt.links.new(mx.outputs["Result"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], ad.inputs[0])
    nt.links.new(gl.outputs["BSDF"], ad.inputs[1])
    nt.links.new(ad.outputs["Shader"], out.inputs["Surface"])
    m["plug_image_node"] = "SCREEN_IMAGE"
    m["plug_toggle_node"] = "SCREEN_FAC"
    m["plug_strength_node"] = "SCREEN_EMISSION"
    S.slots.add(name)
    return m


def plug_image(mat, paths, as_sequence=False, frame_offset=0):
    """Preview helper: load an image (or sequence from the first path) into the empty SCREEN_IMAGE node (not saved)."""
    node = mat.node_tree.nodes["SCREEN_IMAGE"]
    img = bpy.data.images.load(paths[0] if isinstance(paths, (list, tuple)) else paths)
    node.image = img
    for fn in ("SCREEN_FAC", "BOARD_FAC"):
        if fn in mat.node_tree.nodes:
            mat.node_tree.nodes[fn].outputs[0].default_value = 1.0
    if as_sequence:
        img.source = "SEQUENCE"
        node.image_user.frame_duration = 300
        node.image_user.frame_start = 1
        node.image_user.frame_offset = frame_offset
        node.image_user.use_auto_refresh = True
    return img


# ---------------------------------------------------------------- mesh building
def _finish(bm, name, mat, loc_mm, rot_deg=None, smooth=None, parent=None, raw=False, link=True):
    if rot_deg and any(rot_deg):
        R = Matrix.Rotation(math.radians(rot_deg[2]), 4, "Z") @ Matrix.Rotation(math.radians(rot_deg[1]), 4, "Y") \
            @ Matrix.Rotation(math.radians(rot_deg[0]), 4, "X")
        bmesh.ops.transform(bm, matrix=R, verts=bm.verts)
    bm.normal_update()
    me = bpy.data.meshes.new(_name(name, raw))
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(_name(name, raw), me)
    if mat is not None:
        me.materials.append(mat)
    ob.location = Vector(loc_mm) * MM
    if link:
        _register(ob, parent, raw)
    return ob


def box(name, size, loc, mat=None, r=0.0, seg=2, rot=None, parent=None, raw=False, sharp=False):
    """Rounded box. size/loc in mm (loc = centre). r = edge bevel radius in mm. rot = (rx, ry, rz) degrees baked."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2])) * MM
    if r > 0:
        rr = min(r, 0.45 * min(size)) * MM
        res = bmesh.ops.bevel(bm, geom=list(bm.edges), offset=rr, segments=seg, affect="EDGES", clamp_overlap=True)
        if not sharp:
            for f in res["faces"]:
                f.smooth = True
    return _finish(bm, name, mat, loc, rot, parent=parent, raw=raw)


def lathe(name, profile, loc, mat=None, axis="z", seg=32, rot=None, parent=None, raw=False, closed_profile=False):
    """Surface of revolution. profile = [(radius_mm, height_mm), ...] along the axis; (0, h) ends close the solid.
    Faces whose profile segment is perpendicular to the axis are shaded flat, the others smooth."""
    bm = bmesh.new()
    rings = []
    for (r, h) in profile:
        ring = []
        for i in range(seg):
            a = 2 * PI * i / seg
            ring.append(bm.verts.new((r * math.cos(a) * MM, r * math.sin(a) * MM, h * MM)))
        rings.append(ring)
    n = len(profile)
    segs = n if closed_profile else n - 1
    for k in range(segs):
        k2 = (k + 1) % n
        (r1, h1), (r2, h2) = profile[k], profile[k2]
        flat = abs(h1 - h2) < 1e-9
        for i in range(seg):
            j = (i + 1) % seg
            vs = [rings[k][i], rings[k][j], rings[k2][j], rings[k2][i]]
            uniq = []
            for v in vs:
                if not any((v.co - u.co).length < 1e-9 for u in uniq):
                    uniq.append(v)
            if len(uniq) >= 3:
                try:
                    f = bm.faces.new(uniq)
                    f.smooth = not flat
                except ValueError:
                    pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-8)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if axis == "x":
        bmesh.ops.transform(bm, matrix=Matrix.Rotation(PI / 2, 4, "Y"), verts=bm.verts)
    elif axis == "y":
        bmesh.ops.transform(bm, matrix=Matrix.Rotation(-PI / 2, 4, "X"), verts=bm.verts)
    elif axis == "-y":
        bmesh.ops.transform(bm, matrix=Matrix.Rotation(PI / 2, 4, "X"), verts=bm.verts)
    elif axis == "-x":
        bmesh.ops.transform(bm, matrix=Matrix.Rotation(-PI / 2, 4, "Y"), verts=bm.verts)
    return _finish(bm, name, mat, loc, rot, parent=parent, raw=raw)


def cyl(name, r, h, loc, mat=None, axis="z", seg=32, r2=None, bev=0.0, rot=None, parent=None, raw=False):
    """Cylinder / cone about the centre point `loc`; axis in z, x, y, -y ('-y' = pointing toward the viewer front).
    bev = chamfer width in mm on both rims (adds a 45 degree ring)."""
    ra = r
    rb = r if r2 is None else r2
    hh = h / 2.0
    b = min(bev, 0.45 * min(ra, rb) if min(ra, rb) > 0 else bev, 0.45 * h)
    prof = []
    if b > 0:
        prof = [(0, -hh), (ra - b, -hh), (ra, -hh + b), (rb, hh - b), (rb - b, hh), (0, hh)]
    else:
        prof = [(0, -hh), (ra, -hh), (rb, hh), (0, hh)]
    return lathe(name, prof, loc, mat, axis, seg, rot, parent, raw)


def tube(name, pts, r, loc=(0, 0, 0), mat=None, seg=8, closed=False, cap=True, parent=None, raw=False, r_end=None):
    """Swept circular tube along a polyline. pts in mm (relative to loc). r in mm (r_end tapers toward the last point)."""
    P = [Vector(p) * MM for p in pts]
    n = len(P)
    bm = bmesh.new()
    tang = []
    for i in range(n):
        if i == 0:
            t = P[1] - P[0]
        elif i == n - 1:
            t = P[-1] - P[-2]
        else:
            t = (P[i + 1] - P[i - 1])
        tang.append(t.normalized())
    up = Vector((0, 0, 1)) if abs(tang[0].z) < 0.9 else Vector((1, 0, 0))
    nrm = tang[0].cross(up).normalized()
    rings = []
    for i in range(n):
        t = tang[i]
        nrm = (nrm - t * nrm.dot(t))
        if nrm.length < 1e-9:
            nrm = t.orthogonal()
        nrm.normalize()
        bn = t.cross(nrm).normalized()
        rr = r * MM if r_end is None else (r + (r_end - r) * i / max(n - 1, 1)) * MM
        ring = []
        for k in range(seg):
            a = 2 * PI * k / seg
            ring.append(bm.verts.new(P[i] + (nrm * math.cos(a) + bn * math.sin(a)) * rr))
        rings.append(ring)
    rng = range(n) if closed else range(n - 1)
    for i in rng:
        i2 = (i + 1) % n
        for k in range(seg):
            k2 = (k + 1) % seg
            f = bm.faces.new([rings[i][k], rings[i][k2], rings[i2][k2], rings[i2][k]])
            f.smooth = True
    if cap and not closed:
        for ring, rev in ((rings[0], True), (rings[-1], False)):
            c = bm.verts.new(sum((v.co for v in ring), Vector()) / seg)
            for k in range(seg):
                k2 = (k + 1) % seg
                vs = [c, ring[k], ring[k2]] if rev else [c, ring[k2], ring[k]]
                bm.faces.new(vs)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _finish(bm, name, mat, loc, None, parent=parent, raw=raw)


def uvquad(name, w, h, loc, mat=None, facing="-y", parent=None, raw=False, flip_u=False, rot=None):
    """Plane with UV 0..1 across it. facing '-y' (default, normal toward the viewer front), 'z' (up), 'y', 'x', '-x'.
    For '-y': u runs +X, v runs +Z. For 'z': u runs +X, v runs +Y."""
    bm = bmesh.new()
    hw, hh = w / 2.0 * MM, h / 2.0 * MM
    if facing == "-y":
        co = [(-hw, 0, -hh), (hw, 0, -hh), (hw, 0, hh), (-hw, 0, hh)]
    elif facing == "y":
        co = [(hw, 0, -hh), (-hw, 0, -hh), (-hw, 0, hh), (hw, 0, hh)]
    elif facing == "z":
        co = [(-hw, -hh, 0), (hw, -hh, 0), (hw, hh, 0), (-hw, hh, 0)]
    elif facing == "x":
        co = [(0, -hw, -hh), (0, hw, -hh), (0, hw, hh), (0, -hw, hh)]
    elif facing == "-x":
        co = [(0, hw, -hh), (0, -hw, -hh), (0, -hw, hh), (0, hw, hh)]
    else:
        raise ValueError(facing)
    vs = [bm.verts.new(c) for c in co]
    f = bm.faces.new(vs)
    f.smooth = False
    uvl = bm.loops.layers.uv.new("UVMap")
    uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    if flip_u:
        uvs = [(1, 0), (0, 0), (0, 1), (1, 1)]
    for loop, uv in zip(f.loops, uvs):
        loop[uvl].uv = uv
    return _finish(bm, name, mat, loc, rot, parent=parent, raw=raw)


def dup(src, name, loc, parent=None, raw=False, rot_z_deg=None):
    """Linked duplicate (shares mesh data) at a new location (mm)."""
    ob = bpy.data.objects.new(_name(name, raw), src.data)
    ob.location = Vector(loc) * MM
    if rot_z_deg:
        ob.rotation_euler = (0, 0, math.radians(rot_z_deg))
    _register(ob, parent, raw)
    return ob


def hook(name, loc_mm, parent=None, rot_deg=(0, 0, 0), size=0.04, note=""):
    """HOOK_<name> empty. loc_mm is in the PARENT's local frame (root or a part with identity rotation: parent origin + loc)."""
    h = bpy.data.objects.new("HOOK_" + name, None)
    h.empty_display_type = "ARROWS"
    h.empty_display_size = size
    h.location = Vector(loc_mm) * MM
    h.rotation_euler = tuple(math.radians(a) for a in rot_deg)
    S.coll.objects.link(h)
    h.parent = parent if parent is not None else S.root
    S.hooks.append(dict(name="HOOK_" + name, parent=h.parent.name, location_mm=list(loc_mm), note=note))
    return h


def join_group(objs, name, mat=None, parent=None, raw=False):
    """Merge several mesh objects (same material slot set) into one object whose origin is the first object's origin."""
    bm = bmesh.new()
    o0 = objs[0]
    for o in objs:
        me = o.data
        d = o.location - o0.location
        mp = {}
        for v in me.vertices:
            mp[v.index] = bm.verts.new(v.co + d)
        for p in me.polygons:
            try:
                f = bm.faces.new([mp[i] for i in p.vertices])
                f.smooth = p.use_smooth
            except ValueError:
                pass
    bm.normal_update()
    me = bpy.data.meshes.new(_name(name, raw))
    bm.to_mesh(me)
    bm.free()
    nob = bpy.data.objects.new(_name(name, raw), me)
    mm_ = mat if mat is not None else (o0.data.materials[0] if o0.data.materials else None)
    if mm_ is not None:
        me.materials.append(mm_)
    nob.location = o0.location
    par = parent or o0.parent
    for o in objs:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        bpy.data.objects.remove(o)
    S.coll.objects.link(nob)
    nob.parent = par
    return nob


def labels(name, items, mat, extrude_mm=0.0, facing="-y", parent=None):
    """Flat text labels merged into one mesh object. items = [(text, size_mm, (x, y, z) mm, align)]; text lies in the
    XZ plane facing -Y (reads left to right, up = +Z). Uses the built-in font (no external files)."""
    bm = bmesh.new()
    for (txt, size, loc, *al) in items:
        align = al[0] if al else "CENTER"
        cu = bpy.data.curves.new("tmp_lbl", "FONT")
        cu.body = txt
        cu.size = size * MM
        cu.align_x = align
        cu.align_y = "CENTER"
        cu.extrude = extrude_mm * MM
        ob = bpy.data.objects.new("tmp_lbl", cu)
        bpy.context.scene.collection.objects.link(ob)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg)
        me = bpy.data.meshes.new_from_object(ev)
        rot = Matrix.Rotation(PI / 2, 4, "X") if facing == "-y" else Matrix.Identity(4)
        mv = {}
        T = Matrix.Translation(Vector(loc) * MM) @ rot
        for v in me.vertices:
            mv[v.index] = bm.verts.new(T @ v.co)
        for p in me.polygons:
            try:
                f = bm.faces.new([mv[i] for i in p.vertices])
                f.smooth = False
            except ValueError:
                pass
        bpy.data.meshes.remove(me)
        bpy.context.scene.collection.objects.unlink(ob)
        bpy.data.objects.remove(ob)
        bpy.data.curves.remove(cu)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _finish(bm, name, mat, (0, 0, 0), None, parent=parent)


# ---------------------------------------------------------------- shared small parts
def knob(name, loc, r, h, mat=None, cap_mat=None, axis="-y", skirt=True, pointer=True, parent=None, seg=32):
    """Rotary knob; axis '-y' = pointing toward the viewer. loc = centre of the base on the panel surface (mm, parent frame).
    Returns the knob object (origin at the base centre; spin it about its own axis: local Y for '-y', local Z for 'z')."""
    mat = mat or M("knob_black")
    if skirt:
        prof = [(0, 0), (r * 1.18, 0), (r * 1.18, h * 0.12), (r * 1.0, h * 0.15), (r * 1.0, h * 0.62), (r * 0.94, h * 0.8),
                (r * 0.9, h), (0, h)]
    else:
        prof = [(0, 0), (r * 1.0, 0), (r * 1.0, h * 0.62), (r * 0.94, h * 0.8), (r * 0.9, h), (0, h)]
    k = lathe(name, prof, loc, mat, axis=axis, seg=seg, parent=parent)
    if pointer:
        pm = cap_mat or M("ink_white")
        if axis == "-y":
            box(name + "_mark", (r * 0.14, 0.3, r * 0.7), (0, -h - 0.1, r * 0.5), pm, parent=k)
        elif axis == "z":
            box(name + "_mark", (r * 0.14, r * 0.7, 0.3), (0, r * 0.5, h + 0.1), pm, parent=k)
    return k


def bnc_jack(name, loc, mat_metal=None, parent=None, seg=24):
    """Panel-mount BNC female jack facing -Y. loc = centre on the panel surface (mm, parent frame).
    Bayonet barrel OD 9.6 mm (public BNC practice), ID 7.2 mm, centre pin 1.6 mm, flange OD 16 mm (estimate)."""
    mm_ = mat_metal or M("chrome")
    barrel = lathe(name, [(3.6, 0), (4.8, 0), (4.8, 13.0), (4.5, 13.6), (3.7, 13.6), (3.6, 12.8)], loc, mm_, axis="-y",
                   seg=seg, parent=parent)
    cyl(name + "_die", 3.2, 10.0, (0, -4.0, 0), M("ptfe"), axis="-y", seg=seg, parent=barrel)
    cyl(name + "_pin", 0.8, 12.0, (0, -8.0, 0), M("gold"), axis="-y", seg=12, parent=barrel)
    for sgn in (-1, 1):
        cyl(name + "_stud", 0.9, 2.6, (sgn * 5.4, -9.2, 0), mm_, axis="x", seg=8, parent=barrel)
    cyl(name + "_flange", 8.0, 1.2, (0, -0.6, 0), mm_, axis="-y", seg=seg, bev=0.3, parent=barrel)
    return barrel


# ---------------------------------------------------------------- previews
def _bbox(coll):
    bb = C.bbox_mm(coll)
    lo, hi = Vector(bb[0]) / 1000.0, Vector(bb[1]) / 1000.0
    return lo, hi


def render_views(coll, out_dir, asset_id, views, floor=True, floor_mat="PREVIEW", hide=(), res=(900, 675), samples=16,
                 lights="outside", world=0.7, sun=2.5, lens=50.0, sun_shadow=True):
    """Render preview PNGs.  views: list of (view_name, spec) where spec is a dict with either
    {dir:(x,y,z), fit:f}  (camera on a ray from the bbox centre, distance fit * bbox radius * 3.2)
    or {loc:(x,y,z) metres, target:(x,y,z) metres, lens:mm}.  hide: names of objects hidden from render in all views;
    a spec may also hold hide=[names] for that view only."""
    os.makedirs(out_dir, exist_ok=True)
    bpy.context.view_layer.update()
    lo, hi = _bbox(coll)
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 0.01)
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = samples
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.compression = 90
    scn.view_settings.view_transform = "Standard"
    keep = {o for c in [coll] + list(coll.children_recursive) for o in c.objects}
    for o in scn.objects:
        if o not in keep:
            o.hide_render = True
    # world
    w = bpy.data.worlds.new("PREVIEW_world")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.66, 0.72, 0.8, 1)
    bg.inputs["Strength"].default_value = world
    scn.world = w
    extra = []
    if lights in ("outside", "sun"):
        s = bpy.data.lights.new("PREVIEW_sun", "SUN")
        s.energy = sun
        s.angle = 0.08
        s.use_shadow = sun_shadow
        so = bpy.data.objects.new("PREVIEW_sun", s)
        so.rotation_euler = (0.95, 0.25, 0.6)
        scn.collection.objects.link(so)
        extra.append(so)
    if lights == "outside":
        f = bpy.data.lights.new("PREVIEW_fill", "AREA")
        d = radius * 4.0
        f.energy = 30 * d * d
        f.size = radius * 3
        fo = bpy.data.objects.new("PREVIEW_fill", f)
        fo.location = (center.x - radius * 2, center.y - radius * 3, center.z + radius * 2)
        fo.rotation_euler = (1.1, 0, -0.5)
        scn.collection.objects.link(fo)
        extra.append(fo)
    if floor:
        bpy.ops.mesh.primitive_plane_add(size=max(radius, 1.0) * 24, location=(center.x, center.y, 0.0))
        fl = bpy.context.active_object
        fl.name = "PREVIEW_floor"
        fm = C.principled("PREVIEW_floor_mat", (0.5, 0.52, 0.5), rough=0.9)
        fl.data.materials.append(fm)
        extra.append(fl)
    cam_d = bpy.data.cameras.new("PREVIEW_cam")
    cam_d.sensor_width = 36
    cam_d.clip_start = 0.002
    cam_d.clip_end = max(radius * 200, 50)
    cam = bpy.data.objects.new("PREVIEW_cam", cam_d)
    scn.collection.objects.link(cam)
    scn.camera = cam
    cons = cam.constraints.new("TRACK_TO")
    cons.track_axis = "TRACK_NEGATIVE_Z"
    cons.up_axis = "UP_Y"
    tgt = bpy.data.objects.new("PREVIEW_tgt", None)
    scn.collection.objects.link(tgt)
    cons.target = tgt
    out = []
    for (vn, sp) in views:
        for hn in list(hide) + list(sp.get("hide", [])):
            o = bpy.data.objects.get(hn)
            if o:
                o.hide_render = True
        if "loc" in sp:
            cam.location = Vector(sp["loc"])
            tgt.location = Vector(sp["target"])
            cam_d.lens = sp.get("lens", lens)
        else:
            dvec = Vector(sp["dir"]).normalized()
            fit = sp.get("fit", 1.0)
            tgt.location = Vector(sp.get("target", center))
            cam_d.lens = sp.get("lens", lens)
            cam.location = tgt.location + dvec * radius * 3.2 / fit
        path = os.path.join(out_dir, "%s_%s.png" % (asset_id, vn))
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        out.append(path)
        for hn in list(sp.get("hide", [])):
            o = bpy.data.objects.get(hn)
            if o and hn not in hide:
                o.hide_render = False
    return out


STD_VIEWS = [("front", dict(dir=(0, -1, 0.12))), ("three_quarter", dict(dir=(0.8, -0.9, 0.6))),
             ("top", dict(dir=(0, -0.04, 1)))]


def tex_paths(kind, n=1):
    base = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "assets",
                                        "generated_textures"))
    d = {"eye_s1": "eye_s1", "eye_s2": "eye_s2", "graph_s2": "graph_s2", "spec_s5": "spec_s5"}[kind]
    return [os.path.join(base, d, "%s_%04d.png" % (d, i + 1)) for i in range(n)]


def write_meta_extended(blend_path, extra):
    """Merge extra keys into the JSON written by C.finish."""
    import json
    jp = os.path.splitext(blend_path)[0] + ".json"
    with open(jp) as f:
        m = json.load(f)
    m.update(extra)
    C.write_meta(jp, m)


def finish(asset_id, out_dir, meta, accuracy_note=""):
    """Save blend + base JSON (no previews) and add dims/hooks/slots; returns blend path and meta."""
    bp = os.path.join(out_dir, asset_id + ".blend")
    bpy.context.view_layer.update()
    base = dict(meta)
    base["dimension_table"] = S.dims
    base["hooks"] = S.hooks
    base["material_slots"] = sorted(S.slots)
    C.finish(asset_id, bp, S.top_coll, base, preview_dir=None)
    return bp


# ---------------------------------------------------------------- curves
def catmull(pts, n=6, closed=False):
    """Catmull-Rom subdivision of a polyline (list of 3-tuples); n points per span."""
    P = [Vector(p) for p in pts]
    out = []
    m = len(P)
    rng = range(m) if closed else range(m - 1)
    for i in rng:
        p0 = P[(i - 1) % m] if (closed or i > 0) else P[0]
        p1 = P[i]
        p2 = P[(i + 1) % m]
        p3 = P[(i + 2) % m] if (closed or i + 2 < m) else P[-1]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            v = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
            out.append(tuple(v))
    if not closed:
        out.append(tuple(P[-1]))
    return out


def circle_pts(r, n=48, center=(0, 0, 0), plane="xy", z=0.0):
    pts = []
    for i in range(n):
        a = 2 * PI * i / n
        c, s = r * math.cos(a), r * math.sin(a)
        if plane == "xy":
            pts.append((center[0] + c, center[1] + s, center[2]))
        elif plane == "xz":
            pts.append((center[0] + c, center[1], center[2] + s))
        else:
            pts.append((center[0], center[1] + c, center[2] + s))
    return pts


def multi_box(name, specs, mat=None, r=0.0, seg=1, parent=None, origin=(0, 0, 0)):
    """Many boxes merged into ONE mesh object (cheap arrays: vents, key grids, slats).
    specs = [(size_mm(3), loc_mm(3))] relative to `origin` (the object origin, mm)."""
    bm = bmesh.new()
    for (size, loc) in specs:
        T = Matrix.Translation((Vector(loc) - Vector(origin)) * MM) @ Matrix.Diagonal(Vector((*(Vector(size) * MM), 1.0)))
        res = bmesh.ops.create_cube(bm, size=1.0, matrix=T)
        if r > 0:
            vs = res["verts"]
            es = list({e for v in vs for e in v.link_edges})
            rr = min(r, 0.45 * min(size)) * MM
            rb = bmesh.ops.bevel(bm, geom=es, offset=rr, segments=seg, affect="EDGES", clamp_overlap=True)
            for f in rb["faces"]:
                f.smooth = True
    return _finish(bm, name, mat, origin, None, parent=parent)
