"""Datacenter category helpers: materials, numpy-free MeshBuilder, curves, perforation shader, previews, metadata.

Import from a build script in this folder:
    import sys, os
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(HERE, "..", "_common")); sys.path.insert(0, HERE)
    import common as C, dc_common as D
All lengths in metres (use C.mm()).
"""
import json
import math
import os
import sys

import bpy
import bmesh
from mathutils import Vector

mm = lambda x: x * 0.001

# standards (sources recorded in each asset's metadata)
RU = mm(44.45)          # EIA-310 rack unit
OU = mm(48.0)           # OCP Open Rack OpenU
TILE = 0.6              # raised floor tile (600 x 600 mm)


# ------------------------------------------------------------------ args
def parse_args():
    """argv after '--': first = output dir, rest key=value overrides (floats/ints/str)."""
    a = C.argv_after_dashes() if "C" in globals() else sys.argv[sys.argv.index("--") + 1:]
    out = a[0]
    kv = {}
    for s in a[1:]:
        if "=" in s:
            k, v = s.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            kv[k] = v
    return out, kv


# ------------------------------------------------------------------ materials
_MC = {}


def reset_mats():
    _MC.clear()


def M(name, base=(0.5, 0.5, 0.5), metallic=0.0, rough=0.5, emit=None, emit_strength=0.0, alpha=1.0, coat=0.0,
      transmission=0.0):
    """Cached principled material named MAT_datacenter_<name>."""
    if name in _MC:
        return _MC[name]
    m = C.principled("MAT_datacenter_" + name, base, metallic, rough, emit, emit_strength, alpha, coat, transmission)
    _MC[name] = m
    return m


def lib():
    """Standard material palette (lazily created, cached)."""
    return dict(
        frame=M("rack_frame", (0.035, 0.036, 0.04), 0.6, 0.45),
        panel=M("rack_panel", (0.05, 0.052, 0.058), 0.5, 0.5),
        steel=M("steel_zinc", (0.62, 0.64, 0.66), 0.9, 0.38),
        dark_steel=M("steel_dark", (0.12, 0.125, 0.135), 0.8, 0.45),
        alu=M("alu_anodized", (0.72, 0.73, 0.75), 1.0, 0.32),
        black=M("plastic_black", (0.02, 0.02, 0.022), 0.0, 0.45),
        grey=M("plastic_grey", (0.16, 0.165, 0.17), 0.0, 0.5),
        tray_front=M("tray_front", (0.04, 0.042, 0.046), 0.7, 0.4),
        accent=M("accent_green", (0.28, 0.5, 0.05), 0.0, 0.35),
        copper=M("copper", (0.95, 0.45, 0.2), 1.0, 0.28),
        brass=M("brass", (0.85, 0.65, 0.25), 1.0, 0.3),
        gold=M("gold_plating", (0.95, 0.72, 0.28), 1.0, 0.25),
        nickel=M("nickel", (0.7, 0.7, 0.72), 1.0, 0.25),
        hose=M("hose_epdm", (0.015, 0.015, 0.017), 0.0, 0.55),
        qd_blue=M("qd_blue", (0.04, 0.2, 0.75), 0.0, 0.35),
        qd_red=M("qd_red", (0.75, 0.05, 0.04), 0.0, 0.35),
        pcb=M("pcb_soldermask", (0.02, 0.1, 0.05), 0.0, 0.35),
        mold=M("mold_black", (0.012, 0.012, 0.014), 0.0, 0.4),
        led_g=M("led_green", (0.1, 1, 0.2), 0.0, 0.3, (0.1, 1, 0.2), 6.0),
        led_a=M("led_amber", (1, 0.55, 0.05), 0.0, 0.3, (1, 0.55, 0.05), 6.0),
        led_b=M("led_blue", (0.1, 0.4, 1), 0.0, 0.3, (0.1, 0.4, 1), 6.0),
        label=M("label_white", (0.9, 0.9, 0.88), 0.0, 0.6),
        text=M("text_dark", (0.02, 0.02, 0.02), 0.0, 0.7),
        coolant=M("coolant_glass", (0.5, 0.8, 0.9), 0.0, 0.1, alpha=0.4),
    )


def perforated_material(name, base=(0.04, 0.042, 0.046), metallic=0.7, rough=0.45, pitch=0.008, radius=0.0027,
                        scale_axes=(1, 1, 0)):
    """Principled surface with a round-hole perforation (dithered alpha, Object-space coordinates).

    pitch/radius in metres (hole pattern in object-space axes given by scale_axes mask).
    """
    if name in _MC:
        return _MC[name]
    m = bpy.data.materials.new("MAT_datacenter_" + name)
    m.use_nodes = True
    try:
        m.surface_render_method = "DITHERED"
    except Exception:
        pass
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*base, 1)
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = rough
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 / pitch * scale_axes[0], 1.0 / pitch * scale_axes[1],
                                        1.0 / pitch * scale_axes[2])
    fr = nt.nodes.new("ShaderNodeVectorMath")
    fr.operation = "FRACTION"
    sb = nt.nodes.new("ShaderNodeVectorMath")
    sb.operation = "SUBTRACT"
    sb.inputs[1].default_value = (0.5 * scale_axes[0], 0.5 * scale_axes[1], 0.5 * scale_axes[2])
    ln = nt.nodes.new("ShaderNodeVectorMath")
    ln.operation = "LENGTH"
    cmp_ = nt.nodes.new("ShaderNodeMath")
    cmp_.operation = "GREATER_THAN"
    cmp_.inputs[1].default_value = radius / pitch
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], fr.inputs[0])
    nt.links.new(fr.outputs["Vector"], sb.inputs[0])
    nt.links.new(sb.outputs["Vector"], ln.inputs[0])
    nt.links.new(ln.outputs["Value"], cmp_.inputs[0])
    nt.links.new(cmp_.outputs["Value"], bsdf.inputs["Alpha"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    _MC[name] = m
    return m


# ------------------------------------------------------------------ mesh builder
class MB:
    """Collects boxes, cylinders and quads into one mesh with per-face material index.

    Use .box(center, size), .cyl(center, r, h, axis), .quad(a,b,c,d); .obj(name, mats) returns an object (not linked).
    """

    def __init__(self):
        self.v = []
        self.f = []
        self.mi = []

    def _add(self, verts, faces, m=0):
        o = len(self.v)
        self.v.extend(verts)
        for f in faces:
            self.f.append(tuple(o + i for i in f))
            self.mi.append(m)

    def box(self, c, s, m=0):
        x, y, z = c
        dx, dy, dz = s[0] / 2, s[1] / 2, s[2] / 2
        v = [(x - dx, y - dy, z - dz), (x + dx, y - dy, z - dz), (x + dx, y + dy, z - dz), (x - dx, y + dy, z - dz),
             (x - dx, y - dy, z + dz), (x + dx, y - dy, z + dz), (x + dx, y + dy, z + dz), (x - dx, y + dy, z + dz)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        self._add(v, f, m)

    def boxmm(self, c, s, m=0):
        self.box(tuple(a * 0.001 for a in c), tuple(a * 0.001 for a in s), m)

    def cyl(self, c, r, h, axis="z", seg=16, m=0, r2=None):
        x, y, z = c
        r2 = r if r2 is None else r2
        v = []
        for end, rr in ((-1, r), (1, r2)):
            for i in range(seg):
                a = 2 * math.pi * i / seg
                p = (rr * math.cos(a), rr * math.sin(a), end * h / 2)
                if axis == "x":
                    p = (p[2], p[0], p[1])
                elif axis == "y":
                    p = (p[0], p[2], p[1])
                v.append((x + p[0], y + p[1], z + p[2]))
        f = []
        for i in range(seg):
            j = (i + 1) % seg
            f.append((i, j, seg + j, seg + i))
        f.append(tuple(range(seg - 1, -1, -1)))
        f.append(tuple(seg + i for i in range(seg)))
        # axis permutation flips winding for x (cyclic) no; for 'y' swap flips
        if axis == "y":
            f = [tuple(reversed(q)) for q in f]
        self._add(v, f, m)

    def quad(self, a, b, c, d, m=0):
        self._add([a, b, c, d], [(0, 1, 2, 3)], m)

    def tube(self, pts, r, seg=8, m=0):
        """Straight-segment tube through the polyline pts (no end caps except at the ends)."""
        rings = []
        n = len(pts)
        for k, p in enumerate(pts):
            p = Vector(p)
            if k == 0:
                t = Vector(pts[1]) - p
            elif k == n - 1:
                t = p - Vector(pts[k - 1])
            else:
                t = Vector(pts[k + 1]) - Vector(pts[k - 1])
            t.normalize()
            ref = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
            u = t.cross(ref).normalized()
            w = t.cross(u).normalized()
            rings.append([tuple(p + (u * math.cos(2 * math.pi * i / seg) + w * math.sin(2 * math.pi * i / seg)) * r)
                          for i in range(seg)])
        v = [q for ring in rings for q in ring]
        f = []
        for k in range(n - 1):
            for i in range(seg):
                j = (i + 1) % seg
                f.append((k * seg + i, k * seg + j, (k + 1) * seg + j, (k + 1) * seg + i))
        f.append(tuple(range(seg - 1, -1, -1)))
        f.append(tuple((n - 1) * seg + i for i in range(seg)))
        self._add(v, f, m)

    def extend(self, other, offset=(0, 0, 0), mmap=None):
        o = len(self.v)
        ox, oy, oz = offset
        self.v.extend((a + ox, b + oy, c + oz) for a, b, c in other.v)
        self.f.extend(tuple(o + i for i in q) for q in other.f)
        self.mi.extend((mmap[m] if mmap else m) for m in other.mi)

    def mesh(self, name):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.update()
        me.validate()
        for p, m in zip(me.polygons, self.mi):
            p.material_index = m
        return me

    def obj(self, name, mats, bevel=None, smooth=False):
        me = self.mesh(name)
        for m in mats:
            me.materials.append(m)
        o = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(o)
        if bevel:
            C.bevel(o, bevel[0], bevel[1] if len(bevel) > 1 else 2)
        if smooth:
            for p in me.polygons:
                p.use_smooth = True
        return o


def put(o, coll, parent, loc=None, rot=None):
    C.add(o, coll, parent)
    if loc is not None:
        o.location = loc
    if rot is not None:
        o.rotation_euler = rot
    return o


def linked_copy(o, name, coll, parent, loc=(0, 0, 0), rot=(0, 0, 0)):
    """Object sharing mesh data (instance-like) with o."""
    c = bpy.data.objects.new(name, o.data)
    for m in o.modifiers:
        n = c.modifiers.new(m.name, m.type)
        if m.type == "BEVEL":
            n.width, n.segments, n.limit_method, n.angle_limit = m.width, m.segments, m.limit_method, m.angle_limit
    coll.objects.link(c)
    c.parent = parent
    c.location = loc
    c.rotation_euler = rot
    return c


def text(name, body, size, loc, rot, mat, coll, parent, extrude=0.0003, align="CENTER"):
    t = C.text_mesh(name, body, size, loc, rot, extrude=extrude, mat=mat, align=align)
    C.add(t, coll, parent)
    return t


def curve(name, pts, radius, mat, coll, parent, cyclic=False, res=6, smooth=True):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = 2
    cu.resolution_u = res
    cu.use_fill_caps = True
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for p, q in zip(sp.bezier_points, pts):
        p.co = q
        p.handle_left_type = p.handle_right_type = "AUTO" if smooth else "VECTOR"
    sp.use_cyclic_u = cyclic
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    coll.objects.link(o)
    o.parent = parent
    return o


def drive(target, path, index, root, prop, expr):
    """Simple-expression driver on target.<path>[index] reading custom property root[prop] (variable name p)."""
    fc = target.driver_add(path, index) if index is not None else target.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    v = d.variables.new()
    v.name = "p"
    v.targets[0].id = root
    v.targets[0].data_path = '["%s"]' % prop
    return fc


def bbox_visible(coll):
    """Like C.bbox_mm but skips collections hidden in the viewport (variant collections off by default)."""
    pts = []

    def walk(c):
        if c.hide_viewport:
            return
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


def use_visible_bbox():
    C.bbox_mm = bbox_visible


def drive_multi(target, path, index, variables, expr):
    """Driver with several variables: variables = {name: (id_object, data_path)}."""
    fc = target.driver_add(path, index) if index is not None else target.driver_add(path)
    d = fc.driver
    d.type = "SCRIPTED"
    d.expression = expr
    for n, (idb, dp) in variables.items():
        v = d.variables.new()
        v.name = n
        v.targets[0].id = idb
        v.targets[0].data_path = dp
    return fc


def set_collection_visible(coll, on):
    coll.hide_render = not on
    coll.hide_viewport = not on


# ------------------------------------------------------------------ previews and metadata
def _light_setup(kind, center, radius, sun=1.4, world=0.45, extra=None):
    scn = bpy.context.scene
    w = bpy.data.worlds.new("PREVIEW_world")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.75, 0.8, 0.9, 1)
    bg.inputs["Strength"].default_value = world
    scn.world = w
    if sun > 0:
        s = bpy.data.lights.new("PREVIEW_sun", "SUN")
        s.energy = sun
        so = bpy.data.objects.new("PREVIEW_sun", s)
        so.rotation_euler = (0.9, 0.3, 0.7)
        scn.collection.objects.link(so)
    for i, (loc, energy, size) in enumerate(extra or []):
        a = bpy.data.lights.new("PREVIEW_area%d" % i, "AREA")
        a.energy = energy
        a.size = size
        ao = bpy.data.objects.new("PREVIEW_area%d" % i, a)
        ao.location = loc
        scn.collection.objects.link(ao)
        c = ao.constraints.new("TRACK_TO")
        c.track_axis = "TRACK_NEGATIVE_Z"
        c.up_axis = "UP_Y"
        t = bpy.data.objects.new("PREVIEW_tgt_a%d" % i, None)
        t.location = center
        scn.collection.objects.link(t)
        c.target = t


def render_views(coll, out_dir, asset_id, views, floor=True, floor_size=None, floor_color=(0.12, 0.125, 0.12),
                 sun=1.4, world=0.45, lights=None, res=(900, 675), samples=16, exposure=0.0, clip=(0.01, 200)):
    """views: list of dict(name=, loc=, target=, lens=). Lights/floor defined here; call after save."""
    os.makedirs(out_dir, exist_ok=True)
    scn = bpy.context.scene
    for o in [o for o in bpy.data.objects if o.name.startswith("PREVIEW_")]:
        bpy.data.objects.remove(o, do_unlink=True)
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = samples
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.compression = 80
    scn.view_settings.view_transform = "Standard"
    scn.view_settings.exposure = exposure
    keep = {o for c in [coll] + list(coll.children_recursive) for o in c.objects}
    for o in scn.objects:
        if o not in keep and not o.name.startswith("PREVIEW_"):
            o.hide_render = True
    bb = C.bbox_mm(coll)
    lo, hi = Vector(bb[0]) / 1000, Vector(bb[1]) / 1000
    center = (lo + hi) / 2
    _light_setup("studio", center, (hi - lo).length / 2, sun=sun, world=world, extra=lights)
    if floor:
        fs = floor_size or max((hi - lo).length * 4, 4)
        bpy.ops.mesh.primitive_plane_add(size=fs, location=(center.x, center.y, lo.z - 0.002))
        fl = bpy.context.active_object
        fl.name = "PREVIEW_floor"
        fl.data.materials.append(C.principled("PREVIEW_floor_mat", floor_color, rough=0.8))
    cd = bpy.data.cameras.new("PREVIEW_cam")
    cd.clip_start, cd.clip_end = clip
    cam = bpy.data.objects.new("PREVIEW_cam", cd)
    scn.collection.objects.link(cam)
    scn.camera = cam
    cons = cam.constraints.new("TRACK_TO")
    cons.track_axis = "TRACK_NEGATIVE_Z"
    cons.up_axis = "UP_Y"
    tgt = bpy.data.objects.new("PREVIEW_tgt", None)
    scn.collection.objects.link(tgt)
    cons.target = tgt
    outs = []
    for v in views:
        cam.location = Vector(v["loc"])
        tgt.location = Vector(v["target"])
        cd.lens = v.get("lens", 35)
        cd.sensor_width = 36
        p = os.path.join(out_dir, "%s_%s.png" % (asset_id, v["name"]))
        scn.render.filepath = p
        bpy.ops.render.render(write_still=True)
        outs.append(p)
    return outs


def std_views(coll, dist=1.0, lens=40):
    """front / three_quarter / top camera set from the bbox (Z up, -Y front)."""
    bb = C.bbox_mm(coll)
    lo, hi = Vector(bb[0]) / 1000, Vector(bb[1]) / 1000
    c = (lo + hi) / 2
    s = hi - lo
    r = s.length / 2
    d = r * 2.1 * dist
    return [dict(name="front", loc=(c.x, c.y - d, c.z + r * 0.12), target=c, lens=lens),
            dict(name="three_quarter", loc=(c.x + d * 0.62, c.y - d * 0.72, c.z + r * 0.5), target=c, lens=lens),
            dict(name="top", loc=(c.x, c.y - r * 0.05, c.z + d * 1.15), target=c, lens=lens)]


def extend_meta(blend_path, extra):
    p = os.path.splitext(blend_path)[0] + ".json"
    with open(p) as f:
        meta = json.load(f)
    meta.update(extra)
    with open(p, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def dim(item, value, unit, source, level):
    return dict(item=item, value=value, unit=unit, source=source, accuracy=level)


def mat_slots():
    return sorted(m.name for m in bpy.data.materials if m.name.startswith("MAT_datacenter_"))


def hook_list(coll):
    return sorted(o.name for c in [coll] + list(coll.children_recursive) for o in c.objects if o.name.startswith("HOOK_"))


def custom_props(root):
    return {k: root[k] for k in root.keys() if k.startswith("p_")}


import common as C  # noqa: E402  (after sys.path insertion by the caller)
