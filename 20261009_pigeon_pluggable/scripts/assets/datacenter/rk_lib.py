"""Rack asset helpers (workstream R, DevLog-007). Blender 4.2, headless.

Adapted (copied, then extended) from the interconnect film project:
  ../20261002_interconnect_film/scripts/assets/_common/common.py      (asset root, hooks, principled, text, bbox, meta)
  ../20261002_interconnect_film/scripts/assets/datacenter/dc_common.py (MeshBuilder MB, cached materials, curves)
  ../20261002_interconnect_film/scripts/assets/datacenter/dc_parts.py  (Part / instantiate with shared mesh data)
Extensions: box-projected UVs per part, small generated image tiles (perforation, cable bundle, fan grille) so the
look survives glTF export, ellipsoids, text/curves converted to meshes, glb export, simple EEVEE preview renderer.

Units: metres. Build frame of every rack: origin = floor centre of the footprint, FRONT = +X, width along Y, Z up.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector

mm = lambda x: x * 0.001
RU = mm(44.45)


def argv_after_dashes():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse_args():
    a = argv_after_dashes()
    out = a[0] if a else "assets/components/datacenter"
    kv = {}
    for s in a[1:]:
        if "=" in s:
            k, v = s.split("=", 1)
            for cast in (int, float):
                try:
                    v = cast(v)
                    break
                except ValueError:
                    pass
            kv[k] = v
    return out, kv


# ------------------------------------------------------------------ scene / asset
def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    _MC.clear()
    _CACHE.clear()
    return bpy.context.scene


def new_asset(asset_id, accuracy="B", version="1.0.0"):
    coll = bpy.data.collections.new("ASSET_" + asset_id)
    bpy.context.scene.collection.children.link(coll)
    root = bpy.data.objects.new("ROOT_" + asset_id, None)
    root.empty_display_type = "PLAIN_AXES"
    coll.objects.link(root)
    root["asset_id"] = asset_id
    root["accuracy_level"] = accuracy
    root["asset_version"] = version
    return coll, root


def sub(coll, name):
    c = bpy.data.collections.new(name)
    coll.children.link(c)
    return c


def empty(name, coll, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), size=0.05, kind="PLAIN_AXES"):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = kind
    e.empty_display_size = size
    coll.objects.link(e)
    e.parent = parent
    e.location = loc
    e.rotation_euler = rot
    return e


def hook(name, coll, parent, loc=(0, 0, 0), rot=(0, 0, 0), **props):
    h = empty("HOOK_" + name, coll, parent, loc, rot, 0.06, "ARROWS")
    for k, v in props.items():
        h[k] = v
    return h


def set_visible(coll, on):
    coll.hide_render = not on
    coll.hide_viewport = not on


# ------------------------------------------------------------------ materials
_MC = {}


def principled(name, base=(0.5, 0.5, 0.5), metallic=0.0, rough=0.5, emit=None, emit_strength=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*base[:3], 1)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    if emit is not None:
        b.inputs["Emission Color"].default_value = (*emit[:3], 1)
        b.inputs["Emission Strength"].default_value = emit_strength
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        m.surface_render_method = "BLENDED"
    m.diffuse_color = (*base[:3], 1)
    return m


def M(name, base=(0.5, 0.5, 0.5), metallic=0.0, rough=0.5, emit=None, emit_strength=0.0, alpha=1.0):
    if name not in _MC:
        _MC[name] = principled("MAT_dc_" + name, base, metallic, rough, emit, emit_strength, alpha)
    return _MC[name]


def tile_image(name, n, fn):
    """n x n RGBA image from fn(u, v) -> (r, g, b) in 0..1 (u, v in [0, 1)); packed into the .blend."""
    key = "IMG_" + name
    if key in bpy.data.images:
        return bpy.data.images[key]
    img = bpy.data.images.new(key, n, n, alpha=False)
    px = []
    for j in range(n):
        for i in range(n):
            r, g, b = fn((i + 0.5) / n, (j + 0.5) / n)
            px += [r, g, b, 1.0]
    img.pixels = px
    img.pack()
    return img


def tex_M(name, img, metallic=0.5, rough=0.5, emit_img=False, emit_strength=0.0):
    """Principled material whose base colour (and optionally emission) comes from an image (UVs from MB box projection)."""
    if name in _MC:
        return _MC[name]
    m = bpy.data.materials.new("MAT_dc_" + name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = rough
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = img
    t.interpolation = "Linear"
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    if emit_img:
        nt.links.new(t.outputs["Color"], b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = emit_strength
    m.diffuse_color = (0.2, 0.2, 0.2, 1)
    _MC[name] = m
    return m


def palette():
    """Shared material palette (lazily created)."""
    P = dict(
        frame=M("rack_frame_black", (0.022, 0.023, 0.025), 0.3, 0.55),
        panel=M("rack_panel_black", (0.03, 0.031, 0.034), 0.5, 0.5),
        silver=M("tray_silver", (0.40, 0.41, 0.43), 0.8, 0.38),
        silver_dk=M("tray_silver_dark", (0.22, 0.225, 0.24), 0.8, 0.42),
        steel=M("steel_zinc", (0.62, 0.64, 0.66), 0.9, 0.36),
        dark_steel=M("steel_dark", (0.10, 0.105, 0.115), 0.5, 0.55),
        nickel=M("nickel", (0.55, 0.55, 0.57), 1.0, 0.3),
        nickel_dk=M("nickel_dark", (0.30, 0.30, 0.32), 0.6, 0.5),
        cage_wall=M("cage_wall_dark", (0.10, 0.105, 0.11), 0.0, 0.75),
        cage_wall_lt=M("cage_wall_zinc", (0.30, 0.31, 0.32), 0.0, 0.75),
        badge=M("badge_black_gloss", (0.01, 0.01, 0.012), 0.3, 0.2),
        black=M("plastic_black", (0.012, 0.012, 0.014), 0.0, 0.5),
        grey=M("plastic_grey", (0.16, 0.165, 0.17), 0.0, 0.5),
        light_grey=M("panel_light_grey", (0.33, 0.335, 0.34), 0.5, 0.45),
        champagne=M("champagne", (0.80, 0.70, 0.50), 0.75, 0.35),
        champagne_matte=M("champagne_matte", (0.62, 0.54, 0.38), 0.4, 0.6),
        champ_txt=M("label_text_dark", (0.03, 0.025, 0.02), 0.0, 0.6),
        copper=M("copper", (0.95, 0.45, 0.2), 1.0, 0.28),
        brass=M("brass", (0.85, 0.65, 0.25), 1.0, 0.3),
        gold=M("gold", (0.95, 0.72, 0.28), 1.0, 0.25),
        pcb=M("pcb_green", (0.03, 0.16, 0.07), 0.0, 0.35),
        hose=M("hose_epdm", (0.015, 0.015, 0.017), 0.0, 0.55),
        qd_blue=M("qd_blue", (0.04, 0.2, 0.75), 0.0, 0.35),
        qd_red=M("qd_red", (0.75, 0.05, 0.04), 0.0, 0.35),
        label=M("label_white", (0.85, 0.85, 0.83), 0.0, 0.6),
        port_hole=M("port_hole", (0.004, 0.004, 0.005), 0.0, 0.8),
        rj45=M("rj45_shell", (0.38, 0.39, 0.40), 0.9, 0.35),
        cable_blue=M("cable_blue", (0.05, 0.25, 0.75), 0.0, 0.45),
        cable_yellow=M("fiber_yellow", (0.95, 0.75, 0.05), 0.0, 0.45),
        cable_aqua=M("fiber_aqua", (0.1, 0.75, 0.8), 0.0, 0.45),
        cable_black=M("cable_black", (0.02, 0.02, 0.022), 0.0, 0.5),
        led_g=M("led_green", (0.1, 1, 0.2), 0.0, 0.3, (0.1, 1, 0.25), 8.0),
        led_a=M("led_amber", (1, 0.55, 0.05), 0.0, 0.3, (1, 0.55, 0.05), 8.0),
        led_b=M("led_blue", (0.1, 0.4, 1), 0.0, 0.3, (0.15, 0.45, 1), 8.0),
        led_c=M("led_cyan", (0.1, 0.9, 1), 0.0, 0.3, (0.1, 0.9, 1), 10.0),
        led_r=M("led_red", (1, 0.06, 0.04), 0.0, 0.3, (1, 0.06, 0.04), 6.0),
        rubber=M("rubber", (0.02, 0.02, 0.02), 0.0, 0.8),
    )
    # image tiles (one tile = one UV unit; MB parts set uv_tile = tile size in metres)
    sq = tile_image("perf_square", 16, lambda u, v: (0.0,) * 3 if (0.2 < u < 0.8 and 0.2 < v < 0.8) else (0.10, 0.10, 0.11))
    P["perf_black"] = tex_M("perf_square_black", sq, 0.5, 0.5)
    rd = tile_image("perf_round", 16, lambda u, v: (0.01,) * 3 if (u - 0.5) ** 2 + (v - 0.5) ** 2 < 0.11 else (0.5, 0.51, 0.53))
    P["perf_silver"] = tex_M("perf_round_silver", rd, 0.8, 0.38)
    hx = tile_image("perf_hex", 32, _hex_fn)
    P["perf_hex"] = tex_M("perf_hex_silver", hx, 0.8, 0.38)
    cb = tile_image("cable_bundle", 32, _cable_fn)
    P["cables"] = tex_M("cable_bundle", cb, 0.6, 0.4)
    br = tile_image("brick_grille", 32, _brick_fn)
    P["brick"] = tex_M("brick_grille_black", br, 0.4, 0.5)
    rl = tile_image("rail_holes", 16, lambda u, v: (0.0,) * 3 if (0.3 < u < 0.7 and 0.3 < v < 0.7) else (0.03, 0.031, 0.034))
    P["rail"] = tex_M("rail_square_holes", rl, 0.5, 0.45)
    return P


def _hex_fn(u, v):
    # hexagonal holes (approx): two offset round grids
    d1 = (u - 0.5) ** 2 + (v - 0.25) ** 2
    d2 = min((u - 0.0) ** 2, (u - 1.0) ** 2) + (v - 0.75) ** 2
    hole = min(d1, d2) < 0.045
    return (0.01, 0.01, 0.012) if hole else (0.55, 0.56, 0.58)


def _cable_fn(u, v):
    # vertical twinax bundle: silver jackets with dark gaps, every 4th cable darker
    k = int(u * 8)
    f = u * 8 - k
    if f < 0.12 or f > 0.88:
        return (0.02, 0.02, 0.022)
    shade = 0.5 + 0.35 * math.sin(math.pi * f)
    if k % 4 == 3:
        shade *= 0.35
    return (shade * 0.85, shade * 0.86, shade * 0.9)


def _brick_fn(u, v):
    # fan grille: 4 x 4 staggered rectangular openings per tile (Q3400 rear figure), dark rotor visible through the holes
    row = int(v * 4)
    uu = (u * 4 + (0.5 if row % 2 else 0.0)) % 1.0
    vv = (v * 4) % 1.0
    hole = 0.14 < uu < 0.86 and 0.22 < vv < 0.78
    return (0.012, 0.012, 0.014) if hole else (0.16, 0.16, 0.17)


# ------------------------------------------------------------------ mesh builder
class MB:
    """Boxes, cylinders, tubes, ellipsoids into one mesh with per-face material index (metres)."""

    def __init__(self):
        self.v, self.f, self.mi = [], [], []

    def _add(self, verts, faces, m=0):
        o = len(self.v)
        self.v.extend(tuple(p) for p in verts)
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

    def box2(self, lo, hi, m=0):
        self.box(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2),
                 (abs(hi[0] - lo[0]), abs(hi[1] - lo[1]), abs(hi[2] - lo[2])), m)

    def cyl(self, c, r, h, axis="z", seg=16, m=0, r2=None, cap=True):
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
                    p = (p[1], p[2], p[0])
                v.append((x + p[0], y + p[1], z + p[2]))
        f = [(i, (i + 1) % seg, seg + (i + 1) % seg, seg + i) for i in range(seg)]
        if cap:
            f.append(tuple(range(seg - 1, -1, -1)))
            f.append(tuple(seg + i for i in range(seg)))
        self._add(v, f, m)

    def ring(self, c, r_out, r_in, h, axis="z", seg=24, m=0):
        """Flat annulus (washer) of thickness h."""
        x, y, z = c
        v = []
        for end in (-1, 1):
            for rr in (r_out, r_in):
                for i in range(seg):
                    a = 2 * math.pi * i / seg
                    p = (rr * math.cos(a), rr * math.sin(a), end * h / 2)
                    if axis == "x":
                        p = (p[2], p[0], p[1])
                    elif axis == "y":
                        p = (p[1], p[2], p[0])
                    v.append((x + p[0], y + p[1], z + p[2]))
        bo, bi, to, ti = 0, seg, 2 * seg, 3 * seg
        f = []
        for i in range(seg):
            j = (i + 1) % seg
            f += [(bo + i, bo + j, to + j, to + i), (bi + j, bi + i, ti + i, ti + j),
                  (to + i, to + j, ti + j, ti + i), (bo + j, bo + i, bi + i, bi + j)]
        self._add(v, f, m)

    def ellipsoid(self, c, r, seg=16, rings=10, m=0):
        v = [(c[0], c[1], c[2] - r[2])]
        for j in range(1, rings):
            th = math.pi * j / rings - math.pi / 2
            for i in range(seg):
                a = 2 * math.pi * i / seg
                v.append((c[0] + r[0] * math.cos(th) * math.cos(a), c[1] + r[1] * math.cos(th) * math.sin(a),
                          c[2] + r[2] * math.sin(th)))
        v.append((c[0], c[1], c[2] + r[2]))
        f = []
        for i in range(seg):
            f.append((0, 1 + (i + 1) % seg, 1 + i))
        for j in range(rings - 2):
            a0, a1 = 1 + j * seg, 1 + (j + 1) * seg
            for i in range(seg):
                k = (i + 1) % seg
                f.append((a0 + i, a0 + k, a1 + k, a1 + i))
        top = len(v) - 1
        a0 = 1 + (rings - 2) * seg
        for i in range(seg):
            f.append((a0 + i, a0 + (i + 1) % seg, top))
        self._add(v, f, m)

    def tube(self, pts, r, seg=8, m=0, closed=False):
        rings = []
        n = len(pts)
        P = [Vector(p) for p in pts]
        for k, p in enumerate(P):
            if closed:
                t = P[(k + 1) % n] - P[k - 1]
            elif k == 0:
                t = P[1] - p
            elif k == n - 1:
                t = p - P[k - 1]
            else:
                t = P[k + 1] - P[k - 1]
            t.normalize()
            ref = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
            u = t.cross(ref).normalized()
            w = t.cross(u).normalized()
            rings.append([tuple(p + (u * math.cos(2 * math.pi * i / seg) + w * math.sin(2 * math.pi * i / seg)) * r)
                          for i in range(seg)])
        v = [q for ring in rings for q in ring]
        f = []
        for k in range(n - 1 + (1 if closed else 0)):
            k2 = (k + 1) % n
            for i in range(seg):
                j = (i + 1) % seg
                f.append((k * seg + i, k * seg + j, k2 * seg + j, k2 * seg + i))
        if not closed:
            f.append(tuple(range(seg - 1, -1, -1)))
            f.append(tuple((n - 1) * seg + i for i in range(seg)))
        self._add(v, f, m)

    def tri(self, a, b, c, m=0):
        self._add([a, b, c], [(0, 1, 2)], m)

    def quad(self, a, b, c, d, m=0):
        self._add([a, b, c, d], [(0, 1, 2, 3)], m)

    def extend(self, other, offset=(0, 0, 0), mmap=None):
        o = len(self.v)
        ox, oy, oz = offset
        self.v.extend((a + ox, b + oy, c + oz) for a, b, c in other.v)
        self.f.extend(tuple(o + i for i in q) for q in other.f)
        self.mi.extend((mmap[m] if mmap else m) for m in other.mi)

    def mesh(self, name, uv_tile=None):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.update()
        for p, m in zip(me.polygons, self.mi):
            p.material_index = m
        if uv_tile:
            box_uv(me, uv_tile)
        return me


def port(mb, y, z, w, h, wall, depth, out, mi_shell, mi_hole, x_axis_front=True):
    """Recessed port at (y, z) on a face plane x = 0 (mm, tray/chassis-local): shell walls of thickness wall protrude to
    x = out, the opening floor (mi_hole) sits depth mm behind the shell front. Coordinates in mm, written in metres."""
    s_ = 0.001
    x0, x1 = (out - depth - 1.0) * s_, out * s_
    for (ya, yb, za, zb) in ((y - w / 2, y + w / 2, z + h / 2 - wall, z + h / 2), (y - w / 2, y + w / 2, z - h / 2, z - h / 2 + wall),
                             (y - w / 2, y - w / 2 + wall, z - h / 2, z + h / 2), (y + w / 2 - wall, y + w / 2, z - h / 2, z + h / 2)):
        mb.box2((x0, ya * s_, za * s_), (x1, yb * s_, zb * s_), mi_shell)
    mb.box2(((out - depth - 1.0) * s_, (y - w / 2) * s_, (z - h / 2) * s_), ((out - depth) * s_, (y + w / 2) * s_, (z + h / 2) * s_), mi_hole)


def box_uv(me, tile):
    """Box-project UVs: each face uses the two coordinates orthogonal to its dominant normal axis, divided by tile (m)."""
    uv = me.uv_layers.new(name="UVMap")
    for p in me.polygons:
        n = p.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        a, b = [(1, 2), (0, 2), (0, 1)][ax]
        for li in p.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            uv.data[li].uv = (co[a] / tile, co[b] / tile)


# ------------------------------------------------------------------ parts with shared mesh data
_CACHE = {}


class Part:
    def __init__(self, name, mb, mats, uv_tile=None, smooth=False, bevel=None):
        self.name, self.mb, self.mats, self.uv_tile, self.smooth, self.bevel = name, mb, mats, uv_tile, smooth, bevel


def instantiate(parts, coll, parent, prefix, key=None, loc=(0, 0, 0), rot=(0, 0, 0)):
    """Objects from parts; mesh data shared across calls with the same key (cheap instancing). Returns {name: obj}."""
    out = {}
    for p in parts:
        if not p.mb.f:
            continue
        ck = (key, p.name) if key else None
        if ck and ck in _CACHE:
            me = _CACHE[ck]
        else:
            me = p.mb.mesh(("mesh_%s_%s" % (key, p.name)) if key else (prefix + "_" + p.name), p.uv_tile)
            for m in p.mats:
                me.materials.append(m)
            if p.smooth:
                for pl in me.polygons:
                    pl.use_smooth = True
            if ck:
                _CACHE[ck] = me
        o = bpy.data.objects.new("%s_%s" % (prefix, p.name), me)
        if p.bevel:
            bm = o.modifiers.new("Bevel", "BEVEL")
            bm.width = p.bevel
            bm.segments = 1
            bm.limit_method = "ANGLE"
        coll.objects.link(o)
        o.parent = parent
        o.location = loc
        o.rotation_euler = rot
        out[p.name] = o
    return out


def obj_from_mb(name, mb, mats, coll, parent=None, uv_tile=None, smooth=False):
    return instantiate([Part("x", mb, mats, uv_tile, smooth)], coll, parent, name)["x"]


# ------------------------------------------------------------------ text and curves (converted to meshes)
def _to_mesh(o, coll, parent):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
    name = o.name
    mats = list(o.data.materials)
    mw = o.matrix_world.copy()
    data = o.data
    bpy.data.objects.remove(o, do_unlink=True)
    if isinstance(data, bpy.types.Curve):
        bpy.data.curves.remove(data)
    for m in mats:
        if m and m.name not in [x.name for x in me.materials if x]:
            me.materials.append(m)
    n = bpy.data.objects.new(name, me)
    coll.objects.link(n)
    n.parent = parent
    n.matrix_world = mw
    return n


def text(name, body, size, loc, rot, mat, coll, parent, extrude=0.0, align="CENTER"):
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
    o.data.materials.append(mat)
    bpy.context.view_layer.update()
    if parent is not None:
        bpy.context.view_layer.update()
        o.matrix_world = parent.matrix_world @ o.matrix_world
        bpy.context.view_layer.update()
    n = _to_mesh(o, coll, parent)
    return n


def curve(name, pts, radius, mat, coll, parent, res=6, bevel_res=2):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = radius
    cu.bevel_resolution = bevel_res
    cu.resolution_u = res
    cu.use_fill_caps = True
    sp = cu.splines.new("BEZIER")
    sp.bezier_points.add(len(pts) - 1)
    for p, q in zip(sp.bezier_points, pts):
        p.co = q
        p.handle_left_type = p.handle_right_type = "AUTO"
    cu.materials.append(mat)
    o = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(o)
    bpy.context.view_layer.update()
    if parent is not None:
        o.matrix_world = parent.matrix_world @ o.matrix_world
        bpy.context.view_layer.update()
    return _to_mesh(o, coll, parent)


def smooth_path(pts, sub=6):
    """Centripetal-free uniform Catmull-Rom through pts (end points duplicated); returns a dense polyline."""
    P = [Vector(p) for p in pts]
    P = [P[0]] + P + [P[-1]]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for k in range(sub):
            t = k / sub
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-2])
    return [tuple(v) for v in out]


def cable(name, pts, radius, mat, coll, parent, seg=8, sub=6):
    """Cable mesh along a Catmull-Rom path (no Bezier handle overshoot)."""
    mb = MB()
    mb.tube(smooth_path(pts, sub), radius, seg, 0)
    return obj_from_mb(name, mb, [mat], coll, parent, smooth=True)


# ------------------------------------------------------------------ metadata / save / export
def bbox(coll, visible_only=True):
    pts = []

    def walk(c):
        if visible_only and c.hide_render:
            return
        for o in c.objects:
            if o.type == "MESH" and not o.hide_render:
                for v in o.bound_box:
                    pts.append(o.matrix_world @ Vector(v))
        for ch in c.children:
            walk(ch)
    walk(coll)
    if not pts:
        return None
    return ([min(p[i] for p in pts) for i in range(3)], [max(p[i] for p in pts) for i in range(3)])


def count_tris(coll):
    n = 0
    for o in {o for c in [coll] + list(coll.children_recursive) for o in c.objects}:
        if o.type == "MESH":
            n += sum(len(p.vertices) - 2 for p in o.data.polygons)
    return n


def hooks_of(coll):
    out = []
    for o in sorted({o for c in [coll] + list(coll.children_recursive) for o in c.objects}, key=lambda o: o.name):
        if o.name.startswith("HOOK_"):
            mw = o.matrix_world
            d = dict(name=o.name, location=[round(x, 5) for x in mw.translation],
                     facing_x=[round(x, 5) for x in (mw.to_3x3() @ Vector((1, 0, 0))).normalized()])
            for k in o.keys():
                if k not in ("_RNA_UI",):
                    v = o[k]
                    d[k] = v if isinstance(v, (int, float, str)) else list(v)
            out.append(d)
    return out


def save_blend(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0          # no .blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)


def export_glb(coll, path):
    """Export the collection tree (visible objects) to .glb, Y-up, modifiers applied, custom props as extras."""
    bpy.ops.object.select_all(action="DESELECT")
    objs = [o for c in [coll] + list(coll.children_recursive) if not c.hide_render for o in c.objects]
    for o in objs:
        try:
            o.select_set(True)
        except RuntimeError:
            pass
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_apply=True,
                              export_yup=True, export_extras=True, export_cameras=False, export_lights=False)
    bpy.ops.object.select_all(action="DESELECT")


def write_json(path, d):
    with open(path, "w") as f:
        json.dump(d, f, indent=2)


# ------------------------------------------------------------------ preview renderer (EEVEE draft)
def preview_setup(res=(1000, 1250), samples=8, world=0.25, exposure=0.0):
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = samples
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.view_settings.view_transform = "Standard"
    scn.view_settings.exposure = exposure
    w = bpy.data.worlds.get("PREVIEW_world") or bpy.data.worlds.new("PREVIEW_world")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.6, 0.68, 1)
    bg.inputs["Strength"].default_value = world
    scn.world = w
    return scn


def light(name, kind, loc, energy, size=1.0, target=(0, 0, 1.1), color=(1, 1, 1)):
    d = bpy.data.lights.new("PREVIEW_" + name, kind)
    d.energy = energy
    d.color = color
    if kind == "AREA":
        d.size = size
    o = bpy.data.objects.new("PREVIEW_" + name, d)
    o.location = loc
    bpy.context.scene.collection.objects.link(o)
    dirv = Vector(target) - Vector(loc)
    o.rotation_euler = dirv.to_track_quat("-Z", "Y").to_euler()
    return o


def floor(size=30, z=0.0, color=(0.07, 0.07, 0.075)):
    me = bpy.data.meshes.new("PREVIEW_floor")
    s = size / 2
    me.from_pydata([(-s, -s, z), (s, -s, z), (s, s, z), (-s, s, z)], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new("PREVIEW_floor", me)
    me.materials.append(principled("PREVIEW_floor_mat", color, 0.0, 0.7))
    bpy.context.scene.collection.objects.link(o)
    return o


def render_view(path, loc, target, lens=35, ortho=None):
    scn = bpy.context.scene
    cam = bpy.data.objects.get("PREVIEW_cam")
    if cam is None:
        cd = bpy.data.cameras.new("PREVIEW_cam")
        cam = bpy.data.objects.new("PREVIEW_cam", cd)
        scn.collection.objects.link(cam)
    cd = cam.data
    cd.clip_start, cd.clip_end = 0.01, 200
    cd.sensor_width = 36
    if ortho:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho
    else:
        cd.type = "PERSP"
        cd.lens = lens
    cam.location = loc
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    scn.camera = cam
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


# ------------------------------------------------------------------ preview-only bird envelope proxy
def bird_proxy(mb, origin, yaw=0.0, scale=2.6, neck_ext=0.0):
    """Envelope proxy of the crude-film pigeon (scripts/build_crude_film.py primitive, bird-local mm, +x beak, origin
    between the feet) at hall scale; mats: 0 gunmetal, 1 neck, 2 beak/legs, 3 TX eye (right, -y), 4 RX eye (left, +y)."""
    R = Matrix.Rotation(yaw, 3, "Z")
    o = Vector(origin)

    def P(x, y, z):
        return tuple(o + R @ Vector((mm(x * scale), mm(y * scale), mm(z * scale))))

    def E(c, r, mi):
        mb2 = MB()
        mb2.ellipsoid((0, 0, 0), (mm(r[0] * scale), mm(r[1] * scale), mm(r[2] * scale)), 12, 7, mi)
        base = len(mb.v)
        cc_ = Vector(P(*c))
        mb.v.extend(tuple(cc_ + R @ Vector(v)) for v in mb2.v)
        mb.f.extend(tuple(base + i for i in f) for f in mb2.f)
        mb.mi.extend(mb2.mi)
    E((-2, 0, 40), (36, 21, 20), 0)
    E((19, 0, 43), (21, 20.5, 22.5), 1)
    E((32, 0, 63), (11, 10, 15), 1)
    hx = neck_ext / (scale * 0.001)                      # neck stretch (m) -> bird-local mm on head parts
    if neck_ext > 0:
        for k in range(1, 4):
            E((32 + hx * k / 4, 0, 63 + 14 * k / 4), (9, 8.5, 9), 1)
    E((44 + hx, 0, 81.5), (12, 12, 12), 0)
    E((58 + hx, 0, 79.5), (6, 2.6, 2.6), 2)
    E((-46, 0, 22), (22, 12, 4), 0)
    E((47 + hx, -11.0, 84), (3.2, 1.6, 3.2), 3)
    E((47 + hx, 11.0, 84), (3.2, 1.6, 3.2), 4)
    for sy in (-1, 1):
        E((6, sy * 8, 13), (2, 2, 12), 2)


def bird_mats():
    return [principled("PREVIEW_bird_gun", (0.16, 0.17, 0.19), 0.7, 0.4), principled("PREVIEW_bird_neck", (0.1, 0.35, 0.3), 0.8, 0.3),
            principled("PREVIEW_bird_beak", (0.05, 0.05, 0.06), 0.8, 0.3),
            principled("PREVIEW_eye_tx", (1, 0.08, 0.05), 0, 0.3, (1, 0.08, 0.05), 6),
            principled("PREVIEW_eye_rx", (0.1, 0.85, 1), 0, 0.3, (0.1, 0.85, 1), 6)]


# ------------------------------------------------------------------ reuse meshes from the interconnect film library (read-only)
INTERCONNECT_DC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                                                "20261002_interconnect_film", "assets", "components", "datacenter"))


def append_mesh(blend_name, obj_name):
    """Append one object's mesh (with its materials) from ../20261002_interconnect_film/assets/components/datacenter/<blend>
    without touching that file; the object's own transform is baked into the mesh. Returns the mesh or None if missing."""
    path = os.path.join(INTERCONNECT_DC, blend_name)
    if not os.path.exists(path):
        print("WARN interconnect asset missing:", path)
        return None
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        dst.objects = [obj_name]
    o = dst.objects[0]
    me = o.data
    me.transform(o.matrix_basis)
    par = o.parent
    bpy.data.objects.remove(o, do_unlink=True)
    if par is not None and par.users == 0:
        bpy.data.objects.remove(par, do_unlink=True)
    for ob in [ob for ob in bpy.data.objects if ob.users == 0]:
        bpy.data.objects.remove(ob)
    return me
