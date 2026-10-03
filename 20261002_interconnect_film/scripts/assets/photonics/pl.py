"""Photonics category library (Blender 4.2, headless). Built on scripts/assets/_common/common.py.

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import pl as P
    C = P.C

Provides: Geo (mesh accumulator with box/cylinder/annulus/path-strip/tube/sphere-cap primitives), material
table (MAT_photonics_*), Geometry-Nodes point instancer, ribbon/fiber helpers, shared builders (FAU, MT ferrule,
fiber ribbon), custom preview renderer with arbitrary camera shots, evaluated triangle count, metadata helper.
All lengths in metres internally; use MM / UM for readability.
"""
import math
import os
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import common as C  # noqa: E402

MM = 1e-3
UM = 1e-6
PI = math.pi

# ------------------------------------------------------------------------------------------------ materials
# name: (base rgb, metallic, roughness, alpha, extra dict)
_MATS = {
    "silicon": ((0.34, 0.35, 0.38), 0.85, 0.32, 1.0, {}),
    "silicon_die_edge": ((0.45, 0.46, 0.48), 0.9, 0.35, 1.0, {}),
    "waveguide_si": ((0.55, 0.62, 0.72), 0.7, 0.2, 1.0, {}),
    "box_oxide": ((0.55, 0.72, 0.85), 0.0, 0.1, 0.22, {}),
    "cladding": ((0.65, 0.82, 0.95), 0.0, 0.05, 0.05, {}),
    "p_implant": ((0.85, 0.25, 0.22), 0.3, 0.35, 1.0, {}),
    "n_implant": ((0.18, 0.38, 0.9), 0.3, 0.35, 1.0, {}),
    "pplus_implant": ((1.0, 0.1, 0.08), 0.3, 0.35, 1.0, {}),
    "nplus_implant": ((0.05, 0.15, 1.0), 0.3, 0.35, 1.0, {}),
    "gold": ((1.0, 0.77, 0.30), 1.0, 0.22, 1.0, {}),
    "gold_pad": ((0.95, 0.72, 0.28), 1.0, 0.30, 1.0, {}),
    "copper": ((0.95, 0.52, 0.36), 1.0, 0.28, 1.0, {}),
    "aluminum": ((0.80, 0.81, 0.83), 1.0, 0.30, 1.0, {}),
    "metal1": ((0.95, 0.55, 0.30), 1.0, 0.30, 1.0, {}),
    "metal2": ((0.62, 0.80, 0.95), 1.0, 0.30, 1.0, {}),
    "tungsten_via": ((0.45, 0.46, 0.50), 1.0, 0.35, 1.0, {}),
    "tin_heater": ((0.22, 0.19, 0.18), 0.9, 0.35, 1.0, {}),
    "solder": ((0.78, 0.78, 0.80), 1.0, 0.22, 1.0, {}),
    "die_passivation": ((0.015, 0.025, 0.07), 0.2, 0.12, 1.0, {"coat": 0.5}),
    "eic_passivation": ((0.02, 0.03, 0.14), 0.2, 0.10, 1.0, {"coat": 0.6}),
    "eic_macro": ((0.03, 0.07, 0.30), 0.3, 0.2, 1.0, {}),
    "pic_waveguide": ((0.7, 0.9, 1.0), 0.8, 0.2, 1.0, {}),
    "ge_pd": ((0.55, 0.25, 0.65), 0.2, 0.3, 1.0, {}),
    "glass": ((0.80, 0.92, 0.95), 0.0, 0.04, 0.22, {}),
    "glass_edge": ((0.60, 0.85, 0.80), 0.0, 0.08, 0.45, {}),
    "epoxy": ((0.95, 0.65, 0.20), 0.0, 0.2, 0.55, {}),
    "fiber_glass": ((0.85, 0.95, 1.0), 0.0, 0.05, 0.45, {}),
    "fiber_core": ((0.4, 0.9, 1.0), 0.0, 0.2, 1.0, {"emit": (0.3, 0.8, 1.0), "emit_s": 2.0}),
    "organic_substrate": ((0.025, 0.16, 0.07), 0.0, 0.5, 1.0, {}),
    "substrate_edge": ((0.45, 0.38, 0.2), 0.0, 0.6, 1.0, {}),
    "lga_pad": ((0.95, 0.72, 0.30), 1.0, 0.25, 1.0, {}),
    "lid_nickel": ((0.72, 0.73, 0.75), 1.0, 0.22, 1.0, {}),
    "oe_body": ((0.62, 0.64, 0.68), 1.0, 0.30, 1.0, {}),
    "anodized": ((0.12, 0.12, 0.14), 0.9, 0.45, 1.0, {}),
    "mold_black": ((0.012, 0.012, 0.014), 0.0, 0.55, 1.0, {}),
    "ferrule": ((0.06, 0.06, 0.07), 0.0, 0.40, 1.0, {}),
    "ferrule_white": ((0.82, 0.80, 0.74), 0.0, 0.40, 1.0, {}),
    "steel": ((0.80, 0.80, 0.82), 1.0, 0.22, 1.0, {}),
    "boot_black": ((0.03, 0.03, 0.035), 0.0, 0.7, 1.0, {}),
    "plastic_blue": ((0.05, 0.18, 0.55), 0.0, 0.45, 1.0, {}),
    "plastic_aqua": ((0.1, 0.75, 0.80), 0.0, 0.45, 1.0, {}),
    "plastic_green": ((0.08, 0.5, 0.2), 0.0, 0.45, 1.0, {}),
    "label": ((0.9, 0.9, 0.88), 0.0, 0.7, 1.0, {}),
    "etch": ((0.85, 0.7, 0.35), 1.0, 0.4, 1.0, {}),
    "pcb_green": ((0.02, 0.28, 0.12), 0.0, 0.45, 1.0, {}),
    "cap_mlcc": ((0.75, 0.62, 0.40), 0.0, 0.35, 1.0, {}),
    "cap_end": ((0.85, 0.85, 0.88), 1.0, 0.3, 1.0, {}),
    "laser_chip": ((0.9, 0.55, 0.15), 0.8, 0.3, 1.0, {}),
    "ceramic": ((0.85, 0.82, 0.75), 0.0, 0.4, 1.0, {}),
    "kovar": ((0.55, 0.56, 0.58), 1.0, 0.35, 1.0, {}),
    "heatsink_grey": ((0.35, 0.36, 0.38), 0.9, 0.5, 1.0, {}),
}
_FIBER_COLORS = [  # TIA-598 order: blue orange green brown slate white red black yellow violet rose aqua
    (0.04, 0.16, 0.7), (1.0, 0.35, 0.02), (0.05, 0.6, 0.1), (0.35, 0.18, 0.06), (0.35, 0.38, 0.42), (0.9, 0.9, 0.9),
    (0.9, 0.05, 0.05), (0.03, 0.03, 0.03), (1.0, 0.85, 0.05), (0.45, 0.12, 0.7), (1.0, 0.45, 0.65), (0.1, 0.8, 0.8),
]


def mat(name, base=None, metallic=None, rough=None, alpha=None, **kw):
    """Get-or-create MAT_photonics_<name>. Unknown names need base/metallic/rough."""
    full = "MAT_photonics_" + name
    if full in bpy.data.materials:
        return bpy.data.materials[full]
    if name in _MATS and base is None:
        b, me, ro, al, ex = _MATS[name]
    else:
        b, me, ro, al, ex = base or (0.5, 0.5, 0.5), metallic or 0.0, rough if rough is not None else 0.5, \
            alpha if alpha is not None else 1.0, kw
    m = bpy.data.materials.new(full)
    m.use_nodes = True
    bs = m.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (*b[:3], 1)
    bs.inputs["Metallic"].default_value = me
    bs.inputs["Roughness"].default_value = ro
    if ex.get("coat"):
        bs.inputs["Coat Weight"].default_value = ex["coat"]
    if ex.get("emit"):
        bs.inputs["Emission Color"].default_value = (*ex["emit"][:3], 1)
        bs.inputs["Emission Strength"].default_value = ex.get("emit_s", 1.0)
    if al < 1.0:
        bs.inputs["Alpha"].default_value = al
        try:
            m.surface_render_method = "BLENDED"
        except Exception:
            pass
        try:
            m.use_backface_culling = False
        except Exception:
            pass
    return m


def fiber_color_mat(i):
    c = _FIBER_COLORS[i % 12]
    return mat("fiber_coat_%02d" % (i % 12), c, 0.0, 0.5)


# ------------------------------------------------------------------------------------------------ geometry accumulator
class Geo:
    """Accumulates vertices/faces (all quads/tris, outward normals); .build() makes a mesh object."""

    def __init__(self):
        self.v = []
        self.f = []

    def _add(self, verts, faces):
        n = len(self.v)
        self.v.extend(verts)
        self.f.extend([tuple(i + n for i in f) for f in faces])

    def box(self, x0, y0, z0, x1, y1, z1):
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if z1 < z0:
            z0, z1 = z1, z0
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        self._add(v, f)
        return self

    def box_c(self, cx, cy, cz, sx, sy, sz):
        return self.box(cx - sx / 2, cy - sy / 2, cz - sz / 2, cx + sx / 2, cy + sy / 2, cz + sz / 2)

    def cyl(self, cx, cy, z0, z1, r, n=16, r1=None):
        """Z-axis cylinder / frustum (r at z0, r1 at z1), triangle-fan caps."""
        r1 = r if r1 is None else r1
        v = []
        for k in range(n):
            a = 2 * PI * k / n
            v.append((cx + r * math.cos(a), cy + r * math.sin(a), z0))
        for k in range(n):
            a = 2 * PI * k / n
            v.append((cx + r1 * math.cos(a), cy + r1 * math.sin(a), z1))
        v.append((cx, cy, z0))
        v.append((cx, cy, z1))
        f = []
        for k in range(n):
            k2 = (k + 1) % n
            f.append((k, k2, n + k2, n + k))
            f.append((2 * n, k2, k))
            f.append((2 * n + 1, n + k, n + k2))
        self._add(v, f)
        return self

    def cyl_axis(self, p0, p1, r, n=12, caps=True):
        """Cylinder between two points (arbitrary axis)."""
        return self.tube([p0, p1], r, n=n, caps=caps)

    def annulus(self, cx, cy, r0, r1, z0, z1, n=64, a0=0.0, a1=2 * PI):
        """Extruded annulus sector between radii r0<r1 (r0=0 gives a disc / pie slice), angles a0..a1 (rad)."""
        full = abs((a1 - a0) - 2 * PI) < 1e-9
        ns = max(2, int(round(n * (a1 - a0) / (2 * PI))))
        angs = [a0 + (a1 - a0) * k / ns for k in range(ns + (0 if full else 1))]
        m = len(angs)
        if r0 <= 1e-12:
            return self._pie(cx, cy, r1, z0, z1, angs, full)
        v = []
        for a in angs:
            v.append((cx + r1 * math.cos(a), cy + r1 * math.sin(a), z0))  # 0 outer bottom
            v.append((cx + r1 * math.cos(a), cy + r1 * math.sin(a), z1))  # 1 outer top
            v.append((cx + r0 * math.cos(a), cy + r0 * math.sin(a), z0))  # 2 inner bottom
            v.append((cx + r0 * math.cos(a), cy + r0 * math.sin(a), z1))  # 3 inner top
        f = []
        last = m if full else m - 1
        for k in range(last):
            k2 = (k + 1) % m
            a, b = 4 * k, 4 * k2
            f.append((a, b, b + 1, a + 1))  # outer
            f.append((a + 2, a + 3, b + 3, b + 2))  # inner
            f.append((a + 1, b + 1, b + 3, a + 3))  # top
            f.append((a, a + 2, b + 2, b))  # bottom
        if not full:
            f.append((0, 1, 3, 2))  # start cap
            e = 4 * (m - 1)
            f.append((e, e + 2, e + 3, e + 1))
        self._add(v, f)
        return self

    def _pie(self, cx, cy, r, z0, z1, angs, full):
        m = len(angs)
        v = [(cx, cy, z0), (cx, cy, z1)]
        for a in angs:
            v.append((cx + r * math.cos(a), cy + r * math.sin(a), z0))
            v.append((cx + r * math.cos(a), cy + r * math.sin(a), z1))
        f = []
        last = m if full else m - 1
        for k in range(last):
            k2 = (k + 1) % m
            a, b = 2 + 2 * k, 2 + 2 * k2
            f.append((a, b, b + 1, a + 1))
            f.append((0, b, a))
            f.append((1, a + 1, b + 1))
        if not full:
            f.append((0, 2, 3, 1))
            e = 2 + 2 * (m - 1)
            f.append((0, 1, e + 1, e))
        self._add(v, f)
        return self

    def strip(self, pts, w, z0, z1, closed=False):
        """Extruded constant-width polyline strip (2D points, miter joints). Returns self."""
        n = len(pts)
        if n < 2:
            return self
        P2 = [Vector((p[0], p[1])) for p in pts]
        nrm = []
        for i in range(n):
            if closed:
                d0 = (P2[i] - P2[i - 1]).normalized()
                d1 = (P2[(i + 1) % n] - P2[i]).normalized()
            else:
                d0 = (P2[i] - P2[i - 1]).normalized() if i > 0 else (P2[1] - P2[0]).normalized()
                d1 = (P2[i + 1] - P2[i]).normalized() if i < n - 1 else d0
            n0 = Vector((-d0.y, d0.x))
            n1 = Vector((-d1.y, d1.x))
            mi = (n0 + n1)
            if mi.length < 1e-9:
                mi = n0
            mi.normalize()
            cosh = max(mi.dot(n0), 0.25)
            nrm.append(mi * (w / 2 / cosh))
        L = [(P2[i] + nrm[i]) for i in range(n)]
        R = [(P2[i] - nrm[i]) for i in range(n)]
        v = []
        for i in range(n):
            v += [(L[i].x, L[i].y, z0), (L[i].x, L[i].y, z1), (R[i].x, R[i].y, z0), (R[i].x, R[i].y, z1)]
        f = []
        last = n if closed else n - 1
        for i in range(last):
            j = (i + 1) % n
            a, b = 4 * i, 4 * j
            f.append((a + 1, b + 1, b + 3, a + 3))  # top (L top -> R top)
            f.append((a, a + 2, b + 2, b))  # bottom
            f.append((a, b, b + 1, a + 1))  # left side
            f.append((a + 2, a + 3, b + 3, b + 2))  # right side
        if not closed:
            f.append((0, 1, 3, 2))
            e = 4 * (n - 1)
            f.append((e, e + 2, e + 3, e + 1))
        self._add(v, f)
        return self

    def tube(self, pts, r, n=10, caps=True, r_end=None):
        """Round tube along a 3D polyline (parallel-transport frames)."""
        P3 = [Vector(p) for p in pts]
        N = len(P3)
        tans = []
        for i in range(N):
            if i == 0:
                t = P3[1] - P3[0]
            elif i == N - 1:
                t = P3[-1] - P3[-2]
            else:
                t = (P3[i + 1] - P3[i]).normalized() + (P3[i] - P3[i - 1]).normalized()
            tans.append(t.normalized())
        ref = Vector((0, 0, 1)) if abs(tans[0].z) < 0.9 else Vector((1, 0, 0))
        u = tans[0].cross(ref).normalized()
        v_ = []
        f = []
        for i in range(N):
            t = tans[i]
            u = (u - t * u.dot(t))
            u.normalize()
            w = t.cross(u)
            rr = r
            for k in range(n):
                a = 2 * PI * k / n
                p = P3[i] + (u * math.cos(a) + w * math.sin(a)) * rr
                v_.append((p.x, p.y, p.z))
        for i in range(N - 1):
            for k in range(n):
                k2 = (k + 1) % n
                a, b = i * n, (i + 1) * n
                f.append((a + k, a + k2, b + k2, b + k))
        if caps:
            c0 = len(v_)
            v_.append(tuple(P3[0]))
            v_.append(tuple(P3[-1]))
            for k in range(n):
                k2 = (k + 1) % n
                f.append((c0, k2, k))
                f.append((c0 + 1, (N - 1) * n + k, (N - 1) * n + k2))
        self._add(v_, f)
        return self

    def hemisphere(self, cx, cy, z0, r, nseg=10, nring=3, squash=1.0):
        """Upper hemisphere dome (bump cap) sitting on z0; squash scales height."""
        v = []
        f = []
        for j in range(nring + 1):
            el = (PI / 2) * j / nring
            for k in range(nseg):
                a = 2 * PI * k / nseg
                v.append((cx + r * math.cos(el) * math.cos(a), cy + r * math.cos(el) * math.sin(a), z0 + r * squash * math.sin(el)))
        for j in range(nring):
            for k in range(nseg):
                k2 = (k + 1) % nseg
                a, b = j * nseg, (j + 1) * nseg
                if j == nring - 1:
                    pass
                f.append((a + k, a + k2, b + k2, b + k))
        top = (nring) * nseg
        # top ring collapses to a fan: replace last quad ring by tris to a pole
        v.append((cx, cy, z0 + r * squash))
        pole = len(v) - 1
        # remove last quad ring (to the degenerate top ring), then fan
        f = f[: (nring - 1) * nseg]
        base = (nring - 1) * nseg
        for k in range(nseg):
            k2 = (k + 1) % nseg
            f.append((base + k, base + k2, pole))
        # bottom cap
        v.append((cx, cy, z0))
        c = len(v) - 1
        for k in range(nseg):
            k2 = (k + 1) % nseg
            f.append((c, k2, k))
        self._add(v, f)
        return self

    def sphere(self, cx, cy, cz, r, nseg=10, nring=6, sz=1.0):
        """Closed UV sphere (z squashed by sz)."""
        v = [(cx, cy, cz + r * sz)]
        f = []
        for j in range(1, nring):
            el = PI / 2 - PI * j / nring
            for k in range(nseg):
                a = 2 * PI * k / nseg
                v.append((cx + r * math.cos(el) * math.cos(a), cy + r * math.cos(el) * math.sin(a), cz + r * sz * math.sin(el)))
        v.append((cx, cy, cz - r * sz))
        for k in range(nseg):
            k2 = (k + 1) % nseg
            f.append((0, 1 + k2, 1 + k))
        for j in range(nring - 2):
            for k in range(nseg):
                k2 = (k + 1) % nseg
                a, b = 1 + j * nseg, 1 + (j + 1) * nseg
                f.append((a + k, a + k2, b + k2, b + k))
        last = len(v) - 1
        base = 1 + (nring - 2) * nseg
        for k in range(nseg):
            k2 = (k + 1) % nseg
            f.append((last, base + k, base + k2))
        f = [tuple(reversed(t)) for t in f]  # construction above is inward-facing
        self._add(v, f)
        return self

    def merge(self, other, dx=0.0, dy=0.0, dz=0.0):
        self._add([(x + dx, y + dy, z + dz) for (x, y, z) in other.v], other.f)
        return self

    def transformed(self, matrix):
        g = Geo()
        for (x, y, z) in self.v:
            p = matrix @ Vector((x, y, z))
            g.v.append((p.x, p.y, p.z))
        g.f = list(self.f)
        return g

    def count_tris(self):
        return sum(len(f) - 2 for f in self.f)

    def scale(self, k):
        self.v = [(x * k, y * k, z * k) for (x, y, z) in self.v]
        return self

    def build(self, name, material, coll, parent=None, loc=(0, 0, 0), smooth=False, bevel=None):
        me = bpy.data.meshes.new(name)
        me.from_pydata(self.v, [], self.f)
        me.update()
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        if material is not None:
            me.materials.append(material)
        if smooth:
            for p in me.polygons:
                p.use_smooth = True
        ob.location = loc
        if parent is not None:
            ob.parent = parent
        if bevel:
            C.bevel(ob, bevel[0], bevel[1] if len(bevel) > 1 else 2)
        return ob


def arc_pts(cx, cy, r, a0, a1, n=24):
    return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]


# ------------------------------------------------------------------------------------------------ instancing (Geometry Nodes)
def instancer(name, proto, pts, coll, parent=None, hide_proto=True):
    """Instance `proto` (a mesh object, origin = bump centre) on every point in pts (metres) via Geometry Nodes.

    Returns the point-cloud carrier object (renders the instances). The prototype is kept in `coll` but hidden.
    """
    me = bpy.data.meshes.new(name + "_pts")
    me.from_pydata([tuple(p) for p in pts], [], [])
    me.update()
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    ng = bpy.data.node_groups.new("NG_" + name, "GeometryNodeTree")
    ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = ng.nodes, ng.links
    gi = nodes.new("NodeGroupInput")
    go = nodes.new("NodeGroupOutput")
    ip = nodes.new("GeometryNodeInstanceOnPoints")
    oi = nodes.new("GeometryNodeObjectInfo")
    oi.transform_space = "ORIGINAL"
    oi.inputs["Object"].default_value = proto
    links.new(gi.outputs[0], ip.inputs["Points"])
    links.new(oi.outputs["Geometry"], ip.inputs["Instance"])
    links.new(ip.outputs["Instances"], go.inputs[0])
    md = ob.modifiers.new("instances", "NODES")
    md.node_group = ng
    if hide_proto:
        proto.hide_render = True
        proto.hide_viewport = True
    return ob


def proto_obj(name, geo, material, coll, parent=None, smooth=False):
    return geo.build(name, material, coll, parent=parent, smooth=smooth)


# ------------------------------------------------------------------------------------------------ fibers / ribbons
def catmull(pts, samples=8):
    """Catmull-Rom through 3D points (list of tuples) -> dense list of Vectors."""
    P3 = [Vector(p) for p in pts]
    P3 = [P3[0] * 2 - P3[1]] + P3 + [P3[-1] * 2 - P3[-2]]
    out = []
    for i in range(1, len(P3) - 2):
        p0, p1, p2, p3 = P3[i - 1], P3[i], P3[i + 1], P3[i + 2]
        for s in range(samples):
            t = s / samples
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P3[-2])
    return out


def offset_path(path, offset, up=(0, 0, 1)):
    """Offset a dense path sideways (horizontal normal) by `offset` metres (keeps ribbon parallel)."""
    U = Vector(up)
    res = []
    n = len(path)
    for i in range(n):
        t = (path[min(i + 1, n - 1)] - path[max(i - 1, 0)]).normalized()
        side = t.cross(U)
        if side.length < 1e-9:
            side = Vector((0, 1, 0))
        side.normalize()
        res.append(path[i] + side * offset)
    return res


def ribbon_fibers(path, n, pitch, r_coat, coll, parent, name, mat_fn=fiber_color_mat, n_sides=8, flat_color=None):
    """n coated fibers (radius r_coat) side by side along `path` (dense Vectors). Returns list of objects."""
    objs = []
    for i in range(n):
        off = (i - (n - 1) / 2) * pitch
        pp = offset_path(path, off)
        g = Geo().tube([(p.x, p.y, p.z) for p in pp], r_coat, n=n_sides)
        m = flat_color if flat_color is not None else mat_fn(i)
        o = g.build("%s_fiber_%02d" % (name, i), m, coll, parent, smooth=True)
        objs.append(o)
    return objs


def ribbon_matrix_jacket(path, n, pitch, r_coat, thick_extra, coll, parent, name, material):
    """Flat ribbon matrix: a thin extruded strip enveloping the fibers (width n*pitch)."""
    pts2 = [(p.x, p.y) for p in path]
    z = sum(p.z for p in path) / len(path)
    # follow z by building per-segment boxes is overkill; use strip at mean z when path is nearly planar
    g = Geo().strip(pts2, n * pitch, z - r_coat * 0.55, z + r_coat * 0.55)
    return g.build(name + "_ribbon_matrix", material, coll, parent)


# ------------------------------------------------------------------------------------------------ shared component builders
def v_groove_profile(pitch, depth):
    """Half-width at top and apex depth for a 70.53 deg included-angle (KOH <100> Si style) V-groove."""
    half = math.radians(35.26)
    return depth * math.tan(half)


def build_fau(coll, parent, name, n_fib, pitch, base_w=None, length=5.0 * MM, base_t=0.85 * MM, lid_t=0.5 * MM,
              fiber_len_out=2.0 * MM, groove_depth=90 * UM, with_coating=True, loc=(0, 0, 0), coat_len=None, epoxy=True):
    """Glass V-groove fiber array unit. Local frame: fibers run along +X (chip face at x=0 facing -X? no: the cleaved
    chip-facing end is at x=0, the block extends toward -X? ). Convention here: chip-facing end face at x = 0,
    block occupies x in [-length, 0], fibers exit toward -X. Callers rotate/mirror as needed.
    z = 0 is the underside of the glass base; fiber axis at z = base_t + (fiber centre above groove top).
    Returns dict with objects and key numbers.
    """
    r_f = 62.5 * UM
    half = math.radians(35.26)
    apex_to_center = r_f / math.sin(half)  # 108.2 um
    centre_above_top = apex_to_center - groove_depth  # fibre axis above the glass top surface
    z_axis = base_t + centre_above_top
    span = (n_fib - 1) * pitch
    bw = base_w or (span + 2.0 * MM)
    objs = {}
    root = bpy.data.objects.new(name + "_root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.002
    coll.objects.link(root)
    root.parent = parent
    root.location = loc
    # glass base with V-grooves (explicit profile extrusion along X)
    L = length
    verts = []
    faces = []

    def quad(a, b, c, d):
        faces.append((a, b, c, d))

    # build base as a single extruded 2D profile in the YZ plane (profile points include grooves), extruded along X
    prof = []
    prof.append((-bw / 2, 0.0))
    prof.append((bw / 2, 0.0))
    prof.append((bw / 2, base_t))
    wtop = 2 * groove_depth * math.tan(half)
    ys = [(i - (n_fib - 1) / 2) * pitch for i in range(n_fib)]
    right_to_left = []
    for yc in reversed(ys):
        right_to_left.append((yc + wtop / 2, base_t))
        right_to_left.append((yc, base_t - groove_depth))
        right_to_left.append((yc - wtop / 2, base_t))
    prof += right_to_left
    prof.append((-bw / 2, base_t))
    npf = len(prof)
    for (py, pz) in prof:
        verts.append((-L, py, pz))
    for (py, pz) in prof:
        verts.append((0.0, py, pz))
    # side faces
    for i in range(npf):
        j = (i + 1) % npf
        quad(i, j, npf + j, npf + i)
    # end caps (non-convex profile: triangulate with a fan about a lower-centre vertex is invalid; use triangulation by
    # splitting into strips: bottom rectangle and top comb)
    # bottom slab up to z = base_t - groove_depth - small: make caps via ear-clip using mathutils
    from mathutils.geometry import tessellate_polygon
    pv = [Vector((py, pz, 0.0)) for (py, pz) in prof]
    tris = tessellate_polygon([pv])
    for t in tris:
        a, b, c = t
        faces.append((a, c, b))  # x=-L cap (normal -x)  (winding checked visually)
        faces.append((npf + a, npf + b, npf + c))
    me = bpy.data.meshes.new(name + "_base")
    me.from_pydata(verts, [], faces)
    me.update()
    ob = bpy.data.objects.new(name + "_glass_base", me)
    coll.objects.link(ob)
    ob.parent = root
    me.materials.append(mat("glass_edge"))
    objs["base"] = ob
    # fibres: stripped glass part inside the block, coated part out
    glass_m = mat("fiber_glass")
    core_m = mat("fiber_core")
    fib_objs = []
    coat_len = coat_len if coat_len is not None else fiber_len_out
    for i, yc in enumerate(ys):
        g = Geo().tube([(0.0, yc, z_axis), (-L - 1.0 * MM, yc, z_axis)], r_f, n=14)
        fo = g.build("%s_fiber_%02d" % (name, i), glass_m, coll, root, smooth=True)
        fib_objs.append(fo)
        # core: tiny emissive disc at cleaved end face (visible at the end), radius 4.1 um
        gc = Geo()
        gc.tube([(-2 * UM, yc, z_axis), (-L + 0.2 * MM, yc, z_axis)], 4.1 * UM, n=8)
        co = gc.build("%s_core_%02d" % (name, i), core_m, coll, root, smooth=True)
        fib_objs.append(co)
        if with_coating:
            # coated 250 um fibre exits the block towards -X
            gcoat = Geo().tube([(-L - 1.0 * MM, yc, z_axis), (-L - 1.0 * MM - coat_len, yc, z_axis)], 125 * UM if pitch >= 200 * UM else 62.5 * UM, n=10)
            # when pitch is 127 um, coated fibres cannot sit at 127 um pitch; they fan-out (handled by caller); here keep bare
            co2 = gcoat.build("%s_coat_%02d" % (name, i), fiber_color_mat(i), coll, root, smooth=True)
            fib_objs.append(co2)
    objs["fibers"] = fib_objs
    # glass lid
    lid_w = span + 0.8 * MM if span + 0.8 * MM < bw - 0.4 * MM else bw - 0.4 * MM
    g = Geo().box(-L, -bw / 2, base_t + 0.0, 0.0, bw / 2, base_t + lid_t)
    # lid sits on the top surface of the base: z from base_t to base_t+lid_t; fibres protrude into the lid gap slightly
    lid = g.build(name + "_glass_lid", mat("glass"), coll, root, bevel=(0.03 * MM, 1))
    objs["lid"] = lid
    # epoxy bead at the rear end of the block
    ge = Geo().box(-L - 0.9 * MM, -bw / 2 + 0.25 * MM, base_t - 0.05 * MM, -L + 0.5 * MM, bw / 2 - 0.25 * MM, base_t + lid_t * 0.9)
    if epoxy:
        ep = ge.build(name + "_epoxy_bead", mat("epoxy"), coll, root, bevel=(0.15 * MM, 3))
        objs["epoxy"] = ep
    objs["root"] = root
    objs["z_axis"] = z_axis
    objs["base_w"] = bw
    objs["length"] = L
    objs["height"] = base_t + lid_t
    return objs


def build_mt_ferrule(coll, parent, name, n_fib=12, pitch=250 * UM, loc=(0, 0, 0), with_pins=True, pin_len=6.0 * MM,
                     body=(8.0 * MM, 6.4 * MM, 2.5 * MM), mt_mat=None, fibers_out=0.0):
    """MT ferrule per IEC 61754-5 / US Conec dims: 6.4 W x 2.5 H x 8.0 L mm, guide holes 0.7 mm at 4.6 mm pitch,
    fibre holes at `pitch`. Local frame: mating end face at x = 0, body toward -X; pins protrude toward +X.
    """
    L, W, H = body
    root = bpy.data.objects.new(name + "_root", None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.003
    coll.objects.link(root)
    root.parent = parent
    root.location = loc
    mt_mat = mt_mat or mat("ferrule")
    # body with front end-face recess for the fibre window and guide holes (simple: body + front chamfer)
    g = Geo().box(-L, -W / 2, -H / 2, 0.0, W / 2, H / 2)
    body_o = g.build(name + "_body", mt_mat, coll, root, bevel=(0.12 * MM, 2))
    # raised top window (epoxy window) on the rear half of the ferrule
    gw = Geo().box(-L + 1.2 * MM, -2.2 * MM, H / 2, -L + 4.2 * MM, 2.2 * MM, H / 2 + 0.01 * MM)
    win = gw.build(name + "_epoxy_window", mat("epoxy"), coll, root)
    # flange at the rear
    gf = Geo().box(-L, -W / 2 - 0.35 * MM, -H / 2 - 0.35 * MM, -L + 1.6 * MM, W / 2 + 0.35 * MM, H / 2 + 0.35 * MM)
    fl = gf.build(name + "_flange", mt_mat, coll, root, bevel=(0.1 * MM, 2))
    # guide holes (dark discs) and pins
    pin_r = 0.35 * MM
    pin_y = 2.3 * MM
    hole_m = mat("mold_black")
    parts = [body_o, win, fl]
    gh = Geo()
    for sy in (-1, 1):
        gh.cyl_axis((0.0005 * MM, sy * pin_y, 0), (0.0005 * MM + 0.001, sy * pin_y, 0), pin_r * 1.05, n=24)
    # fibre holes ring: small dark discs on end face
    gfh = Geo()
    for i in range(n_fib):
        yc = (i - (n_fib - 1) / 2) * pitch
        gfh.tube([(0.0004 * MM, yc, 0), (0.0006 * MM, yc, 0)], 63 * UM, n=10)
    fh = gfh.build(name + "_fiber_holes", hole_m, coll, root)
    parts.append(fh)
    if not with_pins:
        parts.append(gh.build(name + "_guide_holes", hole_m, coll, root))
    pins = []
    if with_pins:
        for sy in (-1, 1):
            gp = Geo().tube([(-L + 1.0 * MM, sy * pin_y, 0), (pin_len, sy * pin_y, 0)], pin_r, n=16)
            # rounded pin tip: small cone
            gp.tube([(pin_len, sy * pin_y, 0), (pin_len + 0.35 * MM, sy * pin_y, 0)], pin_r * 0.6, n=16)
            po = gp.build("%s_guide_pin_%s" % (name, "L" if sy < 0 else "R"), mat("steel"), coll, root, smooth=True)
            pins.append(po)
    # fibre end faces (cleaved/polished), slightly protruding 2 um
    gfe = Geo()
    for i in range(n_fib):
        yc = (i - (n_fib - 1) / 2) * pitch
        gfe.tube([(-0.5 * MM, yc, 0), (2.0 * UM, yc, 0)], 62.5 * UM, n=14)
    fe = gfe.build(name + "_fiber_ends", mat("fiber_glass"), coll, root, smooth=True)
    gce = Geo()
    for i in range(n_fib):
        yc = (i - (n_fib - 1) / 2) * pitch
        gce.tube([(0.0, yc, 0), (2.5 * UM, yc, 0)], 4.1 * UM, n=8)
    ce = gce.build(name + "_fiber_cores", mat("fiber_core"), coll, root, smooth=True)
    return dict(root=root, parts=parts + pins + [fe, ce], pins=pins)


# ------------------------------------------------------------------------------------------------ metadata / counting
def eval_tris(coll_or_objs):
    """Triangle count of evaluated meshes (includes Geometry-Nodes instances) for objects in a collection tree."""
    dg = bpy.context.evaluated_depsgraph_get()
    n = 0
    objs = []

    def walk(c):
        objs.extend(c.objects)
        for ch in c.children:
            walk(ch)
    if hasattr(coll_or_objs, "objects"):
        walk(coll_or_objs)
    else:
        objs = list(coll_or_objs)
    for o in objs:
        if o.type != "MESH" or o.hide_render and o.hide_viewport:
            continue
        ev = o.evaluated_get(dg)
        try:
            me = ev.to_mesh()
        except Exception:
            continue
        n += sum(len(p.vertices) - 2 for p in me.polygons)
        ev.to_mesh_clear()
        # instances
    for inst in dg.object_instances:
        if inst.is_instance and inst.object.type == "MESH":
            n += sum(len(p.vertices) - 2 for p in inst.object.data.polygons)
    return n


def dim(name, value, unit, source, level, note=""):
    return dict(item=name, value=value, unit=unit, source=source, accuracy=level, note=note)


# ------------------------------------------------------------------------------------------------ preview renderer
def _world(strength=1.0):
    w = bpy.data.worlds.new("PREVIEW_world")
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    bg = nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.new("ShaderNodeOutputWorld")
    mp = nt.nodes.new("ShaderNodeMapRange")
    mp.inputs["From Min"].default_value = -1.0
    mp.inputs["From Max"].default_value = 1.0
    nt.links.new(tc.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], mp.inputs["Value"])
    nt.links.new(mp.outputs["Result"], ramp.inputs["Fac"])
    cr = ramp.color_ramp
    cr.elements[0].position = 0.0
    cr.elements[0].color = (0.10, 0.11, 0.13, 1)
    cr.elements[1].position = 1.0
    cr.elements[1].color = (0.95, 0.97, 1.0, 1)
    e = cr.elements.new(0.5)
    e.color = (0.55, 0.58, 0.62, 1)
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bpy.context.scene.world = w
    return w


def render_shots(show_colls, out_dir, prefix, shots, res=(900, 675), samples=16, floor=None, sun_energy=2.2, world=0.45,
                 bg_color=None):
    """Render named camera shots of the objects in `show_colls` (collections); everything else hidden from render.

    shots: list of dict(name, target=(x,y,z) m, dist m, az deg (0 = from -Y, +90 = from +X), el deg, lens mm).
    floor: z of a ground plane (metres) or None; its size follows the largest shot distance.
    """
    os.makedirs(out_dir, exist_ok=True)
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = samples
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.compression = 80
    scn.view_settings.view_transform = "Standard"
    keep = set()
    for c in show_colls:
        for cc in [c] + list(c.children_recursive):
            for o in cc.objects:
                keep.add(o)
    # include parents of kept objects (empties)
    for o in list(keep):
        p = o.parent
        while p is not None:
            keep.add(p)
            p = p.parent
    created = []
    for o in scn.objects:
        o.hide_render = o not in keep
    _world(world)
    big = max(s["dist"] for s in shots)
    sun = bpy.data.lights.new("PREVIEW_sun", "SUN")
    sun.energy = sun_energy
    sun.angle = 0.1
    so = bpy.data.objects.new("PREVIEW_sun", sun)
    so.rotation_euler = (0.85, 0.2, 0.6)
    scn.collection.objects.link(so)
    sun2 = bpy.data.lights.new("PREVIEW_sun2", "SUN")
    sun2.energy = sun_energy * 0.45
    so2 = bpy.data.objects.new("PREVIEW_sun2", sun2)
    so2.rotation_euler = (1.1, -0.3, -2.2)
    scn.collection.objects.link(so2)
    if floor is not None:
        bpy.ops.mesh.primitive_plane_add(size=big * 30, location=(0, 0, floor))
        fl = bpy.context.active_object
        fl.name = "PREVIEW_floor"
        fm = C.principled("PREVIEW_floor_mat", bg_color or (0.5, 0.52, 0.5), rough=0.9)
        fl.data.materials.append(fm)
    cam_d = bpy.data.cameras.new("PREVIEW_cam")
    cam = bpy.data.objects.new("PREVIEW_cam", cam_d)
    scn.collection.objects.link(cam)
    scn.camera = cam
    tgt = bpy.data.objects.new("PREVIEW_tgt", None)
    scn.collection.objects.link(tgt)
    cons = cam.constraints.new("TRACK_TO")
    cons.track_axis = "TRACK_NEGATIVE_Z"
    cons.up_axis = "UP_Y"
    cons.target = tgt
    cam_d.sensor_width = 36
    outs = []
    for s in shots:
        az = math.radians(s.get("az", 0))
        el = math.radians(s.get("el", 15))
        t = Vector(s["target"])
        d = s["dist"]
        cam.location = t + Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * d
        tgt.location = t
        cam_d.lens = s.get("lens", 50)
        cam_d.clip_start = d * 0.002
        cam_d.clip_end = d * 200
        bpy.context.view_layer.update()
        path = os.path.join(out_dir, "%s_%s.png" % (prefix, s["name"]))
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        shrink_png(path)
        outs.append(path)
    for o in [so, so2, cam, tgt] + ([fl] if floor is not None else []):
        bpy.data.objects.remove(o, do_unlink=True)
    return outs


def shrink_png(path, limit=330_000):
    """Keep preview PNGs small (spec: under ~400 KB): palette-quantise with ImageMagick if above `limit`."""
    import subprocess
    try:
        if os.path.getsize(path) <= limit:
            return
        for ncol in (1024, 512, 256):
            tmp = path + ".tmp.png"
            subprocess.run(["magick", path, "-strip", "-colors", str(ncol), "-define", "png:compression-level=9", tmp],
                           check=True, stderr=subprocess.DEVNULL)
            os.replace(tmp, path)
            if os.path.getsize(path) <= limit:
                break
    except Exception as e:  # pragma: no cover
        print("shrink_png failed", e)


def standard_shots(center, size, extra=None, lens=50):
    """front / three_quarter / top shots framing a bbox given by centre (m) and size vector (m)."""
    R = max(size) * 0.5
    d = R * 3.0 * (50.0 / lens)
    shots = [
        dict(name="front", target=center, dist=d, az=0, el=12, lens=lens),
        dict(name="three_quarter", target=center, dist=d, az=38, el=30, lens=lens),
        dict(name="top", target=center, dist=d, az=0, el=89, lens=lens),
    ]
    if extra:
        shots += extra
    return shots


def meta_base(asset_id, collection_name, notes, sources, dims, hooks, mats, props, usage, simplifications, extra=None):
    m = dict(
        asset_id=asset_id,
        collection=collection_name,
        built="2026-10-02",
        category="photonics",
        notes=notes,
        sources=sources,
        dimension_table=dims,
        hooks=hooks,
        material_slots=mats,
        custom_properties=props,
        intended_scene_usage=usage,
        simplifications=simplifications,
    )
    if extra:
        m.update(extra)
    return m


def used_material_names():
    return sorted(m.name for m in bpy.data.materials if m.name.startswith("MAT_photonics_") and m.users > 0)


# ------------------------------------------------------------------------------------------------ section cutting
def cut_object(ob, axis, value, keep_positive=True, cap=True):
    """Cut a mesh object with the WORLD plane axis=value (axis 0/1/2), keep one side, cap the cut. Handles parent transforms."""
    import bmesh
    from mathutils import Matrix
    me = ob.data
    bm = bmesh.new()
    bm.from_mesh(me)
    mw = ob.matrix_world
    mi = mw.inverted()
    co_w = Vector((0, 0, 0))
    co_w[axis] = value
    no_w = Vector((0, 0, 0))
    no_w[axis] = 1.0
    co = mi @ co_w
    no = (mw.to_3x3().transposed() @ no_w).normalized()
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-12, plane_co=co, plane_no=no,
                           clear_inner=keep_positive, clear_outer=not keep_positive)
    if cap:
        edges = [e for e in bm.edges if e.is_boundary]
        if edges:
            try:
                bmesh.ops.holes_fill(bm, edges=edges, sides=64)
            except Exception:
                pass
    if len(bm.verts) == 0:
        bm.free()
        return False
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    me.update()
    return True


def cut_collection(coll, axis, value, keep_positive=True):
    for o in list(coll.all_objects):
        if o.type != "MESH" or o.modifiers and any(m.type == "NODES" for m in o.modifiers):
            continue
        ok = cut_object(o, axis, value, keep_positive)
        if not ok:
            bpy.data.objects.remove(o, do_unlink=True)


def refresh():
    """Update matrix_world of all objects (needed before bbox_mm / counting when parents have offsets)."""
    bpy.context.view_layer.update()
