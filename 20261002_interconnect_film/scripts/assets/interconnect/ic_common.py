"""Helpers private to the interconnect category (builds on scripts/assets/_common/common.py).

All coordinates passed to the MB (mesh builder) methods are in millimetres; they are converted to metres
(Blender units) when vertices are created. Axes follow ASSET_SPEC: Z up, -Y front, +X right.
"""
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
import common as C  # noqa: E402

mm = C.mm
PREFIX = "MAT_interconnect_"


# ------------------------------------------------------------------ materials
class Mats:
    """Lazy material library (all prefixed MAT_interconnect_)."""

    SPECS = {
        # name: (base rgb, metallic, roughness, extra kwargs)
        "nickel": ((0.60, 0.61, 0.62), 1.0, 0.30, {}),
        "nickel_brushed": ((0.55, 0.56, 0.57), 1.0, 0.45, {}),
        "black_alu": ((0.035, 0.035, 0.04), 0.9, 0.42, {}),
        "steel": ((0.45, 0.46, 0.48), 1.0, 0.35, {}),
        "gold": ((1.0, 0.72, 0.25), 1.0, 0.22, {}),
        "copper": ((0.95, 0.52, 0.36), 1.0, 0.28, {}),
        "tinned_copper": ((0.78, 0.80, 0.82), 1.0, 0.3, {}),
        "foil": ((0.85, 0.87, 0.9), 1.0, 0.18, {}),
        "solder": ((0.7, 0.72, 0.75), 1.0, 0.35, {}),
        "fr4_green": ((0.02, 0.26, 0.09), 0.0, 0.42, {}),
        "fr4_blue": ((0.03, 0.12, 0.5), 0.0, 0.42, {}),
        "fr4_core": ((0.55, 0.5, 0.3), 0.0, 0.7, {}),
        "mold_black": ((0.012, 0.012, 0.014), 0.0, 0.55, {}),
        "ceramic_tan": ((0.42, 0.30, 0.17), 0.0, 0.55, {}),
        "inductor": ((0.06, 0.06, 0.07), 0.3, 0.4, {}),
        "silicon": ((0.14, 0.15, 0.2), 0.7, 0.22, {}),
        "thermal_pad": ((0.62, 0.69, 0.78), 0.0, 0.85, {}),
        "glass": ((0.95, 0.97, 1.0), 0.0, 0.04, {"transmission": 1.0, "ior": 1.5}),
        "potting": ((0.95, 0.72, 0.05), 0.0, 0.15, {"alpha": 0.55, "ior": 1.5}),
        "fiber_coat": ((0.92, 0.92, 0.9), 0.0, 0.35, {"alpha": 0.8}),
        "fiber_glass": ((0.9, 0.95, 1.0), 0.0, 0.02, {"transmission": 1.0, "ior": 1.46}),
        "fiber_core": ((0.5, 0.9, 1.0), 0.0, 0.1, {"emit": (0.3, 0.8, 1.0), "emit_strength": 3.0}),
        "ferrule_ceramic": ((0.93, 0.93, 0.9), 0.0, 0.25, {}),
        "mt_ferrule": ((0.82, 0.78, 0.62), 0.0, 0.4, {}),
        "plastic_black": ((0.02, 0.02, 0.022), 0.0, 0.45, {}),
        "plastic_grey": ((0.35, 0.36, 0.38), 0.0, 0.5, {}),
        "plastic_white": ((0.85, 0.85, 0.82), 0.0, 0.45, {}),
        "plastic_aqua": ((0.0, 0.62, 0.62), 0.0, 0.4, {}),
        "plastic_blue": ((0.03, 0.1, 0.6), 0.0, 0.4, {}),
        "plastic_green": ((0.05, 0.6, 0.1), 0.0, 0.4, {}),
        "plastic_orange": ((0.95, 0.35, 0.03), 0.0, 0.4, {}),
        "plastic_yellow": ((0.9, 0.7, 0.03), 0.0, 0.4, {}),
        "plastic_red": ((0.7, 0.03, 0.03), 0.0, 0.4, {}),
        "plastic_beige": ((0.8, 0.75, 0.6), 0.0, 0.45, {}),
        "tab_orange": ((0.95, 0.38, 0.03), 0.0, 0.5, {}),
        "tab_blue": ((0.03, 0.18, 0.75), 0.0, 0.5, {}),
        "jacket_black": ((0.025, 0.025, 0.028), 0.0, 0.6, {}),
        "jacket_grey": ((0.3, 0.3, 0.32), 0.0, 0.6, {}),
        "jacket_yellow": ((0.95, 0.72, 0.02), 0.0, 0.45, {}),
        "jacket_aqua": ((0.0, 0.58, 0.58), 0.0, 0.45, {}),
        "jacket_orange": ((0.95, 0.35, 0.03), 0.0, 0.45, {}),
        "jacket_blue": ((0.05, 0.15, 0.75), 0.0, 0.45, {}),
        "jacket_white": ((0.9, 0.9, 0.88), 0.0, 0.5, {}),
        "jacket_green": ((0.1, 0.6, 0.15), 0.0, 0.45, {}),
        "jacket_red": ((0.8, 0.05, 0.05), 0.0, 0.45, {}),
        "jacket_violet": ((0.5, 0.1, 0.7), 0.0, 0.45, {}),
        "dielectric": ((0.92, 0.9, 0.78), 0.0, 0.4, {}),
        "dielectric_blue": ((0.35, 0.55, 0.95), 0.0, 0.4, {}),
        "dielectric_orange": ((0.95, 0.5, 0.1), 0.0, 0.4, {}),
        "braid": ((0.7, 0.72, 0.74), 1.0, 0.45, {}),
        "label_white": ((0.9, 0.9, 0.88), 0.0, 0.6, {}),
        "ink_black": ((0.01, 0.01, 0.012), 0.0, 0.8, {}),
        "led_green": ((0.1, 1.0, 0.2), 0.0, 0.2, {"emit": (0.1, 1.0, 0.2), "emit_strength": 2.0}),
        "lightpipe": ((0.9, 1.0, 0.95), 0.0, 0.05, {"transmission": 0.9, "ior": 1.49}),
        "velcro_black": ((0.02, 0.02, 0.02), 0.0, 0.9, {}),
        "cable_tie": ((0.9, 0.9, 0.88), 0.0, 0.5, {}),
        "galv_steel": ((0.58, 0.6, 0.62), 1.0, 0.4, {}),
        "painted_grey": ((0.2, 0.21, 0.23), 0.0, 0.55, {}),
        "painted_black": ((0.03, 0.03, 0.035), 0.0, 0.5, {}),
        "dark_void": ((0.0, 0.0, 0.0), 0.0, 1.0, {}),
        "rubber_boot": ((0.015, 0.015, 0.017), 0.0, 0.7, {}),
        "vcsel": ((0.9, 0.55, 0.2), 1.0, 0.3, {}),
        "nvl_cartridge": ((0.5, 0.52, 0.55), 1.0, 0.45, {}),
        "nvl_sleeve": ((0.05, 0.05, 0.055), 0.0, 0.85, {}),
        "nvl_connector": ((0.08, 0.08, 0.09), 0.0, 0.5, {}),
    }

    def __init__(self):
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            base, met, rough, kw = self.SPECS[name]
            m = C.principled(PREFIX + name, base, metallic=met, rough=rough, **kw)
            self.cache[name] = m
        return self.cache[name]

    def __getitem__(self, name):
        return self.get(name)

    def named(self, name, base, metallic=0.0, rough=0.5, **kw):
        if name not in self.cache:
            self.cache[name] = C.principled(PREFIX + name, base, metallic=metallic, rough=rough, **kw)
        return self.cache[name]


# ------------------------------------------------------------------ mesh builder (mm input)
def _v(x, y, z):
    return (x * 0.001, y * 0.001, z * 0.001)


def rounded_rect(w, h, r, n=4, cx=0.0, cy=0.0):
    """Polygon (list of (x, y)) of a rounded rectangle, counter-clockwise."""
    r = min(r, w / 2 - 1e-6, h / 2 - 1e-6)
    pts = []
    corners = [(w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)]
    for ccx, ccy, a0 in corners:
        if r <= 1e-9:
            pts.append((ccx + cx, ccy + cy))
            continue
        for k in range(n + 1):
            a = math.radians(a0 + 90.0 * k / n)
            pts.append((ccx + r * math.cos(a) + cx, ccy + r * math.sin(a) + cy))
    return pts


class MB:
    """Mesh builder collecting geometry in one bmesh; build() returns an unlinked-from-asset Blender object."""

    def __init__(self):
        self.bm = bmesh.new()

    # -- boxes
    def box(self, xr, yr, zr, bev=0.0, seg=2):
        x0, x1 = xr
        y0, y1 = yr
        z0, z1 = zr
        sx, sy, sz = (x1 - x0), (y1 - y0), (z1 - z0)
        M = Matrix.Translation(Vector(_v((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))) @ Matrix.Diagonal((sx * 0.001, sy * 0.001, sz * 0.001, 1.0))
        ret = bmesh.ops.create_cube(self.bm, size=1.0, matrix=M)
        verts = ret["verts"]
        if bev > 0:
            b = min(bev, min(sx, sy, sz) * 0.45)
            edges = list({e for v in verts for e in v.link_edges})
            bmesh.ops.bevel(self.bm, geom=edges, offset=b * 0.001, offset_type="OFFSET", segments=seg, profile=0.5, affect="EDGES")
        return verts

    def boxes(self, items, bev=0.0):
        """items: list of (cx, cy, cz, sx, sy, sz) in mm."""
        for cx, cy, cz, sx, sy, sz in items:
            self.box((cx - sx / 2, cx + sx / 2), (cy - sy / 2, cy + sy / 2), (cz - sz / 2, cz + sz / 2), bev=bev, seg=1)

    # -- cylinders / cones (axis aligned)
    def cyl(self, c, r, length, axis="Z", segs=24, r2=None, caps=True, rot_deg=0.0):
        r2 = r if r2 is None else r2
        cx, cy, cz = c
        if axis == "Z":
            R = Matrix.Rotation(math.radians(rot_deg), 4, "Z")
        elif axis == "X":
            R = Matrix.Rotation(math.radians(90), 4, "Y") @ Matrix.Rotation(math.radians(rot_deg), 4, "Z")
        else:  # axis Y: cone axis +Z maps to -Y so radius1 sits at +Y... use explicit mapping below
            R = Matrix.Rotation(math.radians(-90), 4, "X") @ Matrix.Rotation(math.radians(rot_deg), 4, "Z")
        M = Matrix.Translation(Vector(_v(cx, cy, cz))) @ R
        ret = bmesh.ops.create_cone(self.bm, cap_ends=caps, cap_tris=False, segments=segs, radius1=r * 0.001, radius2=r2 * 0.001,
                                    depth=length * 0.001, matrix=M)
        return ret["verts"]

    def sphere(self, c, r, segs=12, rings=8, sx=1.0, sy=1.0, sz=1.0):
        M = Matrix.Translation(Vector(_v(*c))) @ Matrix.Diagonal((sx, sy, sz, 1.0))
        ret = bmesh.ops.create_uvsphere(self.bm, u_segments=segs, v_segments=rings, radius=r * 0.001, matrix=M)
        return ret["verts"]

    # -- prism: polygon in the XZ plane (list of (x, z)) extruded along Y between y0 and y1
    def prism_xz(self, poly, y0, y1):
        bm = self.bm
        a = [bm.verts.new(_v(x, y0, z)) for x, z in poly]
        b = [bm.verts.new(_v(x, y1, z)) for x, z in poly]
        n = len(poly)
        bm.faces.new(a[::-1])
        bm.faces.new(b)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((a[i], a[j], b[j], b[i]))
        return a + b

    # -- prism: polygon in XY plane extruded along Z
    def prism_xy(self, poly, z0, z1):
        bm = self.bm
        a = [bm.verts.new(_v(x, y, z0)) for x, y in poly]
        b = [bm.verts.new(_v(x, y, z1)) for x, y in poly]
        n = len(poly)
        bm.faces.new(a[::-1])
        bm.faces.new(b)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((a[i], a[j], b[j], b[i]))
        return a + b

    # -- prism: polygon in the YZ plane (list of (y, z)) extruded along X between x0 and x1
    def prism_yz(self, poly, x0, x1):
        bm = self.bm
        a = [bm.verts.new(_v(x0, y, z)) for y, z in poly]
        b = [bm.verts.new(_v(x1, y, z)) for y, z in poly]
        n = len(poly)
        bm.faces.new(a[::-1])
        bm.faces.new(b)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((a[i], a[j], b[j], b[i]))
        return a + b

    # -- ring (annulus) prism in XY: outer and inner polygons with the same vertex count, extruded along Z
    def ring_xy(self, outer, inner, z0, z1):
        bm = self.bm
        n = len(outer)
        assert len(inner) == n
        oa = [bm.verts.new(_v(x, y, z0)) for x, y in outer]
        ob = [bm.verts.new(_v(x, y, z1)) for x, y in outer]
        ia = [bm.verts.new(_v(x, y, z0)) for x, y in inner]
        ib = [bm.verts.new(_v(x, y, z1)) for x, y in inner]
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((oa[i], oa[j], ob[j], ob[i]))      # outer wall
            bm.faces.new((ia[j], ia[i], ib[i], ib[j]))      # inner wall
            bm.faces.new((ob[i], ob[j], ib[j], ib[i]))      # top strip
            bm.faces.new((oa[j], oa[i], ia[i], ia[j]))      # bottom strip

    # -- ring in the XZ plane (outer/inner polygons of (x, z), equal vertex counts) extruded along Y
    def ring_xz(self, outer, inner, y0, y1):
        bm = self.bm
        n = len(outer)
        assert len(inner) == n
        oa = [bm.verts.new(_v(x, y0, z)) for x, z in outer]
        ob = [bm.verts.new(_v(x, y1, z)) for x, z in outer]
        ia = [bm.verts.new(_v(x, y0, z)) for x, z in inner]
        ib = [bm.verts.new(_v(x, y1, z)) for x, z in inner]
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((oa[j], oa[i], ob[i], ob[j]))
            bm.faces.new((ia[i], ia[j], ib[j], ib[i]))
            bm.faces.new((oa[i], oa[j], ia[j], ia[i]))
            bm.faces.new((ob[j], ob[i], ib[i], ib[j]))

    # -- loft along +Y between superellipse sections: sections = [(y, w, h, exponent)], mm
    def loft_y(self, sections, npts=28, cx=0.0, cz=0.0):
        bm = self.bm
        rings = []
        th = np.linspace(0, 2 * np.pi, npts, endpoint=False)
        for (y, w, h, e) in sections:
            c, s_ = np.cos(th), np.sin(th)
            x = cx + (w / 2) * np.sign(c) * np.abs(c) ** (2.0 / e)
            z = cz + (h / 2) * np.sign(s_) * np.abs(s_) ** (2.0 / e)
            rings.append([bm.verts.new(_v(x[k], y, z[k])) for k in range(npts)])
        for i in range(len(rings) - 1):
            for k in range(npts):
                k2 = (k + 1) % npts
                bm.faces.new((rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]))
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])
        return rings

    # -- arbitrary quads/tris (mm coords)
    def face(self, pts):
        vs = [self.bm.verts.new(_v(*p)) for p in pts]
        return self.bm.faces.new(vs)

    def translate(self, verts, dx=0, dy=0, dz=0):
        bmesh.ops.translate(self.bm, verts=verts, vec=Vector(_v(dx, dy, dz)))

    def build(self, name, mat=None, smooth_deg=35.0):
        bm = self.bm
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        if smooth_deg:
            ang = math.radians(smooth_deg)
            for f in bm.faces:
                f.smooth = True
            for e in bm.edges:
                if len(e.link_faces) == 2:
                    e.smooth = e.calc_face_angle(0.0) < ang
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        if mat is not None:
            ob.data.materials.append(mat)
        return ob


def place(obj, coll, parent, loc_mm=None, rot=None, scale=None):
    C.add(obj, coll, parent)
    if loc_mm is not None:
        obj.location = Vector(_v(*loc_mm))
    if rot is not None:
        obj.rotation_euler = rot
    if scale is not None:
        obj.scale = scale
    return obj


def empty(name, coll, parent, loc_mm=(0, 0, 0), rot=(0, 0, 0), kind="PLAIN_AXES", size=0.01):
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = kind
    e.empty_display_size = size
    coll.objects.link(e)
    e.parent = parent
    e.location = Vector(_v(*loc_mm))
    e.rotation_euler = rot
    return e


# ------------------------------------------------------------------ numpy tubes (cables, fibers)
def catmull(points, per_seg=8, closed=False):
    """Uniform Catmull-Rom through points (n,3) -> (m,3)."""
    P = np.asarray(points, dtype=float)
    if len(P) < 3:
        t = np.linspace(0, 1, per_seg + 1)[:, None]
        return P[0] * (1 - t) + P[-1] * t
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    t = np.linspace(0, 1, per_seg, endpoint=False)[:, None]
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        seg = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)
        out.append(seg)
    out.append(P[-1][None, :])
    return np.vstack(out)


def tube_mesh(name, paths, radius, sides=8, mat=None, caps=False, smooth=True):
    """Build ONE mesh holding many tubes. paths: (m, n, 3) array in METRES (or list of (n,3)); radius: scalar, (m,) or (m,n) in METRES."""
    P = np.asarray(paths, dtype=np.float64)
    if P.ndim == 2:
        P = P[None]
    m, n, _ = P.shape
    R = np.broadcast_to(np.asarray(radius, dtype=np.float64), (m, n)) if np.ndim(radius) < 2 else np.asarray(radius)
    if np.ndim(radius) == 1:
        R = np.repeat(np.asarray(radius)[:, None], n, axis=1)
    T = np.zeros_like(P)
    T[:, 1:-1] = P[:, 2:] - P[:, :-2]
    T[:, 0] = P[:, 1] - P[:, 0]
    T[:, -1] = P[:, -1] - P[:, -2]
    Tn = np.linalg.norm(T, axis=2, keepdims=True)
    T = T / np.maximum(Tn, 1e-12)
    # parallel transport
    ref = np.zeros((m, 3))
    ax = np.argmin(np.abs(T[:, 0]), axis=1)
    ref[np.arange(m), ax] = 1.0
    N = np.cross(T[:, 0], ref)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    Ns = np.zeros_like(P)
    Ns[:, 0] = N
    for i in range(1, n):
        b = np.cross(T[:, i - 1], T[:, i])
        bn = np.linalg.norm(b, axis=1, keepdims=True)
        ok = bn[:, 0] > 1e-9
        bhat = np.where(ok[:, None], b / np.maximum(bn, 1e-12), 0.0)
        ang = np.arctan2(bn[:, 0], np.sum(T[:, i - 1] * T[:, i], axis=1))
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        v = N
        N = v * c + np.cross(bhat, v) * s + bhat * np.sum(bhat * v, axis=1, keepdims=True) * (1 - c)
        N -= T[:, i] * np.sum(N * T[:, i], axis=1, keepdims=True)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        Ns[:, i] = N
    Bs = np.cross(T, Ns)
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    ring = (Ns[:, :, None, :] * ct[None, None, :, None] + Bs[:, :, None, :] * st[None, None, :, None]) * R[:, :, None, None]
    V = P[:, :, None, :] + ring  # (m,n,sides,3)
    nv_tube = n * sides
    verts = V.reshape(-1, 3)
    idx = np.arange(m * nv_tube).reshape(m, n, sides)
    a = idx[:, :-1, :]
    b = idx[:, :-1, (np.arange(sides) + 1) % sides]
    c2 = idx[:, 1:, (np.arange(sides) + 1) % sides]
    d = idx[:, 1:, :]
    F = np.stack([a, b, c2, d], axis=-1).reshape(-1, 4)
    extra_faces = []
    if caps:
        base = m * nv_tube
        cap_verts = []
        cap_faces = []
        for end in (0, n - 1):
            ctr = P[:, end]
            cap_verts.append(ctr)
        verts = np.vstack([verts, cap_verts[0], cap_verts[1]])
        ci0 = base + np.arange(m)
        ci1 = base + m + np.arange(m)
        for s in range(sides):
            s2 = (s + 1) % sides
            f0 = np.stack([ci0, idx[:, 0, s2], idx[:, 0, s], ci0], axis=1)
            f1 = np.stack([ci1, idx[:, n - 1, s], idx[:, n - 1, s2], ci1], axis=1)
            extra_faces.append(f0)
            extra_faces.append(f1)
    me = bpy.data.meshes.new(name)
    nvt = len(verts)
    me.vertices.add(nvt)
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    # quads + degenerate-quad caps encoded as triangles
    quads = F
    nq = len(quads)
    tris = np.vstack([e[:, :3] for e in extra_faces]) if extra_faces else np.zeros((0, 3), dtype=np.int64)
    nt = len(tris)
    me.loops.add(nq * 4 + nt * 3)
    loops = np.concatenate([quads.ravel(), tris.ravel()])
    me.loops.foreach_set("vertex_index", loops.astype(np.int32))
    me.polygons.add(nq + nt)
    starts = np.concatenate([np.arange(nq) * 4, nq * 4 + np.arange(nt) * 3])
    me.polygons.foreach_set("loop_start", starts.astype(np.int32))
    if smooth:
        me.polygons.foreach_set("use_smooth", np.ones(nq + nt, dtype=bool))
    me.update(calc_edges=True)
    me.validate()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if mat is not None:
        ob.data.materials.append(mat)
    return ob


# ------------------------------------------------------------------ drivers / props
def add_prop(root, name, value, lo=None, hi=None, doc=""):
    root[name] = value
    try:
        ui = root.id_properties_ui(name)
        kw = {}
        if lo is not None:
            kw["min"] = lo
        if hi is not None:
            kw["max"] = hi
        if doc:
            kw["description"] = doc
        ui.update(**kw)
    except Exception:
        pass


def drive_prop(target_obj, data_path, index, root, prop, expr_scale=1.0, offset=0.0, expr=None):
    """Driver: target.data_path[index] = prop * scale + offset (or custom expr using var 'p')."""
    fc = target_obj.driver_add(data_path, index) if index is not None else target_obj.driver_add(data_path)
    d = fc.driver
    d.type = "SCRIPTED"
    var = d.variables.new()
    var.name = "p"
    var.type = "SINGLE_PROP"
    var.targets[0].id = root
    var.targets[0].data_path = '["%s"]' % prop
    d.expression = expr if expr is not None else "p*%r+%r" % (expr_scale, offset)
    return fc


# ------------------------------------------------------------------ text
def text_obj(name, body, size_mm, loc_mm, rot=(0, 0, 0), mat=None, extrude_mm=0.02, align="CENTER", convert=True):
    o = C.text_mesh(name, body, mm(size_mm), loc=_v(*loc_mm), rot=rot, extrude=mm(extrude_mm), mat=mat, align=align)
    if convert:
        bpy.ops.object.select_all(action="DESELECT")
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.convert(target="MESH")
        o = bpy.context.active_object
        o.name = name
    return o


# ------------------------------------------------------------------ misc
def hide_collection(c, hide=True):
    c.hide_render = hide
    c.hide_viewport = hide


def link_dup(obj, coll, parent, loc_mm, rot=None, name=None):
    """Linked duplicate (shares mesh data) of obj."""
    o = bpy.data.objects.new(name or obj.name + "_i", obj.data)
    coll.objects.link(o)
    o.parent = parent
    o.location = Vector(_v(*loc_mm))
    if rot is not None:
        o.rotation_euler = rot
    return o


def write_extra_meta(blend_path, extra):
    import json
    jp = os.path.splitext(blend_path)[0] + ".json"
    with open(jp) as f:
        meta = json.load(f)
    meta.update(extra)
    with open(jp, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta


def preview_views(coll, out_dir, asset_id, specs, res=(900, 675), samples=16, floor=True, lens=50, sun=1.3, world=0.35):
    """Custom camera previews. specs: list of (view_name, target_mm(x,y,z), direction(x,y,z), distance_mm, [lens]).
    Reuses studio lights from common. Call after saving the .blend."""
    os.makedirs(out_dir, exist_ok=True)
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
    bb = C.bbox_mm(coll)
    lo, hi = Vector(bb[0]) / 1000.0, Vector(bb[1]) / 1000.0
    center = (lo + hi) / 2
    radius = max(((hi - lo).length) / 2, 0.01)
    st = C._studio(center, radius) if floor else []
    for o in st:
        if o.name.startswith("PREVIEW_sun"):
            o.data.energy = sun
        if o.name.startswith("PREVIEW_fill"):
            o.data.energy *= 0.5
    scn.world.node_tree.nodes["Background"].inputs["Strength"].default_value = world
    cam_d = bpy.data.cameras.new("PREVIEW_cam")
    cam = bpy.data.objects.new("PREVIEW_cam", cam_d)
    scn.collection.objects.link(cam)
    scn.camera = cam
    cons = cam.constraints.new("TRACK_TO")
    cons.track_axis = "TRACK_NEGATIVE_Z"
    cons.up_axis = "UP_Y"
    tgt = bpy.data.objects.new("PREVIEW_tgt", None)
    scn.collection.objects.link(tgt)
    cons.target = tgt
    cam_d.sensor_width = 36
    outs = []
    for spec in specs:
        name, tmm, dirv, dist = spec[:4]
        cam_d.lens = spec[4] if len(spec) > 4 else lens
        cam_d.clip_start = 1e-4
        cam_d.clip_end = 100
        t = Vector(_v(*tmm))
        tgt.location = t
        d = Vector(dirv).normalized()
        cam.location = t + d * (dist * 0.001)
        path = os.path.join(out_dir, "%s_%s.png" % (asset_id, name))
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        outs.append(path)
    return outs


def unique_tris(coll):
    """Triangles counting each distinct mesh datablock once (linked duplicates share data)."""
    seen = set()
    n = 0

    def walk(c):
        nonlocal n
        for o in c.objects:
            if o.type == "MESH" and o.data.name not in seen:
                seen.add(o.data.name)
                n += sum(len(p.vertices) - 2 for p in o.data.polygons)
        for ch in c.children:
            walk(ch)
    walk(coll)
    return n
