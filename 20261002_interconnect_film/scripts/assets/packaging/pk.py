"""Packaging category helpers (Blender 4.2, headless). Sits on top of the shared scripts/assets/_common/common.py.

All pk geometry functions take MILLIMETRES (locations, sizes) and convert to metres (1 BU = 1 m) internally.

Provides:
  - MB: bmesh-based mesh builder (box with bevel, cylinder/cone, lathe, rounded-rect prism, polygon prism, ribbon)
  - scatter(): instancing of a source object on a point set with a Geometry Nodes group (balls, pads, MLCCs ...)
  - mat(): cached packaging material library (MAT_packaging_<name>)
  - text(): text as a real mesh (flat, shallow extrusion)
  - Dims / meta helpers, eval_stats() (bbox + triangle count including GN instances)
  - render_views(): previews with explicit cameras (asset-local mm coordinates)
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

MM = 1e-3
PI = math.pi
bpy.context.preferences.filepaths.save_version = 0


def mmv(v):
    return Vector((v[0] * MM, v[1] * MM, v[2] * MM))


# ------------------------------------------------------------------------------------------------ materials
_MATS = {}


def _noise_bump(m, scale=400.0, strength=0.05, detail=6.0):
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    no = nt.nodes.new("ShaderNodeTexNoise")
    no.inputs["Scale"].default_value = scale
    no.inputs["Detail"].default_value = detail
    bm = nt.nodes.new("ShaderNodeBump")
    bm.inputs["Strength"].default_value = strength
    bm.inputs["Distance"].default_value = 0.0005
    nt.links.new(tc.outputs["Object"], no.inputs["Vector"])
    nt.links.new(no.outputs["Fac"], bm.inputs["Height"])
    nt.links.new(bm.outputs["Normal"], b.inputs["Normal"])


_SPEC = {
    # name: (base, metallic, rough, kwargs)
    "mask_green": ((0.012, 0.14, 0.05), 0.0, 0.32, dict(coat=0.3)),
    "mask_blue": ((0.02, 0.07, 0.30), 0.0, 0.32, dict(coat=0.3)),
    "mask_black": ((0.012, 0.012, 0.014), 0.0, 0.38, dict(coat=0.2)),
    "mask_green_light": ((0.05, 0.38, 0.12), 0.0, 0.35, dict(coat=0.3)),
    "copper": ((0.95, 0.46, 0.32), 1.0, 0.28, {}),
    "copper_dull": ((0.80, 0.42, 0.30), 1.0, 0.45, {}),
    "gold_enig": ((1.0, 0.70, 0.22), 1.0, 0.22, {}),
    "gold_lid": ((0.86, 0.66, 0.34), 1.0, 0.28, {}),
    "nickel": ((0.62, 0.62, 0.64), 1.0, 0.30, {}),
    "steel": ((0.72, 0.73, 0.75), 1.0, 0.35, {}),
    "steel_dark": ((0.30, 0.31, 0.33), 1.0, 0.45, {}),
    "aluminum": ((0.78, 0.79, 0.82), 1.0, 0.38, {}),
    "anodized_black": ((0.03, 0.03, 0.035), 1.0, 0.42, {}),
    "solder": ((0.72, 0.73, 0.76), 1.0, 0.26, {}),
    "tin": ((0.68, 0.68, 0.70), 1.0, 0.40, {}),
    "mold_black": ((0.014, 0.014, 0.016), 0.0, 0.50, {}),
    "die_gold": ((0.62, 0.50, 0.30), 1.0, 0.30, dict(emit=(1.0, 0.35, 0.08), emit_strength=0.0)),
    "hbm_tan": ((0.66, 0.54, 0.34), 1.0, 0.36, {}),
    "silicon": ((0.11, 0.12, 0.15), 0.9, 0.28, {}),
    "silicon_light": ((0.30, 0.31, 0.36), 0.9, 0.30, {}),
    "interposer": ((0.04, 0.26, 0.24), 0.5, 0.34, {}),
    "substrate": ((0.020, 0.055, 0.032), 0.0, 0.42, dict(coat=0.15)),
    "substrate_land": ((0.05, 0.075, 0.04), 0.0, 0.45, {}),
    "underfill": ((0.26, 0.16, 0.04), 0.0, 0.30, dict(coat=0.4)),
    "mlcc_body": ((0.42, 0.30, 0.17), 0.0, 0.55, {}),
    "mlcc_gray": ((0.18, 0.17, 0.16), 0.0, 0.55, {}),
    "res_black": ((0.008, 0.008, 0.009), 0.0, 0.45, {}),
    "alumina": ((0.80, 0.78, 0.72), 0.0, 0.55, {}),
    "ferrite": ((0.045, 0.045, 0.05), 0.0, 0.55, {}),
    "inductor_body": ((0.07, 0.07, 0.075), 0.2, 0.5, {}),
    "inductor_orange": ((0.8, 0.28, 0.03), 0.0, 0.4, {}),
    "tantalum": ((0.86, 0.40, 0.04), 0.0, 0.40, dict(coat=0.2)),
    "polymer_cap": ((0.025, 0.025, 0.03), 0.0, 0.42, {}),
    "electrolytic_can": ((0.04, 0.05, 0.12), 0.3, 0.35, {}),
    "plastic_black": ((0.02, 0.02, 0.022), 0.0, 0.45, {}),
    "plastic_white": ((0.85, 0.85, 0.82), 0.0, 0.5, {}),
    "plastic_gray": ((0.30, 0.30, 0.32), 0.0, 0.5, {}),
    "plastic_red": ((0.55, 0.04, 0.03), 0.0, 0.4, {}),
    "ferrite_core": ((0.10, 0.10, 0.11), 0.1, 0.6, {}),
    "enamel_wire": ((0.72, 0.36, 0.14), 1.0, 0.30, {}),
    "silkscreen": ((0.90, 0.90, 0.84), 0.0, 0.60, {}),
    "fr4": ((0.55, 0.50, 0.25), 0.0, 0.55, {}),
    "prepreg": ((0.75, 0.68, 0.40), 0.0, 0.5, {}),
    "core_dielectric": ((0.50, 0.55, 0.28), 0.0, 0.5, {}),
    "buildup_film": ((0.80, 0.74, 0.50), 0.0, 0.45, {}),
    "bond_layer": ((0.30, 0.19, 0.05), 0.0, 0.35, {}),
    "dram_si": ((0.16, 0.18, 0.24), 0.8, 0.30, {}),
    "base_die": ((0.24, 0.26, 0.33), 0.8, 0.30, {}),
    "laser_gold": ((0.95, 0.68, 0.22), 1.0, 0.35, {}),
    "thermal_pad": ((0.35, 0.40, 0.45), 0.0, 0.7, {}),
    "kapton": ((0.75, 0.42, 0.08), 0.0, 0.35, {}),
    "led_green": ((0.1, 0.8, 0.2), 0.0, 0.2, dict(emit=(0.1, 1.0, 0.2), emit_strength=1.0)),
    "led_red": ((0.8, 0.1, 0.05), 0.0, 0.2, dict(emit=(1.0, 0.1, 0.05), emit_strength=1.0)),
    "glow": ((1.0, 0.45, 0.10), 0.0, 0.5, dict(emit=(1.0, 0.45, 0.10), emit_strength=0.0)),
}


def mat(name):
    """Cached MAT_packaging_<name> (one per .blend session; call reset_mats() after C.reset())."""
    if name in _MATS:
        return _MATS[name]
    base, metal, rough, kw = _SPEC[name]
    m = C.principled("MAT_packaging_" + name, base, metallic=metal, rough=rough, **kw)
    if name == "mold_black":
        _noise_bump(m, scale=900.0, strength=0.15)
    elif name in ("mask_green", "mask_blue", "mask_black", "mask_green_light", "substrate"):
        _noise_bump(m, scale=1500.0, strength=0.04, detail=3.0)
    elif name in ("silicon", "die_gold", "hbm_tan", "interposer"):
        _noise_bump(m, scale=3000.0, strength=0.03, detail=2.0)
    elif name in ("mlcc_body", "ferrite", "inductor_body", "plastic_black"):
        _noise_bump(m, scale=1200.0, strength=0.10)
    _MATS[name] = m
    return m


def reset_mats():
    _MATS.clear()


def custom_mat(name, base, metallic=0.0, rough=0.5, **kw):
    key = "custom:" + name
    if key not in _MATS:
        _MATS[key] = C.principled("MAT_packaging_" + name, base, metallic=metallic, rough=rough, **kw)
    return _MATS[key]


# ------------------------------------------------------------------------------------------------ mesh builder
class MB:
    """Accumulates geometry in one bmesh; one object per MB. Dimensions in mm. Material index per primitive."""

    def __init__(self):
        self.bm = bmesh.new()

    # -- primitives
    def _setfaces(self, faces, mi, smooth=False):
        for f in faces:
            f.material_index = mi
            f.smooth = smooth

    def box(self, c, s, mi=0, bevel=0.0, seg=1, rz=0.0):
        r = bmesh.ops.create_cube(self.bm, size=1.0)
        vs = r["verts"]
        rot = Matrix.Rotation(rz, 3, "Z")
        for v in vs:
            p = Vector((v.co.x * s[0], v.co.y * s[1], v.co.z * s[2]))
            v.co = (rot @ p) + Vector(c)
            v.co *= MM
        faces = list({f for v in vs for f in v.link_faces})
        tup = isinstance(mi, (tuple, list))
        self._setfaces(faces, mi[1] if tup else mi)
        if bevel > 0:
            edges = list({e for v in vs for e in v.link_edges})
            res = bmesh.ops.bevel(self.bm, geom=edges, offset=bevel * MM, segments=seg, affect="EDGES", profile=0.5)
            self._setfaces(res["faces"], mi[1] if tup else mi, smooth=seg > 1)
            for f in faces:
                if f.is_valid:
                    f.smooth = False
        if tup:
            allf = [f for f in {f for v in vs if v.is_valid for f in v.link_faces}]
            for f in allf:
                nz = f.normal.z
                if nz > 0.9:
                    f.material_index = mi[0]
                elif nz < -0.9:
                    f.material_index = mi[2]
        return vs

    def loft_rects(self, a, b, mi=0):
        """Quad strip between two concentric axis-aligned rectangles a=(w,d,z), b=(w,d,z) (mm), outward facing normals."""
        def corners(r):
            w, d, z = r
            return [self.bm.verts.new(Vector((x * MM, y * MM, z * MM))) for x, y in
                    ((-w / 2, -d / 2), (w / 2, -d / 2), (w / 2, d / 2), (-w / 2, d / 2))]
        A, B = corners(a), corners(b)
        fs = []
        for k in range(4):
            k2 = (k + 1) % 4
            fs.append(self.bm.faces.new((A[k], A[k2], B[k2], B[k])))
        self._setfaces(fs, mi)
        return fs

    def cyl(self, c, r, h, axis="Z", seg=24, mi=0, r2=None, caps=True, smooth=True, rot=0.0):
        r2 = r if r2 is None else r2
        res = bmesh.ops.create_cone(self.bm, cap_ends=caps, cap_tris=False, segments=seg, radius1=r * MM, radius2=r2 * MM,
                                    depth=h * MM)
        vs = res["verts"]
        R = Matrix.Identity(3)
        if axis == "X":
            R = Matrix.Rotation(PI / 2, 3, "Y")
        elif axis == "Y":
            R = Matrix.Rotation(PI / 2, 3, "X")
        R = Matrix.Rotation(rot, 3, "Z") @ R
        for v in vs:
            v.co = R @ v.co + mmv(c)
        faces = list({f for v in vs for f in v.link_faces})
        self._setfaces(faces, mi, smooth=False)
        if smooth:
            for f in faces:
                if len(f.verts) == 4:
                    f.smooth = True
        return vs

    def lathe(self, prof, c=(0, 0, 0), seg=24, mi=0, smooth=True, axis="Z", close=False):
        """Body of revolution about Z. prof = [(r_mm, z_mm), ...] from bottom to top. r=0 points close on the axis."""
        rings = []
        for (r, z) in prof:
            if r <= 1e-9:
                rings.append([self.bm.verts.new(Vector((0, 0, z * MM)))])
            else:
                rings.append([self.bm.verts.new(Vector((r * MM * math.cos(2 * PI * k / seg),
                                                        r * MM * math.sin(2 * PI * k / seg), z * MM))) for k in range(seg)])
        faces = []
        for a, b in zip(rings[:-1], rings[1:]):
            if len(a) == 1 and len(b) == 1:
                continue
            if len(a) == 1:
                for k in range(seg):
                    faces.append(self.bm.faces.new((a[0], b[(k + 1) % seg], b[k])))
            elif len(b) == 1:
                for k in range(seg):
                    faces.append(self.bm.faces.new((a[k], a[(k + 1) % seg], b[0])))
            else:
                for k in range(seg):
                    faces.append(self.bm.faces.new((a[k], a[(k + 1) % seg], b[(k + 1) % seg], b[k])))
        if close:
            a, b = rings[-1], rings[0]
            for k in range(seg):
                faces.append(self.bm.faces.new((a[k], a[(k + 1) % seg], b[(k + 1) % seg], b[k])))
        # caps if the end rings are not on the axis
        for ring, flip in (() if close else ((rings[0], True), (rings[-1], False))):
            if len(ring) > 1:
                cv = self.bm.verts.new(Vector((0, 0, ring[0].co.z)))
                for k in range(seg):
                    t = (ring[(k + 1) % seg], ring[k], cv) if flip else (ring[k], ring[(k + 1) % seg], cv)
                    faces.append(self.bm.faces.new(t))
        allv = [v for ring in rings for v in ring]
        R = Matrix.Identity(3)
        if axis == "X":
            R = Matrix.Rotation(PI / 2, 3, "Y")
        elif axis == "Y":
            R = Matrix.Rotation(-PI / 2, 3, "X")
        for v in allv:
            v.co = R @ v.co + mmv(c)
        self._setfaces(faces, mi, smooth)
        # flat caps stay flat: unsmooth faces whose normal is axial and at ends
        return allv

    def rrect(self, c, w, d, h, rad, seg=4, mi=0, bevel=0.0):
        """Rounded-rectangle prism centered at c (z from c.z-h/2 to c.z+h/2)."""
        rad = min(rad, w / 2 - 1e-6, d / 2 - 1e-6)
        pts = []
        for (cx, cy, a0) in ((w / 2 - rad, d / 2 - rad, 0), (-w / 2 + rad, d / 2 - rad, 90),
                             (-w / 2 + rad, -d / 2 + rad, 180), (w / 2 - rad, -d / 2 + rad, 270)):
            for k in range(seg + 1):
                a = math.radians(a0 + 90.0 * k / seg)
                pts.append((c[0] + cx + rad * math.cos(a), c[1] + cy + rad * math.sin(a)))
        return self.prism(pts, c[2] - h / 2, c[2] + h / 2, mi=mi, smooth_sides=True)

    def prism(self, pts, z0, z1, mi=0, smooth_sides=False):
        """Extrude a convex 2D polygon (mm); caps are triangle fans from the centroid."""
        n = len(pts)
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n
        bot = [self.bm.verts.new(Vector((p[0] * MM, p[1] * MM, z0 * MM))) for p in pts]
        top = [self.bm.verts.new(Vector((p[0] * MM, p[1] * MM, z1 * MM))) for p in pts]
        cb = self.bm.verts.new(Vector((cx * MM, cy * MM, z0 * MM)))
        ct = self.bm.verts.new(Vector((cx * MM, cy * MM, z1 * MM)))
        faces = []
        for k in range(n):
            k2 = (k + 1) % n
            f = self.bm.faces.new((bot[k], bot[k2], top[k2], top[k]))
            f.smooth = smooth_sides
            faces.append(f)
            faces.append(self.bm.faces.new((bot[k2], bot[k], cb)))
            faces.append(self.bm.faces.new((top[k], top[k2], ct)))
        if isinstance(mi, (tuple, list)):
            for k, f in enumerate(faces):
                kind = k % 3
                f.material_index = mi[1] if kind == 0 else (mi[2] if kind == 1 else mi[0])
        else:
            for f in faces:
                f.material_index = mi
        return bot + top

    def ring_rect(self, wo, do, wi, di, z0, z1, rad_o=1.0, chamfer=0.0, seg=4, mi=(0, 0, 0), c=(0, 0)):
        """Rectangular ring (frame): outer rounded rectangle wo x do, rectangular window wi x di, z0..z1 (mm).
        mi = (top, side, bottom) material indices. A top outer chamfer of `chamfer` mm is optional."""
        def loop(w, d, r, off):
            pts = []
            r = max(r, 1e-3)
            for (cx, cy, a0) in ((w / 2 - r, d / 2 - r, 0), (-w / 2 + r, d / 2 - r, 90), (-w / 2 + r, -d / 2 + r, 180),
                                 (w / 2 - r, -d / 2 + r, 270)):
                for k in range(seg + 1):
                    a = math.radians(a0 + 90.0 * k / seg)
                    pts.append((c[0] + cx + r * math.cos(a), c[1] + cy + r * math.sin(a)))
            return pts
        O = loop(wo, do, rad_o, 0)
        Oc = loop(wo - 2 * chamfer, do - 2 * chamfer, max(rad_o - chamfer, 1e-3), 0) if chamfer > 0 else O
        I = loop(wi, di, 1e-3, 0)
        n = len(O)
        V = self.bm.verts.new
        Ob = [V(Vector((p[0] * MM, p[1] * MM, z0 * MM))) for p in O]
        Om = [V(Vector((p[0] * MM, p[1] * MM, (z1 - chamfer) * MM))) for p in O] if chamfer > 0 else Ob
        Ot = [V(Vector((p[0] * MM, p[1] * MM, z1 * MM))) for p in Oc]
        Ib = [V(Vector((p[0] * MM, p[1] * MM, z0 * MM))) for p in I]
        It = [V(Vector((p[0] * MM, p[1] * MM, z1 * MM))) for p in I]
        F = self.bm.faces.new
        fs = []
        for k in range(n):
            j = (k + 1) % n
            if chamfer > 0:
                fs.append((F((Ob[k], Ob[j], Om[j], Om[k])), mi[1], True))
                fs.append((F((Om[k], Om[j], Ot[j], Ot[k])), mi[1], False))
            else:
                fs.append((F((Ob[k], Ob[j], Ot[j], Ot[k])), mi[1], True))
            fs.append((F((Ot[k], Ot[j], It[j], It[k])), mi[0], False))
            fs.append((F((Ib[j], Ib[k], It[k], It[j])), mi[1], False))
            fs.append((F((Ob[j], Ob[k], Ib[k], Ib[j])), mi[2], False))
        for f, m, sm in fs:
            f.material_index = m
            f.smooth = sm
        return fs

    def tube(self, pts3, radius, seg=6, mi=0, closed=False):
        """Round wire swept along a 3D polyline (mm)."""
        n = len(pts3)
        P = [Vector(p) for p in pts3]
        rings = []
        prev_n = None
        for i in range(n):
            if closed:
                t = (P[(i + 1) % n] - P[(i - 1) % n]).normalized()
            else:
                t = (P[min(i + 1, n - 1)] - P[max(i - 1, 0)]).normalized()
            ref = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
            u = t.cross(ref).normalized()
            v = t.cross(u).normalized()
            rings.append([self.bm.verts.new((P[i] + (u * math.cos(2 * PI * k / seg) + v * math.sin(2 * PI * k / seg)) * radius) * MM)
                          for k in range(seg)])
        rng = range(n) if closed else range(n - 1)
        for i in rng:
            a, b = rings[i], rings[(i + 1) % n]
            for k in range(seg):
                f = self.bm.faces.new((a[k], a[(k + 1) % seg], b[(k + 1) % seg], b[k]))
                f.material_index = mi
                f.smooth = True

    def rotate_z(self, ang):
        bmesh.ops.rotate(self.bm, cent=(0, 0, 0), matrix=Matrix.Rotation(ang, 3, "Z"), verts=list(self.bm.verts))

    def weld(self, dist=1e-7):
        bmesh.ops.remove_doubles(self.bm, verts=list(self.bm.verts), dist=dist)

    def mirror_z(self):
        bmesh.ops.scale(self.bm, vec=(1, 1, -1), verts=list(self.bm.verts))
        bmesh.ops.reverse_faces(self.bm, faces=list(self.bm.faces))

    def cut_y(self, y_mm=0.0):
        """Keep y >= y_mm (cut the mesh with a plane and close the section with a face)."""
        geom = list(self.bm.verts) + list(self.bm.edges) + list(self.bm.faces)
        res = bmesh.ops.bisect_plane(self.bm, geom=geom, plane_co=(0, y_mm * MM, 0), plane_no=(0, 1, 0), clear_inner=True,
                                     use_snap_center=False)
        cut_edges = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
        if cut_edges:
            try:
                bmesh.ops.holes_fill(self.bm, edges=[e for e in cut_edges if e.is_valid], sides=0)
            except Exception:
                pass

    def extrude_xz(self, pts, y0, y1, mi=0):
        """Convex polygon in the XZ plane (mm) extruded from y0 to y1 (mm)."""
        n = len(pts)
        cx = sum(p[0] for p in pts) / n
        cz = sum(p[1] for p in pts) / n
        a = [self.bm.verts.new(Vector((p[0] * MM, y0 * MM, p[1] * MM))) for p in pts]
        b = [self.bm.verts.new(Vector((p[0] * MM, y1 * MM, p[1] * MM))) for p in pts]
        ca = self.bm.verts.new(Vector((cx * MM, y0 * MM, cz * MM)))
        cb = self.bm.verts.new(Vector((cx * MM, y1 * MM, cz * MM)))
        for k in range(n):
            j = (k + 1) % n
            for f in (self.bm.faces.new((a[k], b[k], b[j], a[j])), self.bm.faces.new((a[k], a[j], ca)),
                      self.bm.faces.new((b[j], b[k], cb))):
                f.material_index = mi

    def ribbon(self, pts, width, z0, z1, mi=0, closed=False):
        """Mitered ribbon (trace/stroke) along a 2D polyline (mm), extruded from z0 to z1 (mm)."""
        n = len(pts)
        P = [Vector((p[0], p[1])) for p in pts]
        left, right = [], []
        for i in range(n):
            if closed:
                a, b = P[(i - 1) % n], P[(i + 1) % n]
            else:
                a = P[i - 1] if i > 0 else None
                b = P[i + 1] if i < n - 1 else None
            d1 = (P[i] - a).normalized() if a is not None else None
            d2 = (b - P[i]).normalized() if b is not None else None
            if d1 is None:
                d1 = d2
            if d2 is None:
                d2 = d1
            n1 = Vector((-d1.y, d1.x))
            n2 = Vector((-d2.y, d2.x))
            m = n1 + n2
            if m.length < 1e-9:
                m = n1
            m.normalize()
            k = width / 2 / max(m.dot(n1), 0.3)
            left.append(P[i] + m * k)
            right.append(P[i] - m * k)
        faces = []
        L0 = [self.bm.verts.new(Vector((p.x * MM, p.y * MM, z0 * MM))) for p in left]
        R0 = [self.bm.verts.new(Vector((p.x * MM, p.y * MM, z0 * MM))) for p in right]
        L1 = [self.bm.verts.new(Vector((p.x * MM, p.y * MM, z1 * MM))) for p in left]
        R1 = [self.bm.verts.new(Vector((p.x * MM, p.y * MM, z1 * MM))) for p in right]
        rng = range(n) if closed else range(n - 1)
        for i in rng:
            j = (i + 1) % n
            faces.append(self.bm.faces.new((L1[i], R1[i], R1[j], L1[j])))        # top
            faces.append(self.bm.faces.new((R0[i], L0[i], L0[j], R0[j])))        # bottom
            faces.append(self.bm.faces.new((L0[i], L1[i], L1[j], L0[j])))        # left side
            faces.append(self.bm.faces.new((R1[i], R0[i], R0[j], R1[j])))        # right side
        if not closed:
            faces.append(self.bm.faces.new((L0[0], R0[0], R1[0], L1[0])))
            faces.append(self.bm.faces.new((R0[-1], L0[-1], L1[-1], R1[-1])))
        for f in faces:
            f.material_index = mi
        return faces

    def grid_plane(self, c, w, d, mi=0, uv=True):
        """Single quad (facing +Z) with UV 0..1."""
        x0, x1, y0, y1 = c[0] - w / 2, c[0] + w / 2, c[1] - d / 2, c[1] + d / 2
        vs = [self.bm.verts.new(Vector((x * MM, y * MM, c[2] * MM))) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        f = self.bm.faces.new(vs)
        f.material_index = mi
        if uv:
            lay = self.bm.loops.layers.uv.verify()
            for l, (u, v) in zip(f.loops, ((0, 0), (1, 0), (1, 1), (0, 1))):
                l[lay].uv = (u, v)
        return f

    # -- output
    def to_obj(self, name, mats, coll=None, parent=None, loc=(0, 0, 0)):
        me = bpy.data.meshes.new(name)
        bmesh.ops.recalc_face_normals(self.bm, faces=self.bm.faces)
        self.bm.to_mesh(me)
        self.bm.free()
        for m in mats:
            me.materials.append(m)
        o = bpy.data.objects.new(name, me)
        o.location = mmv(loc)
        if coll is not None:
            coll.objects.link(o)
        else:
            bpy.context.scene.collection.objects.link(o)
        if parent is not None:
            o.parent = parent
        return o


def obj_box(name, size, loc, material, coll, parent, bevel=0.0, seg=2, rz=0.0):
    """Quick single box object (mm)."""
    b = MB()
    b.box((0, 0, 0), size, bevel=bevel, seg=seg, rz=rz)
    return b.to_obj(name, [material], coll, parent, loc)


# ------------------------------------------------------------------------------------------------ instancing (GN)
def _scatter_group():
    if "NG_pk_scatter" in bpy.data.node_groups:
        return bpy.data.node_groups["NG_pk_scatter"]
    ng = bpy.data.node_groups.new("NG_pk_scatter", "GeometryNodeTree")
    it = ng.interface
    it.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    it.new_socket("Instance", in_out="INPUT", socket_type="NodeSocketObject")
    it.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n = ng.nodes
    gi = n.new("NodeGroupInput")
    go = n.new("NodeGroupOutput")
    iop = n.new("GeometryNodeInstanceOnPoints")
    oi = n.new("GeometryNodeObjectInfo")
    oi.transform_space = "RELATIVE"
    na_r = n.new("GeometryNodeInputNamedAttribute")
    na_r.data_type = "FLOAT"
    na_r.inputs["Name"].default_value = "rz"
    na_s = n.new("GeometryNodeInputNamedAttribute")
    na_s.data_type = "FLOAT"
    na_s.inputs["Name"].default_value = "sc"
    cx = n.new("ShaderNodeCombineXYZ")
    cs = n.new("ShaderNodeCombineXYZ")
    e2r = n.new("FunctionNodeEulerToRotation")
    L = ng.links
    L.new(gi.outputs["Geometry"], iop.inputs["Points"])
    L.new(gi.outputs["Instance"], oi.inputs["Object"])
    L.new(oi.outputs["Geometry"], iop.inputs["Instance"])
    L.new(na_r.outputs["Attribute"], cx.inputs["Z"])
    L.new(cx.outputs["Vector"], e2r.inputs["Euler"])
    L.new(e2r.outputs["Rotation"], iop.inputs["Rotation"])
    for k in "XYZ":
        L.new(na_s.outputs["Attribute"], cs.inputs[k])
    L.new(cs.outputs["Vector"], iop.inputs["Scale"])
    L.new(iop.outputs["Instances"], go.inputs["Geometry"])
    return ng


def z_exaggeration(name, coll, parent, root, prop="p_z_exaggeration"):
    """Empty (child of root) whose Z scale is driven by root[prop] (default 1.0). Parent cross-section geometry to it
    so the assembler can exaggerate layer thickness without touching mesh transforms."""
    e = bpy.data.objects.new(name, None)
    e.empty_display_type = "PLAIN_AXES"
    e.empty_display_size = 0.002
    coll.objects.link(e)
    e.parent = parent
    if prop not in root.keys():
        root[prop] = 1.0
    fc = e.driver_add("scale", 2)
    drv = fc.driver
    drv.type = "SCRIPTED"
    drv.expression = "v"
    var = drv.variables.new()
    var.name = "v"
    var.type = "SINGLE_PROP"
    var.targets[0].id = root
    var.targets[0].data_path = '["%s"]' % prop
    return e


def emission_driver(material, root, prop, strength=3.0):
    """Drive the Principled emission strength of `material` with root[prop] * strength (prop 0..1)."""
    nt = material.node_tree
    b = nt.nodes["Principled BSDF"]
    b.inputs["Emission Strength"].default_value = 0.0
    fc = nt.driver_add('nodes["Principled BSDF"].inputs["Emission Strength"].default_value')
    drv = fc.driver
    drv.type = "SCRIPTED"
    drv.expression = "v*%g" % strength
    var = drv.variables.new()
    var.name = "v"
    var.type = "SINGLE_PROP"
    var.targets[0].id = root
    var.targets[0].data_path = '["%s"]' % prop


def _scatter_group_density():
    if "NG_pk_scatter_density" in bpy.data.node_groups:
        return bpy.data.node_groups["NG_pk_scatter_density"]
    ng = bpy.data.node_groups.new("NG_pk_scatter_density", "GeometryNodeTree")
    it = ng.interface
    it.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    it.new_socket("Instance", in_out="INPUT", socket_type="NodeSocketObject")
    it.new_socket("Density", in_out="INPUT", socket_type="NodeSocketFloat")
    it.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    n = ng.nodes
    gi = n.new("NodeGroupInput")
    go = n.new("NodeGroupOutput")
    iop = n.new("GeometryNodeInstanceOnPoints")
    oi = n.new("GeometryNodeObjectInfo")
    oi.transform_space = "RELATIVE"
    na = {}
    for nm in ("rz", "sc", "rnd"):
        x = n.new("GeometryNodeInputNamedAttribute")
        x.data_type = "FLOAT"
        x.inputs["Name"].default_value = nm
        na[nm] = x
    cx = n.new("ShaderNodeCombineXYZ")
    cs = n.new("ShaderNodeCombineXYZ")
    e2r = n.new("FunctionNodeEulerToRotation")
    cmp_ = n.new("FunctionNodeCompare")
    cmp_.data_type = "FLOAT"
    cmp_.operation = "GREATER_EQUAL"
    dg = n.new("GeometryNodeDeleteGeometry")
    dg.domain = "POINT"
    L = ng.links
    L.new(gi.outputs["Geometry"], dg.inputs["Geometry"])
    L.new(na["rnd"].outputs["Attribute"], cmp_.inputs[0])
    L.new(gi.outputs["Density"], cmp_.inputs[1])
    L.new(cmp_.outputs["Result"], dg.inputs["Selection"])
    L.new(dg.outputs["Geometry"], iop.inputs["Points"])
    L.new(gi.outputs["Instance"], oi.inputs["Object"])
    L.new(oi.outputs["Geometry"], iop.inputs["Instance"])
    L.new(na["rz"].outputs["Attribute"], cx.inputs["Z"])
    L.new(cx.outputs["Vector"], e2r.inputs["Euler"])
    L.new(e2r.outputs["Rotation"], iop.inputs["Rotation"])
    for k in "XYZ":
        L.new(na["sc"].outputs["Attribute"], cs.inputs[k])
    L.new(cs.outputs["Vector"], iop.inputs["Scale"])
    L.new(iop.outputs["Instances"], go.inputs["Geometry"])
    return ng


def scatter_density(name, pts_mm, src, coll, parent, root, rz=None, rnd=None, prop="p_density", seed=1):
    """Like scatter() but each point has a random value 'rnd' in [0,1); points with rnd >= root[prop] are dropped (Delete Geometry),
    so the custom property p_density (0..1) thins the population uniformly (driver on the modifier input)."""
    import random as _r
    ng = _scatter_group_density()
    n = len(pts_mm)
    me = bpy.data.meshes.new(name)
    me.from_pydata([(p[0] * MM, p[1] * MM, p[2] * MM) for p in pts_mm], [], [])
    for a in ("rz", "sc", "rnd"):
        me.attributes.new(a, "FLOAT", "POINT")
    rr = _r.Random(seed)
    me.attributes["rz"].data.foreach_set("value", rz if rz is not None else [0.0] * n)
    me.attributes["sc"].data.foreach_set("value", [1.0] * n)
    me.attributes["rnd"].data.foreach_set("value", rnd if rnd is not None else [rr.random() for _ in range(n)])
    o = bpy.data.objects.new(name, me)
    coll.objects.link(o)
    o.parent = parent
    md = o.modifiers.new("scatter", "NODES")
    md.node_group = ng
    ids = {i.name: i.identifier for i in ng.interface.items_tree}
    md[ids["Instance"]] = src
    md[ids["Density"]] = 1.0
    if prop not in root.keys():
        root[prop] = 1.0
    fc = o.driver_add('modifiers["scatter"]["%s"]' % ids["Density"])
    drv = fc.driver
    drv.type = "SCRIPTED"
    drv.expression = "v"
    var = drv.variables.new()
    var.name = "v"
    var.type = "SINGLE_PROP"
    var.targets[0].id = root
    var.targets[0].data_path = '["%s"]' % prop
    o["pk_instance_count"] = n
    return o


def refresh_drivers(root):
    """Force driver re-evaluation after changing a custom property from Python (headless)."""
    root.update_tag()
    for o in bpy.data.objects:
        if o.animation_data and o.animation_data.drivers:
            o.update_tag()
    bpy.context.view_layer.update()


def source(obj):
    """Mark an object as an instancing source (hidden in viewport and render, stays in the asset collection)."""
    obj.hide_render = True
    obj.hide_viewport = True
    obj["pk_instance_source"] = True
    return obj


def scatter(name, pts_mm, src, coll, parent, rz=None, sc=None, loc=(0, 0, 0)):
    """Instance `src` at every point (mm, relative to `loc`) with optional per-point z-rotation (rad) and scale.

    The source object must be centered on its own instance origin (it is positioned at the point). Returns the point object.
    """
    ng = _scatter_group()
    me = bpy.data.meshes.new(name)
    me.from_pydata([(p[0] * MM, p[1] * MM, p[2] * MM) for p in pts_mm], [], [])
    me.attributes.new("rz", "FLOAT", "POINT")
    me.attributes.new("sc", "FLOAT", "POINT")
    n = len(pts_mm)
    me.attributes["rz"].data.foreach_set("value", rz if rz is not None else [0.0] * n)
    me.attributes["sc"].data.foreach_set("value", sc if sc is not None else [1.0] * n)
    o = bpy.data.objects.new(name, me)
    o.location = mmv(loc)
    coll.objects.link(o)
    o.parent = parent
    md = o.modifiers.new("scatter", "NODES")
    md.node_group = ng
    ids = {i.name: i.identifier for i in ng.interface.items_tree}
    md[ids["Instance"]] = src
    o["pk_instance_count"] = n
    o["pk_instance_source_name"] = src.name
    return o


# ------------------------------------------------------------------------------------------------ text
def text(name, body, size_mm, loc_mm, mat_, coll, parent, rot=(0, 0, 0), extrude_mm=0.01, align="CENTER", bold=False):
    """Text as a mesh object (converted from a font curve)."""
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size_mm * MM
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.extrude = extrude_mm * MM / 2.0
    o = bpy.data.objects.new(name, cu)
    bpy.context.scene.collection.objects.link(o)
    o.location = mmv(loc_mm)
    o.rotation_euler = rot
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.select_all(action="DESELECT")
    o.select_set(True)
    bpy.ops.object.convert(target="MESH")
    o = bpy.context.active_object
    o.name = name
    o.data.name = name
    o.data.materials.clear()
    o.data.materials.append(mat_)
    for l in list(o.users_collection):
        l.objects.unlink(o)
    coll.objects.link(o)
    o.parent = parent
    return o


def valley_mark_mesh(width_mm, stroke_mm, z0, z1, mi=0):
    """Parody mark: a smooth valley (arch tip flipped upside down) over a wave line. Returns MB centered at 0 (x,y), unit width."""
    b = MB()
    w = width_mm
    # valley: half-sine dip, drawn as a thick stroke
    pts = [(-0.5 * w + w * k / 20.0, 0.18 * w - 0.34 * w * math.sin(PI * k / 20.0) * 0.9 + 0.0) for k in range(21)]
    pts = [(x, y + 0.17 * w) for x, y in pts]
    b.ribbon(pts, stroke_mm, z0, z1, mi)
    wave = [(-0.62 * w + 1.24 * w * k / 30.0, -0.30 * w + 0.075 * w * math.sin(2 * PI * k / 15.0)) for k in range(31)]
    b.ribbon(wave, stroke_mm * 0.8, z0, z1, mi)
    return b


# ------------------------------------------------------------------------------------------------ distribution helpers
def grid_points(nx, ny, pitch_x, pitch_y=None, center=(0, 0, 0), skip=None, stagger=False):
    """Centered rectangular grid (mm). skip(i, j, x, y) -> True to omit a site."""
    pitch_y = pitch_x if pitch_y is None else pitch_y
    pts = []
    for j in range(ny):
        for i in range(nx):
            x = (i - (nx - 1) / 2.0) * pitch_x + (pitch_x / 2.0 if stagger and j % 2 else 0)
            y = (j - (ny - 1) / 2.0) * pitch_y
            if skip is not None and skip(i, j, x, y):
                continue
            pts.append((center[0] + x, center[1] + y, center[2]))
    return pts


# ------------------------------------------------------------------------------------------------ stats / meta
def eval_stats(coll):
    """Bounding box (mm) and triangle count including Geometry Nodes instances for objects in a collection tree."""
    objs = set()

    def walk(c):
        for o in c.objects:
            objs.add(o)
        for ch in c.children:
            walk(ch)
    walk(coll)
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    tris = 0
    for inst in dg.object_instances:
        o = inst.object
        owner = inst.parent if inst.is_instance else o
        orig = owner.original if owner is not None else None
        if orig is None or orig not in objs:
            continue
        if o.type not in {"MESH"}:
            continue
        if not inst.is_instance and o.original.get("pk_instance_source"):
            continue
        try:
            me = o.data
            me.calc_loop_triangles()
            tris += len(me.loop_triangles)
        except Exception:
            pass
        mw = inst.matrix_world
        for v in o.bound_box:
            pts.append(mw @ Vector(v))
    if not pts:
        return None, 0
    lo = [min(p[k] for p in pts) * 1000 for k in range(3)]
    hi = [max(p[k] for p in pts) * 1000 for k in range(3)]
    return (lo, hi), tris


def collect_auto_meta(coll, root):
    hooks, mats = [], set()
    objs = []

    def walk(c):
        for o in c.objects:
            objs.append(o)
        for ch in c.children:
            walk(ch)
    walk(coll)
    for o in objs:
        if o.name.startswith("HOOK_"):
            hooks.append({"name": o.name, "parent": o.parent.name if o.parent else None,
                          "location_mm": [round(v * 1000, 4) for v in o.matrix_world.translation],
                          "rotation_euler_deg": [round(math.degrees(a), 3) for a in o.matrix_world.to_euler()]})
        if o.type in {"MESH", "CURVE"} and o.data is not None:
            for m in o.data.materials:
                if m:
                    mats.add(m.name)
    props = {k: root[k] for k in root.keys() if k.startswith("p_")}
    return hooks, sorted(mats), props


class Dims:
    def __init__(self):
        self.rows = []

    def add(self, item, value, unit, source, level):
        self.rows.append({"item": item, "value": value, "unit": unit, "source": source, "accuracy": level})
        return value


def write_json(blend_path, meta):
    path = os.path.splitext(blend_path)[0] + ".json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return path


# ------------------------------------------------------------------------------------------------ previews
def _look_rot(cam, target):
    d = Vector(target) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def auto_views(bb):
    """front / three_quarter / top views from a mm bbox (lo, hi)."""
    lo, hi = Vector(bb[0]), Vector(bb[1])
    c = (lo + hi) / 2
    s = hi - lo
    r = max(s.length / 2, 1.0)
    f = 2.6
    return [
        dict(name="front", loc=(c.x, c.y - r * f, c.z + r * 0.35 * f * 0.5), tgt=tuple(c), lens=50),
        dict(name="three_quarter", loc=(c.x + r * f * 0.62, c.y - r * f * 0.78, c.z + r * f * 0.62), tgt=tuple(c), lens=50),
        dict(name="top", loc=(c.x, c.y - 0.02 * r, c.z + r * f * 1.05), tgt=tuple(c), lens=50),
    ]


def render_views(coll, out_dir, asset_id, views, res=(900, 675), samples=16, floor_z_mm=None, bg=(0.50, 0.53, 0.58),
                 hide=()):
    """Render each view dict(name, loc, tgt, lens[, floor][, clip]) in mm. Hides everything outside `coll`."""
    os.makedirs(out_dir, exist_ok=True)
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = samples
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.color_mode = "RGB"
    scn.render.image_settings.compression = 100
    bpy.context.preferences.filepaths.save_version = 0
    scn.view_settings.view_transform = "Standard"
    keep = set()

    def walk(c):
        for o in c.objects:
            keep.add(o)
        for ch in c.children:
            walk(ch)
    walk(coll)
    saved = {o: o.hide_render for o in scn.objects}
    for o in list(scn.objects):
        if o not in keep:
            o.hide_render = True
    for o in hide:
        o.hide_render = True
    old_world = scn.world
    w = bpy.data.worlds.new("PREVIEW_world")
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (*bg, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.55
    scn.world = w
    bb, _ = eval_stats(coll)
    lo, hi = Vector(bb[0]) * MM, Vector(bb[1]) * MM
    ctr = (lo + hi) / 2
    rad = max((hi - lo).length / 2, 1e-3)
    sun = bpy.data.lights.new("PREVIEW_sun", "SUN")
    sun.energy = 1.3
    so = bpy.data.objects.new("PREVIEW_sun", sun)
    so.rotation_euler = (0.85, 0.25, 0.6)
    scn.collection.objects.link(so)
    fill = bpy.data.lights.new("PREVIEW_fill", "AREA")
    d = rad * 4.0
    fill.energy = 14 * d * d
    fill.size = rad * 3
    fo = bpy.data.objects.new("PREVIEW_fill", fill)
    fo.location = ctr + Vector((-rad * 2, -rad * 3, rad * 2.5))
    fo.rotation_euler = (1.0, 0, -0.5)
    scn.collection.objects.link(fo)
    fl = None
    fl_mat = C.principled("PREVIEW_floor_mat", (0.42, 0.44, 0.42), rough=0.85)
    cam_d = bpy.data.cameras.new("PREVIEW_cam")
    cam_d.sensor_width = 36
    cam = bpy.data.objects.new("PREVIEW_cam", cam_d)
    scn.collection.objects.link(cam)
    scn.camera = cam
    out = []
    for v in views:
        cam.location = mmv(v["loc"])
        _look_rot(cam, mmv(v["tgt"]))
        cam_d.lens = v.get("lens", 50)
        dist = (mmv(v["loc"]) - mmv(v["tgt"])).length
        cam_d.clip_start = max(dist * 0.01, 1e-5)
        cam_d.clip_end = max(dist * 200, 2.0)
        if "ortho" in v:
            cam_d.type = "ORTHO"
            cam_d.ortho_scale = v["ortho"] * MM
        else:
            cam_d.type = "PERSP"
        if fl is not None:
            bpy.data.objects.remove(fl)
            fl = None
        if v.get("floor", False):
            fz = (floor_z_mm if floor_z_mm is not None else bb[0][2]) - 0.05
            bpy.ops.mesh.primitive_plane_add(size=rad * 300, location=(ctr.x, ctr.y, fz * MM))
            fl = bpy.context.active_object
            fl.name = "PREVIEW_floor"
            fl.data.materials.append(fl_mat)
        path = os.path.join(out_dir, "%s_%s.png" % (asset_id, v["name"]))
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        out.append(path)
        if cam_d.type == "ORTHO":
            cam_d.type = "PERSP"
    # cleanup so that render_views can be called again (variants) without stacking lights
    if fl is not None:
        bpy.data.objects.remove(fl)
    for o in (so, fo, cam):
        bpy.data.objects.remove(o)
    scn.world = old_world
    bpy.data.worlds.remove(w)
    for o, h in saved.items():
        if o.name in bpy.data.objects:
            o.hide_render = h
    return out


def finish(asset_id, blend_path, coll, root, meta, dims, preview_dir, views=None, extra_views=(), res=(900, 675),
           floor=True):
    """Save blend + JSON metadata (with eval bbox, tris incl. instances, auto hooks/materials/props) + previews."""
    bb, tris = eval_stats(coll)
    hooks, mats, props = collect_auto_meta(coll, root)
    meta = dict(meta)
    meta.update({"asset_id": asset_id, "bbox_mm": bb, "size_mm": [round(bb[1][k] - bb[0][k], 3) for k in range(3)],
                 "triangles_including_instances": tris, "dimension_table": dims.rows, "hooks": hooks,
                 "material_slots": mats, "custom_properties": props, "blender_version": bpy.app.version_string,
                 "units": "1 Blender unit = 1 m; z up; -Y front; root origin documented under origin"})
    C.save(blend_path)
    vs = list(views) if views is not None else auto_views(bb)
    vs = [dict(v, floor=v.get("floor", floor and v["name"] in ("front", "three_quarter"))) for v in vs] + list(extra_views)
    pngs = render_views(coll, preview_dir, asset_id, vs, res=res)
    meta["previews"] = [os.path.relpath(p, os.path.dirname(blend_path)) for p in pngs]
    write_json(blend_path, meta)
    print("ASSET", asset_id, "tris", tris, "size_mm", meta["size_mm"], "previews", len(pngs))
    return meta
