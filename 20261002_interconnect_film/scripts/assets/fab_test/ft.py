"""fab_test helper layer on top of scripts/assets/_common/common.py.

All public geometry helpers take millimetres and ASSET-space coordinates (origin at the asset root,
Z up, -Y front). Parts parented under a hook (a moving frame, translation only at rest) are specified in
asset space too; the helper converts to the parent's local frame. Mesh objects get scale 1 / rotation 0,
origin at the bbox centre of the part. No bpy.ops are used for geometry (deterministic, fast).
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_common"))
import common as C  # noqa: E402
import bpy  # noqa: E402
import bmesh  # noqa: E402
from mathutils import Vector, Matrix, geometry  # noqa: E402

MM = 0.001
CAT = "fab_test"

# ------------------------------------------------------------------ material library
LIB = {
    "black_paint": dict(base=(0.014, 0.014, 0.016), metallic=0.0, rough=0.42, coat=0.35),
    "black_plastic": dict(base=(0.02, 0.02, 0.022), metallic=0.0, rough=0.5),
    "black_anodized": dict(base=(0.03, 0.03, 0.034), metallic=0.85, rough=0.33),
    "silver": dict(base=(0.80, 0.81, 0.84), metallic=1.0, rough=0.2),
    "steel": dict(base=(0.60, 0.62, 0.66), metallic=1.0, rough=0.36),
    "steel_dark": dict(base=(0.30, 0.31, 0.33), metallic=1.0, rough=0.42),
    "deck_silver": dict(base=(0.70, 0.72, 0.76), metallic=1.0, rough=0.28),
    "chrome": dict(base=(0.9, 0.9, 0.92), metallic=1.0, rough=0.07),
    "gold": dict(base=(1.0, 0.77, 0.33), metallic=1.0, rough=0.22),
    "copper": dict(base=(0.95, 0.52, 0.38), metallic=1.0, rough=0.3),
    "tungsten": dict(base=(0.45, 0.46, 0.50), metallic=1.0, rough=0.3),
    "rubber": dict(base=(0.015, 0.015, 0.016), metallic=0.0, rough=0.8),
    "white_plastic": dict(base=(0.82, 0.83, 0.84), metallic=0.0, rough=0.4),
    "grey_plastic": dict(base=(0.22, 0.23, 0.25), metallic=0.0, rough=0.5),
    "ceramic": dict(base=(0.85, 0.82, 0.72), metallic=0.0, rough=0.45),
    "silicon": dict(base=(0.50, 0.53, 0.60), metallic=1.0, rough=0.04),
    "pcb_green": dict(base=(0.01, 0.20, 0.07), metallic=0.0, rough=0.35, coat=0.2),
    "pcb_blue": dict(base=(0.02, 0.07, 0.30), metallic=0.0, rough=0.35, coat=0.2),
    "blue_anodized": dict(base=(0.02, 0.10, 0.55), metallic=0.8, rough=0.35),
    "yellow": dict(base=(0.95, 0.72, 0.02), metallic=0.0, rough=0.45),
    "red": dict(base=(0.75, 0.03, 0.03), metallic=0.0, rough=0.35),
    "orange": dict(base=(0.9, 0.35, 0.03), metallic=0.0, rough=0.45),
    "green_led": dict(base=(0.05, 0.9, 0.2), metallic=0.0, rough=0.3, emit=(0.1, 1.0, 0.25), emit_strength=3.0),
    "red_led": dict(base=(0.9, 0.05, 0.05), metallic=0.0, rough=0.3, emit=(1.0, 0.1, 0.05), emit_strength=3.0),
    "glass": dict(base=(0.9, 0.95, 1.0), metallic=0.0, rough=0.03, transmission=1.0, ior=1.45),
    "glass_tint": dict(base=(0.6, 0.8, 0.9), metallic=0.0, rough=0.05, transmission=0.9, ior=1.45),
    "tape_uv": dict(base=(0.12, 0.35, 0.85), metallic=0.0, rough=0.25, alpha=0.55),
    "oven_glow": dict(base=(1.0, 0.35, 0.08), metallic=0.0, rough=0.5, emit=(1.0, 0.32, 0.05), emit_strength=8.0),
    "label_white": dict(base=(0.9, 0.9, 0.88), metallic=0.0, rough=0.5),
    "fiber_glass": dict(base=(0.9, 0.9, 0.95), metallic=0.0, rough=0.2, transmission=0.6),
    "fiber_yellow": dict(base=(0.95, 0.75, 0.1), metallic=0.0, rough=0.4),
    "fiber_aqua": dict(base=(0.05, 0.75, 0.8), metallic=0.0, rough=0.4),
    "polyimide": dict(base=(0.75, 0.42, 0.08), metallic=0.0, rough=0.35),
    "mesh_belt": dict(base=(0.55, 0.57, 0.6), metallic=1.0, rough=0.5),
    "foup_poly": dict(base=(0.05, 0.05, 0.06), metallic=0.0, rough=0.35, alpha=0.8),
    "foup_poly_clear": dict(base=(0.30, 0.34, 0.40), metallic=0.0, rough=0.15, alpha=0.16),
    "cleanroom_floor": dict(base=(0.55, 0.57, 0.6), metallic=0.0, rough=0.6),
}


class Asset:
    """One asset collection plus builder helpers."""

    def __init__(self, asset_id, accuracy="B"):
        C.reset()
        self.id = asset_id
        self.coll, self.root = C.new_asset(asset_id, accuracy)
        self.cur = self.coll
        self.mats = {}
        self.hooks = {}
        self.props = {}
        self.dims = []
        self.sources = []
        self.parts = {}
        self.sub = {}
        self.default_parent = self.root

    # ---------------------------------------------------------- collections
    def variant(self, name):
        c = C.sub_collection(self.coll, "VARIANT_" + name)
        self.sub[name] = c
        return c

    # ---------------------------------------------------------- materials
    def mat(self, key, **over):
        full = "MAT_%s_%s" % (CAT, key)
        if full in bpy.data.materials:
            return bpy.data.materials[full]
        kw = dict(LIB[key])
        base = kw.pop("base")
        kw.update(over)
        m = C.principled(full, base, **kw)
        self.mats[full] = m
        return m

    def custom_mat(self, key, base, **kw):
        full = "MAT_%s_%s" % (CAT, key)
        if full in bpy.data.materials:
            return bpy.data.materials[full]
        m = C.principled(full, base, **kw)
        self.mats[full] = m
        return m

    def _m(self, m):
        if isinstance(m, str):
            return self.mat(m)
        return m

    # ---------------------------------------------------------- frames
    @staticmethod
    def world_off(parent):
        """Rest-pose world translation (m) of a parent chain (rotation ignored: parents are translation-only)."""
        v = Vector((0, 0, 0))
        p = parent
        while p is not None:
            v += p.location
            p = p.parent
        return v

    # ---------------------------------------------------------- raw mesh -> object
    def obj_from_mesh(self, name, verts_m, faces, mat=None, parent=None, coll=None, smooth=True, sharp_deg=32.0,
                      recenter=True):
        parent = parent or self.default_parent
        coll = coll or self.cur
        off = self.world_off(parent)
        bm = bmesh.new()
        vs = [bm.verts.new(Vector(v) - off) for v in verts_m]
        for f in faces:
            try:
                bm.faces.new([vs[i] for i in f])
            except ValueError:
                pass
        bm.normal_update()
        return self._finish_bm(name, bm, mat, parent, coll, smooth, sharp_deg, recenter)

    def _finish_bm(self, name, bm, mat, parent, coll, smooth, sharp_deg, recenter=True, fix_normals=True):
        if fix_normals:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        if smooth:
            lim = math.radians(sharp_deg)
            for f in bm.faces:
                f.smooth = True
            for e in bm.edges:
                if len(e.link_faces) == 2:
                    try:
                        ang = e.calc_face_angle()
                    except ValueError:
                        ang = 0
                    e.smooth = ang < lim
        c = Vector((0, 0, 0))
        if recenter and bm.verts:
            xs = [v.co.x for v in bm.verts]
            ys = [v.co.y for v in bm.verts]
            zs = [v.co.z for v in bm.verts]
            c = Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2))
            for v in bm.verts:
                v.co -= c
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(name, me)
        coll.objects.link(ob)
        ob.parent = parent
        ob.location = c
        if mat is not None:
            me.materials.append(self._m(mat))
        self.parts[name] = ob
        return ob

    # ---------------------------------------------------------- primitives (mm, asset space)
    def box(self, name, size, pos, mat="steel", bev=0.0, anchor="c", parent=None, coll=None, bev_seg=2, smooth=True,
            rot_z=0.0):
        sx, sy, sz = [s * MM for s in size]
        px, py, pz = [p * MM for p in pos]
        if anchor == "b":
            pz += sz / 2
        elif anchor == "t":
            pz -= sz / 2
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            v.co.x *= sx
            v.co.y *= sy
            v.co.z *= sz
        if bev > 0:
            b = min(bev * MM, min(sx, sy, sz) * 0.49)
            bmesh.ops.bevel(bm, geom=bm.edges[:], offset=b, segments=bev_seg, profile=0.5, affect="EDGES",
                            clamp_overlap=True)
        if rot_z:
            bmesh.ops.rotate(bm, verts=bm.verts[:], matrix=Matrix.Rotation(rot_z, 3, "Z"))
        for v in bm.verts:
            v.co += Vector((px, py, pz)) - self.world_off(parent or self.default_parent)
        return self._finish_bm(name, bm, mat, parent or self.default_parent, coll or self.cur, smooth, 35.0)

    def lathe(self, name, prof, pos, mat="steel", axis="z", seg=48, parent=None, coll=None, smooth=True,
              sharp_deg=35.0, angle=360.0, start_deg=0.0):
        """Revolve a closed (r_mm, z_mm) polygon (relative to pos) about the local Z axis, then orient axis."""
        px, py, pz = [p * MM for p in pos]
        full = abs(angle - 360.0) < 1e-6
        nseg = seg if full else seg + 1
        rings = []
        verts = []
        for (r, z) in prof:
            if abs(r) < 1e-9:
                verts.append((0.0, 0.0, z * MM))
                rings.append(("apex", len(verts) - 1))
            else:
                idx = []
                for j in range(nseg):
                    a = math.radians(start_deg + angle * j / seg)
                    verts.append((r * MM * math.cos(a), r * MM * math.sin(a), z * MM))
                    idx.append(len(verts) - 1)
                rings.append(("ring", idx))
        faces = []
        n = len(prof)
        jmax = seg if full else seg
        for i in range(n):
            a, b = rings[i], rings[(i + 1) % n]
            for j in range(jmax):
                j2 = (j + 1) % seg if full else j + 1
                if a[0] == "ring" and b[0] == "ring":
                    faces.append((a[1][j], a[1][j2], b[1][j2], b[1][j]))
                elif a[0] == "ring" and b[0] == "apex":
                    faces.append((a[1][j], a[1][j2], b[1]))
                elif a[0] == "apex" and b[0] == "ring":
                    faces.append((a[1], b[1][j2], b[1][j]))
        # axis orientation
        if axis == "x":
            M3 = Matrix.Rotation(math.pi / 2, 3, "Y")
        elif axis == "y":
            M3 = Matrix.Rotation(-math.pi / 2, 3, "X")
        else:
            M3 = Matrix.Identity(3)
        off = self.world_off(parent or self.default_parent)
        vv = [M3 @ Vector(v) + Vector((px, py, pz)) - off for v in verts]
        bm = bmesh.new()
        bv = [bm.verts.new(v) for v in vv]
        for f in faces:
            try:
                bm.faces.new([bv[i] for i in f])
            except ValueError:
                pass
        bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=1e-9)
        return self._finish_bm(name, bm, mat, parent or self.default_parent, coll or self.cur, smooth, sharp_deg,
                               fix_normals=full)

    def cyl(self, name, r, h, pos, mat="steel", axis="z", seg=48, bev=0.0, r_in=0.0, anchor="c", parent=None,
            coll=None, bev_seg=2, r_top=None):
        """Cylinder / tube / cone frustum. pos = centre (anchor c), bottom centre (b) or top centre (t) along its axis."""
        h2 = h / 2.0
        if anchor == "b":
            pos = self._shift(pos, axis, h2)
        elif anchor == "t":
            pos = self._shift(pos, axis, -h2)
        rt = r if r_top is None else r_top
        b = min(bev, (r - r_in) * 0.49, h * 0.49) if bev > 0 else 0.0
        prof = []
        if r_in <= 0:
            prof.append((0.0, -h2))
            if b > 0:
                prof += self._arc(r - b, -h2 + b, b, -90, 0, bev_seg)
                prof += self._arc(rt - b, h2 - b, b, 0, 90, bev_seg)
            else:
                prof += [(r, -h2), (rt, h2)]
            prof.append((0.0, h2))
        else:
            if b > 0:
                prof += self._arc(r_in + b, -h2 + b, b, -180, -90, bev_seg)
                prof += self._arc(r - b, -h2 + b, b, -90, 0, bev_seg)
                prof += self._arc(rt - b, h2 - b, b, 0, 90, bev_seg)
                prof += self._arc(r_in + b, h2 - b, b, 90, 180, bev_seg)
            else:
                prof += [(r_in, -h2), (r, -h2), (rt, h2), (r_in, h2)]
        return self.lathe(name, prof, pos, mat, axis, seg, parent, coll)

    @staticmethod
    def _arc(cr, cz, rad, a0, a1, n):
        out = []
        for k in range(n + 1):
            a = math.radians(a0 + (a1 - a0) * k / n)
            out.append((cr + rad * math.cos(a), cz + rad * math.sin(a)))
        return out

    @staticmethod
    def _shift(pos, axis, d):
        p = list(pos)
        p["xyz".index(axis)] += d
        return tuple(p)

    def prism(self, name, pts_xy, z0, z1, mat="steel", parent=None, coll=None, smooth=False, holes=None, bev=0.0):
        """Extrude a simple polygon (mm, asset XY) from z0 to z1 mm. Handles concave outlines (tessellated)."""
        off = self.world_off(parent or self.default_parent)
        n = len(pts_xy)
        top = [Vector((x * MM, y * MM, z1 * MM)) - off for x, y in pts_xy]
        bot = [Vector((x * MM, y * MM, z0 * MM)) - off for x, y in pts_xy]
        tess = geometry.tessellate_polygon([[Vector((x * MM, y * MM, 0)) for x, y in pts_xy]])
        bm = bmesh.new()
        tv = [bm.verts.new(v) for v in top]
        bv = [bm.verts.new(v) for v in bot]
        for t in tess:
            try:
                bm.faces.new([tv[t[0]], tv[t[1]], tv[t[2]]])
                bm.faces.new([bv[t[2]], bv[t[1]], bv[t[0]]])
            except ValueError:
                pass
        for i in range(n):
            j = (i + 1) % n
            try:
                bm.faces.new([tv[i], bv[i], bv[j], tv[j]])
            except ValueError:
                pass
        return self._finish_bm(name, bm, mat, parent or self.default_parent, coll or self.cur, smooth, 35.0)

    def tube_path(self, name, pts, r, mat="rubber", seg=10, parent=None, coll=None, caps=True, r_end=None):
        """Swept circular tube along a polyline (mm). r in mm (may be a list per point)."""
        off = self.world_off(parent or self.default_parent)
        P = [Vector((p[0] * MM, p[1] * MM, p[2] * MM)) for p in pts]
        n = len(P)
        radii = r if isinstance(r, (list, tuple)) else [r] * n
        bm = bmesh.new()
        rings = []
        prev_n = None
        for i in range(n):
            if i == 0:
                t = (P[1] - P[0]).normalized()
            elif i == n - 1:
                t = (P[-1] - P[-2]).normalized()
            else:
                t = ((P[i + 1] - P[i]).normalized() + (P[i] - P[i - 1]).normalized()).normalized()
            if prev_n is None:
                ref = Vector((0, 0, 1)) if abs(t.z) < 0.9 else Vector((1, 0, 0))
                nn = t.cross(ref).normalized()
            else:
                nn = (prev_n - t * prev_n.dot(t))
                nn = nn.normalized() if nn.length > 1e-9 else prev_n
            bb = t.cross(nn).normalized()
            prev_n = nn
            ring = []
            for k in range(seg):
                a = 2 * math.pi * k / seg
                co = P[i] + (nn * math.cos(a) + bb * math.sin(a)) * radii[i] * MM - off
                ring.append(bm.verts.new(co))
            rings.append(ring)
        for i in range(n - 1):
            for k in range(seg):
                k2 = (k + 1) % seg
                bm.faces.new([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]])
        if caps:
            bm.faces.new(rings[0][::-1])
            bm.faces.new(rings[-1])
        return self._finish_bm(name, bm, mat, parent or self.default_parent, coll or self.cur, True, 50.0)

    def sweep_rect(self, name, path_xz, w, d, y, mat="black_paint", parent=None, coll=None, plane="xz"):
        """Sweep a w x d rectangle along a polyline in the XZ plane (mm); d is the extent along Y centred on y.
        plane='yz' sweeps in the YZ plane instead (d along X centred on y=x position)."""
        off = self.world_off(parent or self.default_parent)
        P = [Vector((p[0], p[1])) for p in path_xz]
        n = len(P)
        bm = bmesh.new()
        rings = []
        for i in range(n):
            if i == 0:
                t = (P[1] - P[0]).normalized()
            elif i == n - 1:
                t = (P[-1] - P[-2]).normalized()
            else:
                t = ((P[i + 1] - P[i]).normalized() + (P[i] - P[i - 1]).normalized()).normalized()
            nr = Vector((-t.y, t.x))
            ring = []
            for (a, b) in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                q = P[i] + nr * (a * w / 2)
                if plane == "xz":
                    co = Vector((q.x, y + b * d / 2, q.y))
                else:
                    co = Vector((y + b * d / 2, q.x, q.y))
                ring.append(bm.verts.new(co * MM - off))
            rings.append(ring)
        for i in range(n - 1):
            for k in range(4):
                k2 = (k + 1) % 4
                bm.faces.new([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]])
        bm.faces.new(rings[0][::-1])
        bm.faces.new(rings[-1])
        return self._finish_bm(name, bm, mat, parent or self.default_parent, coll or self.cur, True, 40.0)

    def cut(self, target, cutters):
        """Boolean difference (exact) of cutter objects from target, applied; cutters are deleted."""
        for c in cutters:
            m = target.modifiers.new("cut", "BOOLEAN")
            m.operation = "DIFFERENCE"
            m.object = c
            m.solver = "EXACT"
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(target.evaluated_get(dg))
        old = target.data
        target.modifiers.clear()
        target.data = me
        bpy.data.meshes.remove(old)
        for c in cutters:
            cm = c.data
            bpy.data.objects.remove(c)
            bpy.data.meshes.remove(cm)
        bm = bmesh.new()
        bm.from_mesh(me)
        lim = math.radians(35)
        for f in bm.faces:
            f.smooth = True
        for e in bm.edges:
            if len(e.link_faces) == 2:
                try:
                    e.smooth = e.calc_face_angle() < lim
                except ValueError:
                    pass
        bm.to_mesh(me)
        bm.free()
        return target

    def boxes_obj(self, name, boxes, mats, pos=(0, 0, 0), parent=None, coll=None, bottom=False, recenter=False):
        """One mesh from many axis-aligned boxes. boxes: (cx, cy, cz, sx, sy, sz, slot) in mm relative to pos.
        mats: list of materials (slot index per box). Object origin = pos (asset space, mm) unless recenter."""
        parent = parent or self.default_parent
        off = self.world_off(parent)
        bm = bmesh.new()
        for (cx, cy, cz, sx, sy, sz, slot) in boxes:
            x0, x1 = (cx - sx / 2) * MM, (cx + sx / 2) * MM
            y0, y1 = (cy - sy / 2) * MM, (cy + sy / 2) * MM
            z0, z1 = (cz - sz / 2) * MM, (cz + sz / 2) * MM
            v = [bm.verts.new(c) for c in ((x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                                            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))]
            fs = [(4, 5, 6, 7), (0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2)]
            if bottom:
                fs.append((0, 3, 2, 1))
            for f in fs:
                fc = bm.faces.new([v[i] for i in f])
                fc.material_index = slot
                fc.smooth = False
        # outward normals: faces above are built with consistent winding for top/side; fix per face
        for f in bm.faces:
            c = f.calc_center_median()
            # box centre not stored: use face normal vs vector from (face centre projected) -> check z up for top
        bm.normal_update()
        bm.verts.ensure_lookup_table()
        pv = Vector((pos[0] * MM, pos[1] * MM, pos[2] * MM))
        ob_loc = pv - off
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(name, me)
        (coll or self.cur).objects.link(ob)
        ob.parent = parent
        ob.location = ob_loc
        for m in mats:
            me.materials.append(self._m(m))
        self.parts[name] = ob
        return ob

    def dup(self, src, name, pos, mat=None, rot_z=0.0, parent=None, coll=None, offset_is_local=False):
        """Linked duplicate (shares mesh data). pos mm in asset space (position of the object origin)."""
        parent = parent or self.default_parent
        ob = bpy.data.objects.new(name, src.data)
        (coll or self.cur).objects.link(ob)
        ob.parent = parent
        p = Vector((pos[0] * MM, pos[1] * MM, pos[2] * MM))
        ob.location = p if offset_is_local else p - self.world_off(parent)
        ob.rotation_euler = (0, 0, rot_z)
        if mat is not None and len(ob.material_slots) > 0:
            sl = ob.material_slots[0]
            sl.link = "OBJECT"
            sl.material = self._m(mat)
        self.parts[name] = ob
        return ob

    def text(self, name, body, size, pos, mat="label_white", rot=(0, 0, 0), extrude=0.0, align="CENTER", parent=None,
             coll=None):
        """Text converted to a real mesh. size in mm. rot (rad) about X,Y,Z euler applied about pos."""
        cu = bpy.data.curves.new(name, "FONT")
        cu.body = body
        cu.size = size * MM
        cu.align_x = align
        cu.align_y = "CENTER"
        cu.extrude = extrude * MM
        tmp = bpy.data.objects.new(name + "_tmp", cu)
        bpy.context.scene.collection.objects.link(tmp)
        dg = bpy.context.evaluated_depsgraph_get()
        me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
        bpy.context.scene.collection.objects.unlink(tmp)
        bpy.data.objects.remove(tmp)
        bpy.data.curves.remove(cu)
        R = Matrix.Rotation(rot[2], 3, "Z") @ Matrix.Rotation(rot[1], 3, "Y") @ Matrix.Rotation(rot[0], 3, "X")
        off = self.world_off(parent or self.default_parent)
        P = Vector((pos[0] * MM, pos[1] * MM, pos[2] * MM))
        bm = bmesh.new()
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
        for v in bm.verts:
            v.co = R @ v.co + P - off
        return self._finish_bm(name, bm, mat, parent or self.default_parent, coll or self.cur, False, 35.0,
                               fix_normals=False)

    # ---------------------------------------------------------- empties, props, drivers
    def empty(self, name, pos, parent=None, kind="PLAIN_AXES", size=0.02):
        parent = parent or self.default_parent
        e = bpy.data.objects.new(name, None)
        e.empty_display_type = kind
        e.empty_display_size = size
        self.cur.objects.link(e)
        e.parent = parent
        e.location = Vector((pos[0] * MM, pos[1] * MM, pos[2] * MM)) - self.world_off(parent)
        return e

    def hook(self, name, pos, parent=None, size=0.05):
        parent = parent or self.default_parent
        h = self.empty("HOOK_" + name, pos, parent, kind="ARROWS", size=size)
        self.hooks[h.name] = h
        return h

    def frame(self, name, pos, parent=None):
        """Plain moving-frame empty (not a documented hook)."""
        return self.empty(name, pos, parent, kind="PLAIN_AXES", size=0.02)

    def prop(self, name, default, doc, lo=None, hi=None, unit=""):
        self.root[name] = default
        try:
            ui = self.root.id_properties_ui(name)
            kw = {}
            if lo is not None:
                kw["min"] = lo
                kw["soft_min"] = lo
            if hi is not None:
                kw["max"] = hi
                kw["soft_max"] = hi
            ui.update(description=doc, **kw)
        except Exception:
            pass
        self.props[name] = {"default": default, "unit": unit, "description": doc, "min": lo, "max": hi}

    def drive(self, ob, path, index, expr, props):
        """Scripted driver on ob.<path>[index] = expr, with variables named like the root custom props."""
        fc = ob.driver_add(path, index)
        d = fc.driver
        d.type = "SCRIPTED"
        d.expression = expr
        for p in props:
            v = d.variables.new()
            v.name = p
            v.type = "SINGLE_PROP"
            v.targets[0].id = self.root
            v.targets[0].data_path = '["%s"]' % p
        return fc

    # ---------------------------------------------------------- metadata
    def dim(self, item, value, unit, source, level):
        self.dims.append({"item": item, "value": value, "unit": unit, "source": source, "accuracy": level})

    def src(self, what, where, used_for, accessed="2026-10-02"):
        self.sources.append({"what": what, "where": where, "used_for": used_for, "accessed": accessed})

    def screen_plane(self, name, center, size, y_face, parent=None):
        """Screen plane facing -Y at y = y_face (mm), centred at (cx, cz) with (w, h) mm; UV 0..1; MAT_fab_test_screen with an empty
        image-texture node (IMG_screen) mixed by MIX_use_image (0 = default pattern, 1 = image)."""
        full = "MAT_%s_screen" % CAT
        if full in bpy.data.materials:
            m = bpy.data.materials[full]
        else:
            m = bpy.data.materials.new(full)
            m.use_nodes = True
            nt = m.node_tree
            for n_ in list(nt.nodes):
                nt.nodes.remove(n_)
            o = nt.nodes.new("ShaderNodeOutputMaterial")
            em = nt.nodes.new("ShaderNodeEmission")
            mx = nt.nodes.new("ShaderNodeMix")
            mx.data_type = "RGBA"
            fv = nt.nodes.new("ShaderNodeValue")
            fv.name = "MIX_use_image"
            fv.label = "MIX_use_image (0 default, 1 image)"
            fv.outputs[0].default_value = 0.0
            im = nt.nodes.new("ShaderNodeTexImage")
            im.name = "IMG_screen"
            im.label = "IMG_screen (assign image sequence)"
            tc = nt.nodes.new("ShaderNodeTexCoord")
            gr = nt.nodes.new("ShaderNodeTexGradient")
            rp = nt.nodes.new("ShaderNodeValToRGB")
            rp.color_ramp.elements[0].color = (0.0, 0.04, 0.1, 1)
            rp.color_ramp.elements[1].color = (0.1, 0.5, 0.9, 1)
            nt.links.new(tc.outputs["UV"], im.inputs["Vector"])
            nt.links.new(tc.outputs["UV"], gr.inputs["Vector"])
            nt.links.new(gr.outputs["Fac"], rp.inputs["Fac"])
            nt.links.new(rp.outputs["Color"], mx.inputs[6])
            nt.links.new(im.outputs["Color"], mx.inputs[7])
            nt.links.new(fv.outputs[0], mx.inputs[0])
            nt.links.new(mx.outputs[2], em.inputs["Color"])
            nt.links.new(em.outputs[0], o.inputs["Surface"])
            self.mats[full] = m
        cx, cz = center
        w, h = size
        vv = [((cx + w / 2) * MM, y_face * MM, (cz - h / 2) * MM), ((cx - w / 2) * MM, y_face * MM, (cz - h / 2) * MM),
              ((cx - w / 2) * MM, y_face * MM, (cz + h / 2) * MM), ((cx + w / 2) * MM, y_face * MM, (cz + h / 2) * MM)]
        ob = self.obj_from_mesh(name, vv, [(0, 1, 2, 3)], m, parent=parent, smooth=False)
        uv = ob.data.uv_layers.new(name="UVMap")
        for li, c in zip(ob.data.polygons[0].loop_indices, ((0, 0), (1, 0), (1, 1), (0, 1))):
            uv.data[li].uv = c
        return ob

    def used_materials(self):
        names = set()
        for o in bpy.data.objects:
            for s in o.material_slots:
                if s.material is not None:
                    names.add(s.material.name)
        return sorted(n for n in names if n.startswith("MAT_"))

    def finish(self, out_dir, meta, views=("front", "three_quarter", "top"), closeups=(), preview=True):
        meta = dict(meta)
        meta["category"] = CAT
        meta["date_built"] = "2026-10-02"
        meta["units"] = "1 Blender unit = 1 m; geometry in real size; Z up; -Y front"
        meta["sources"] = self.sources
        meta["dimension_table"] = self.dims
        meta["hooks"] = [{"name": n, "parent": (h.parent.name if h.parent else None)} for n, h in sorted(self.hooks.items())]
        meta["custom_properties"] = self.props
        meta["material_slots"] = self.used_materials()
        meta["root"] = self.root.name
        meta["collection"] = self.coll.name
        meta["objects"] = len([o for o in self.coll.all_objects])
        bpy.context.view_layer.update()
        blend = os.path.join(out_dir, self.id + ".blend")
        C.finish(self.id, blend, self.coll, meta, preview_dir=None)
        if preview:
            render_previews(self, os.path.join(out_dir, "previews"), views, closeups)
        return meta


# ------------------------------------------------------------------ previews
def _studio2(center, radius, floor_z):
    w = bpy.data.worlds.new("PREVIEW_world")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.62, 0.68, 0.78, 1)
    bg.inputs["Strength"].default_value = 0.6
    bpy.context.scene.world = w
    sun = bpy.data.lights.new("PREVIEW_sun", "SUN")
    sun.energy = 2.2
    so = bpy.data.objects.new("PREVIEW_sun", sun)
    so.rotation_euler = (0.9, 0.3, 0.7)
    bpy.context.scene.collection.objects.link(so)
    for nm, loc, e in (("PREVIEW_key", (-1.6, -2.2, 1.6), 1.0), ("PREVIEW_rim", (1.8, -1.0, 1.2), 0.5)):
        a = bpy.data.lights.new(nm, "AREA")
        d = max(radius, 1e-3) * 4.0
        a.energy = 60 * d * d * e
        a.size = max(radius, 1e-3) * 3
        ao = bpy.data.objects.new(nm, a)
        ao.location = (center[0] + loc[0] * radius * 2, center[1] + loc[1] * radius * 2, center[2] + loc[2] * radius * 2)
        c = ao.constraints.new("TRACK_TO")
        c.track_axis = "TRACK_NEGATIVE_Z"
        c.up_axis = "UP_Y"
        t = bpy.data.objects.new(nm + "_t", None)
        t.location = center
        bpy.context.scene.collection.objects.link(t)
        c.target = t
        bpy.context.scene.collection.objects.link(ao)
    bpy.ops.mesh.primitive_plane_add(size=max(radius, 1e-3) * 24, location=(center[0], center[1], floor_z))
    fl = bpy.context.active_object
    fl.name = "PREVIEW_floor"
    fl.data.materials.append(C.principled("PREVIEW_floor_mat", (0.5, 0.52, 0.5), rough=0.85))


def render_previews(asset, out_dir, views, closeups, res=(900, 675), samples=16, floor=True):
    """views: names from the standard set; closeups: list of (name, cam_pos_mm, target_mm, lens_mm).
    Variants hidden for render are respected by the caller through asset.preview_hide (list of collection names)."""
    os.makedirs(out_dir, exist_ok=True)
    coll = asset.coll
    bb = C.bbox_mm(coll)
    lo, hi = Vector(bb[0]) * MM, Vector(bb[1]) * MM
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 0.01)
    scn = bpy.context.scene
    scn.render.engine = "BLENDER_EEVEE_NEXT"
    scn.eevee.taa_render_samples = samples
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.image_settings.file_format = "PNG"
    scn.render.image_settings.compression = 90
    scn.render.image_settings.color_mode = "RGB"
    scn.view_settings.view_transform = "Standard"
    scn.eevee.use_shadows = getattr(asset, "preview_shadows", True)
    keep = {o for c in [coll] + list(coll.children_recursive) for o in c.objects}
    for o in scn.objects:
        if o not in keep:
            o.hide_render = True
    for cn in getattr(asset, "preview_hide", []):
        bpy.data.collections[cn].hide_render = True
    _studio2(center, radius, lo.z)
    cam_d = bpy.data.cameras.new("PREVIEW_cam")
    cam_d.sensor_width = 36
    cam = bpy.data.objects.new("PREVIEW_cam", cam_d)
    scn.collection.objects.link(cam)
    scn.camera = cam
    tgt = bpy.data.objects.new("PREVIEW_tgt", None)
    scn.collection.objects.link(tgt)
    cons = cam.constraints.new("TRACK_TO")
    cons.track_axis = "TRACK_NEGATIVE_Z"
    cons.up_axis = "UP_Y"
    cons.target = tgt
    dirs = {"front": Vector((0, -1, 0.15)), "three_quarter": Vector((0.8, -0.9, 0.6)), "top": Vector((0, -0.05, 1)),
            "back": Vector((0, 1, 0.3)), "side": Vector((1, 0, 0.2)), "low": Vector((0.6, -0.8, -0.1)),
            "three_quarter_hi": Vector((-0.8, -0.9, 0.7))}
    out = []

    def shoot(name, loc, target, lens, clip_start=None):
        cam.location = loc
        tgt.location = target
        cam_d.lens = lens
        cam_d.clip_start = clip_start or max(radius * 0.002, 1e-4)
        cam_d.clip_end = max(radius * 200, 10)
        path = os.path.join(out_dir, "%s_%s.png" % (asset.id, name))
        scn.render.filepath = path
        bpy.ops.render.render(write_still=True)
        out.append(path)

    fit = getattr(asset, "preview_fit", 1.0)
    for v in views:
        d = dirs[v].normalized()
        t = center + Vector(getattr(asset, "preview_target_off", (0, 0, 0)))
        dist = radius * 3.2 / fit
        shoot(v, t + d * dist, t, 50, clip_start=max(dist - radius * 1.3, dist * 0.2))
    for ent in closeups:
        name, cpos, tpos, lens = ent[:4]
        if len(ent) > 4 and ent[4] is not None:
            ent[4]()
            bpy.context.view_layer.update()
        shoot(name, Vector(cpos) * MM, Vector(tpos) * MM, lens, clip_start=(ent[5] if len(ent) > 5 else 2e-4))
    return out


def write_extra_meta(out_dir, asset_id, extra):
    p = os.path.join(out_dir, asset_id + ".json")
    with open(p) as f:
        m = json.load(f)
    m.update(extra)
    with open(p, "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
