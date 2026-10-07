"""upper group: middle bay (switch ASIC area, boards, capacitors, baffles), rear bay (serpentine cold plate,
board2, ribbon cable), rear UQD fittings with brackets, and the visible copper tubes.

All dimensions come from config/model/upper.toml (mm, tray frame). Data-driven: each TOML entry
([[box]], [[prism]], [[cyl]], [[cyls]], [[tube]]) becomes one object "upper.<name>".
Textures: an entry with tex = "<t>" gets assets/textures/upper_<t>.png on its top faces (normal +Z) when the file
exists (flat material otherwise). UV = bounding box of the entry in XY (u along +X, v along +Y), so the bake quad
is TL (x0, y1, z1), TR (x1, y1, z1), BR (x1, y0, z1), BL (x0, y0, z1); [[cyls]] map every cap top to the full
image (one instanced texture). Bakes: scripts/model/upper/bake_upper.py.
"""

from __future__ import annotations


import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib  # noqa: E402

G = "upper"
ROOT = Path(__file__).resolve().parents[3]


def _texmat(e):
    t = e.get("tex")
    if not t:
        return None
    f = ROOT / "assets" / "textures" / f"upper_{t}.png"
    return lib.mat_image(f"{G}_tex_{t}", f) if f.exists() else None


def _apply_top_tex(ob, e, rects):
    """Assign the texture to top faces; rects = list of (x0, x1, y0, y1) in mm; a face uses the rect containing
    its center."""
    mt = _texmat(e)
    if mt is None:
        return
    ob.data.materials.append(mt)
    me = ob.data
    uvl = me.uv_layers.new(name="UVMap")
    for f in me.polygons:
        if f.normal.z < 0.9:
            continue
        c = f.center / lib.MM
        r = min(rects, key=lambda q: ((q[0] + q[1]) / 2 - c.x) ** 2 + ((q[2] + q[3]) / 2 - c.y) ** 2)
        f.material_index = 1
        for li in f.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co / lib.MM
            uvl.data[li].uv = ((co.x - r[0]) / (r[1] - r[0]), (co.y - r[2]) / (r[3] - r[2]))


def _lin(c8: float) -> float:
    c = c8 / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _mats(P: dict) -> dict:
    out = {}
    for name, v in P.get("mats", {}).items():
        r, g, b, rough, metal = v
        m = lib.mat_pbr(f"{G}_{name}", color=(_lin(r), _lin(g), _lin(b)), roughness=rough, metallic=metal)
        b_ = m.node_tree.nodes["Principled BSDF"]
        b_.inputs["Specular IOR Level"].default_value = 0.25
        out[name] = m
    return out


SIDES = ("xp", "xn", "yp", "yn")


def side_uv(side, b, co):
    """UV of a point co (mm) on a box side; image orientation as in the bake quads (bake_upper.side_quad):
    up = +Z, right = (up) x (outward normal) seen from outside."""
    x0, x1, y0, y1, z0, z1 = b
    v = (co.z - z0) / (z1 - z0)
    u = {"xp": (co.y - y0) / (y1 - y0), "xn": (y1 - co.y) / (y1 - y0),
         "yp": (x1 - co.x) / (x1 - x0), "yn": (co.x - x0) / (x1 - x0)}[side]
    return u, v


def _apply_side_tex(ob, e):
    """tex_sides = true: side faces of a box get assets/textures/upper_<tex>_<side>.png (side = xp/xn/yp/yn by the
    face normal) when the file exists."""
    t = e.get("tex")
    if not (t and e.get("tex_sides")):
        return
    me = ob.data
    uvl = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    for side in SIDES:
        f = ROOT / "assets" / "textures" / f"upper_{t}_{side}.png"
        if not f.exists():
            continue
        ob.data.materials.append(lib.mat_image(f"{G}_tex_{t}_{side}", f))
        mi = len(ob.data.materials) - 1
        ax, sg = {"xp": (0, 1), "xn": (0, -1), "yp": (1, 1), "yn": (1, -1)}[side]
        for p in me.polygons:
            if p.normal[ax] * sg < 0.9:
                continue
            p.material_index = mi
            for li in p.loop_indices:
                uvl.data[li].uv = side_uv(side, e["b"], me.vertices[me.loops[li].vertex_index].co / lib.MM)


def _box(e, coll, M):
    x0, x1, y0, y1, z0, z1 = e["b"]
    ob = lib.box(f"{G}.{e['name']}", (x1 - x0, y1 - y0, z1 - z0), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
                 coll, M[e["mat"]], bevel_mm=e.get("bevel", 0.0))
    _apply_top_tex(ob, e, [(x0, x1, y0, y1)])
    _apply_side_tex(ob, e)
    return ob


def plane_z(e, x, y):
    """Top height of a prism at (x, y): plane = [a, b, c] -> a + b x + c y, else z[1]."""
    if "plane" in e:
        a, b, c = e["plane"]
        return a + b * x + c * y
    return e["z"][1]


def _prism_plane(e, pts, coll, M):
    """Prism whose top follows the plane e["plane"] and whose bottom is e["thick"] below it (slanted sheet)."""
    th = e.get("thick", 3.0)
    bm = bmesh.new()
    top = [bm.verts.new(lib.mm(x, y, plane_z(e, x, y))) for x, y in pts]
    bot = [bm.verts.new(lib.mm(x, y, plane_z(e, x, y) - th)) for x, y in pts]
    bm.faces.new(top)
    bm.faces.new(bot[::-1])
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new([bot[i], bot[j], top[j], top[i]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._obj_from_bm(f"{G}.{e['name']}", bm, coll, M[e["mat"]])


def _prism(e, coll, M):
    pts = e["pts"]
    # make the outline counter-clockwise seen from +Z
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    if area < 0:
        pts = pts[::-1]
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    if "plane" in e:
        ob = _prism_plane(e, pts, coll, M)
        _apply_top_tex(ob, e, [(min(xs), max(xs), min(ys), max(ys))])
        return ob
    z0, z1 = e["z"]
    bm = bmesh.new()
    vs = [bm.verts.new(lib.mm(x, y, z0)) for x, y in pts]
    f = bm.faces.new(vs)
    f.normal_update()
    if f.normal.z > 0:  # bottom face must point down before extrusion
        f.normal_flip()
    r = bmesh.ops.extrude_face_region(bm, geom=[f])
    top = [g for g in r["geom"] if isinstance(g, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=Vector((0, 0, (z1 - z0) * lib.MM)), verts=top)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = lib._obj_from_bm(f"{G}.{e['name']}", bm, coll, M[e["mat"]])
    _apply_top_tex(ob, e, [(min(xs), max(xs), min(ys), max(ys))])
    return ob


def _cyl_side_tex(ob, t, axis, centers, h):
    """Side faces of cylinder(s) (normal perpendicular to the axis) get assets/textures/upper_<t>_side.png:
    u = angle (-180..180 deg from r0 toward r1 = axis x r0; r0 = +X, or +Y for an X axis), v = along the axis
    from the -h/2 end (0) to +h/2 (1); matches bake_upper.cyl_surface. centers: one per instance (mm)."""
    f = ROOT / "assets" / "textures" / f"upper_{t}_side.png"
    if not f.exists():
        return
    ob.data.materials.append(lib.mat_image(f"{G}_tex_{t}_side", f))
    mi = len(ob.data.materials) - 1
    me = ob.data
    uvl = me.uv_layers.get("UVMap") or me.uv_layers.new(name="UVMap")
    ai = {"x": 0, "y": 1, "z": 2}[axis]
    r0i, r1i, r1s = {"x": (1, 2, 1), "y": (0, 2, -1), "z": (0, 1, 1)}[axis]
    for p in me.polygons:
        if abs(p.normal[ai]) > 0.5:
            continue
        pc = p.center / lib.MM
        c = min(centers, key=lambda q: sum((q[i] - pc[i]) ** 2 for i in range(3) if i != ai))
        p.material_index = mi
        uvs = []
        for li in p.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co / lib.MM
            a = math.atan2(r1s * (co[r1i] - c[r1i]), co[r0i] - c[r0i])
            uvs.append([(a + math.pi) / (2 * math.pi), (co[ai] - (c[ai] - h / 2)) / h])
        us = [q[0] for q in uvs]
        if max(us) - min(us) > 0.5:  # face across the seam
            for q in uvs:
                if q[0] < 0.5:
                    q[0] += 1.0
        for li, q in zip(p.loop_indices, uvs):
            uvl.data[li].uv = q


def _cyl(e, coll, M):
    ob = lib.cylinder(f"{G}.{e['name']}", e["r"], e["h"], e["c"], coll, axis=e.get("axis", "z"),
                      mat=M[e["mat"]], verts=e.get("verts", 32))
    if e.get("tex"):
        _cyl_side_tex(ob, e["tex"], e.get("axis", "z"), [e["c"]], e["h"])
    return ob


def _cyls(e, coll, M):
    z0, z1 = e["z"]
    tr = e.get("top_round", 0.0)  # rounded top edge: frustum r -> r - tr over the top tr mm
    seg = e.get("verts", 20)
    bm = bmesh.new()
    for x, y in e["xy"]:
        r = bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=e["r"] * lib.MM,
                                  radius2=e["r"] * lib.MM, depth=(z1 - tr - z0) * lib.MM)
        bmesh.ops.translate(bm, vec=Vector(lib.mm(x, y, (z0 + z1 - tr) / 2)), verts=r["verts"])
        if tr > 0:
            r = bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=e["r"] * lib.MM,
                                      radius2=(e["r"] - tr) * lib.MM, depth=tr * lib.MM)
            bmesh.ops.translate(bm, vec=Vector(lib.mm(x, y, z1 - tr / 2)), verts=r["verts"])
    ob = lib._obj_from_bm(f"{G}.{e['name']}", bm, coll, M[e["mat"]])
    R = e["r"]
    _apply_top_tex(ob, e, [(x - R, x + R, y - R, y + R) for x, y in e["xy"]])
    if e.get("tex") and e.get("tex_sides"):
        _cyl_side_tex(ob, e["tex"], "z", [(x, y, (z0 + z1) / 2) for x, y in e["xy"]], z1 - z0)
    return ob


def _fillet(pts, rad, n=6):
    """Polyline with rounded corners (quadratic Bezier per corner; good arc approximation for a block-out)."""
    P = [Vector(p) for p in pts]
    if len(P) < 3 or rad <= 0:
        return P
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        la, lc = (a - b).length, (c - b).length
        d = min(rad, la / 2, lc / 2)
        if d < 1e-6:
            out.append(b)
            continue
        p0 = b + (a - b).normalized() * d
        p2 = b + (c - b).normalized() * d
        for k in range(n + 1):
            t = k / n
            out.append((1 - t) ** 2 * p0 + 2 * (1 - t) * t * b + t ** 2 * p2)
    out.append(P[-1])
    return out


def _tube(e, coll, M):
    pts = _fillet(e["pts"], e.get("fillet", 0.0), e.get("fillet_n", 6))
    name = f"{G}.{e['name']}"
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = e["r"] * lib.MM
    cu.bevel_resolution = e.get("bevel_res", 3)
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for p, v in zip(sp.points, pts):
        p.co = (v.x * lib.MM, v.y * lib.MM, v.z * lib.MM, 1.0)
    ob = bpy.data.objects.new(name, cu)
    coll.objects.link(ob)
    ob.data.materials.append(M[e["mat"]])
    return ob


def _boxes(e, coll, M):
    """Repeated boxes (b = list of [x0, x1, y0, y1, z0, z1]) as one object (clip teeth, small parts)."""
    bm = bmesh.new()
    for x0, x1, y0, y1, z0, z1 in e["b"]:
        r = bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=Vector(((x1 - x0) * lib.MM, (y1 - y0) * lib.MM, (z1 - z0) * lib.MM)), verts=r["verts"])
        bmesh.ops.translate(bm, vec=Vector(lib.mm((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)), verts=r["verts"])
    return lib._obj_from_bm(f"{G}.{e['name']}", bm, coll, M[e["mat"]])


def build(P: dict, coll) -> None:
    M = _mats(P)
    for kind, fn in (("box", _box), ("prism", _prism), ("cyl", _cyl), ("cyls", _cyls), ("tube", _tube),
                     ("boxes", _boxes)):
        for e in P.get(kind, []):
            if e.get("skip"):
                continue
            fn(e, coll, M)
