"""Package group: Spectrum-6 CPO package (substrate, stiffener ring with notched corners, 4 x 8 optical engines,
central lid) and its black display stand (fake solid back plate, ledge, rod, post, base). Built in the corrected
package frame (origin at the substrate top center, Z out of the package top), then moved with
lib.apply_frame(objs, "package") after the frame correction in config/model/package.toml. Phase 1A block-out
(package agent, 2026-10-05); measurements in DevLog/parts/DevLog-002-package.md."""

import json
import math
import os

import bmesh
import bpy
import lib
from mathutils import Matrix, Vector


def _boxes(name, items, coll, mat):
    """One mesh object from several axis-aligned boxes: items = [(x0, x1, y0, y1, z0, z1), ...] in mm."""
    bm = bmesh.new()
    for x0, x1, y0, y1, z0, z1 in items:
        M = Matrix.Translation(Vector(lib.mm((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))) @ \
            Matrix.Diagonal(Vector((*lib.mm(x1 - x0, y1 - y0, z1 - z0), 1.0)))
        bmesh.ops.create_cube(bm, size=1.0, matrix=M)
    return lib._obj_from_bm(name, bm, coll, mat)


def _bx(name, x0, x1, y0, y1, z0, z1, coll, mat):
    return lib.box(name, (x1 - x0, y1 - y0, z1 - z0), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), coll, mat)


def _rod(name, p0, p1, r, coll, mat):
    """Cylinder between two points (mm)."""
    a, b = Vector(p0), Vector(p1)
    d = b - a
    ob = lib.cylinder(name, r, d.length, (0, 0, 0), coll, mat=mat, verts=24)
    q = Vector((0, 0, 1)).rotation_difference(d.normalized())
    ob.data.transform(Matrix.Translation(Vector(lib.mm(*((a + b) / 2)))) @ q.to_matrix().to_4x4())
    return ob


def _fillet_poly(pts, radii, seg=8):
    """Round each vertex of a closed 2D polygon (mm) with its radius (0 = sharp)."""
    out = []
    n = len(pts)
    for i in range(n):
        P, A, B, r = Vector(pts[i]), Vector(pts[i - 1]), Vector(pts[(i + 1) % n]), radii[i]
        if r <= 0:
            out.append(P)
            continue
        u1, u2 = (A - P).normalized(), (B - P).normalized()
        th = u1.angle(u2) / 2
        d = r / math.tan(th)
        C = P + (u1 + u2).normalized() * (r / math.sin(th))
        t1, t2 = P + u1 * d, P + u2 * d
        a1 = math.atan2(t1.y - C.y, t1.x - C.x)
        a2 = math.atan2(t2.y - C.y, t2.x - C.x)
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        out += [C + Vector((math.cos(a1 + da * k / seg), math.sin(a1 + da * k / seg))) * r for k in range(seg + 1)]
    return out


def _prism(name, pts, z0, z1, coll, mat=None):
    """Vertical prism from a closed 2D polygon (mm, either winding) between z0 and z1."""
    bm = bmesh.new()
    vb = [bm.verts.new(Vector(lib.mm(p[0], p[1], z0))) for p in pts]
    fb = bm.faces.new(vb)
    ext = bmesh.ops.extrude_face_region(bm, geom=[fb])
    top = [v for v in ext["geom"] if isinstance(v, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, vec=Vector((0, 0, (z1 - z0) * lib.MM)), verts=top)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._obj_from_bm(name, bm, coll, mat)


def _rrect(x0, x1, y0, y1, r):
    return _fillet_poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], [r] * 4)


def _oriented_box(name, center, ax, ay, az, size, coll, mat):
    """Box with local axes ax, ay, az (unit vectors) and size (mm) about center (mm)."""
    ob = lib.box(name, size, (0, 0, 0), coll, mat)
    M = Matrix((ax, ay, az)).transposed().to_4x4()
    M.translation = Vector(lib.mm(*center))
    ob.data.transform(M)
    return ob


def _planar_uv(ob, mat, hx, hy, min_nz):
    """UV = package xy over the substrate square (v up = +y); faces with normal z > min_nz get mat."""
    ob.data.materials.append(mat)
    slot = len(ob.data.materials) - 1
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    uv = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        if f.normal.z > min_nz:
            f.material_index = slot
        for lo in f.loops:
            x, y = lo.vert.co.x / lib.MM, lo.vert.co.y / lib.MM
            lo[uv].uv = ((x + hx) / (2 * hx), (y + hy) / (2 * hy))
    bm.to_mesh(ob.data)
    bm.free()


def _side_uv(ob, mat, hx, hy, top, bot, tol=0.05):
    """Outer-wall atlas (bake_top.py --sides): faces on |x| = hx or |y| = hy get mat; 4 bands top to bottom
    (+y, -y, +x, -x), u along +x (y sides) or +y (x sides), v from z = top (band top) to z = bot."""
    ob.data.materials.append(mat)
    slot = len(ob.data.materials) - 1
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    uv = bm.loops.layers.uv.get("UVMap") or bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        if abs(f.normal.z) > 0.2:
            continue
        P = [v.co / lib.MM for v in f.verts]
        for k, (ax, sg, h, L) in enumerate(((1, 1, hy, hx), (1, -1, hy, hx), (0, 1, hx, hy), (0, -1, hx, hy))):
            if all(abs(p[ax] - sg * h) < tol for p in P) and f.normal[ax] * sg > 0.8:
                f.material_index = slot
                for lo in f.loops:
                    c = lo.vert.co / lib.MM
                    s_ = c[1 - ax]
                    lo[uv].uv = ((s_ + L) / (2 * L), 1 - (k + (top - c.z) / (top - bot)) / 4)
                break
    bm.to_mesh(ob.data)
    bm.free()


def _wall_charts(objs, mat, ppm, width_px=2048, skip_mats=("package_side_tex",), max_nz=0.5):
    """Inner wall atlas (bake_walls.py): every near-vertical face of objs (|n.z| < max_nz; faces whose material
    is in skip_mats keep theirs) gets its own rectangular chart (face extent along its horizontal tangent x z,
    1 px padding), shelf-packed in object/face order into width_px columns. Faces get mat when it is given.
    Returns (charts, W, H); chart = object, origin o (mm, package frame), tangent t, normal n, w/h (mm),
    x0/y0/wpx/hpx (atlas px, row 0 = top)."""
    charts, x, y, row_h = [], 0, 0, 0
    per_obj = []
    for ob in objs:
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        uv = bm.loops.layers.uv.get("UVMap") or bm.loops.layers.uv.new("UVMap")
        slot = None
        if mat is not None:
            ob.data.materials.append(mat)
            slot = len(ob.data.materials) - 1
        sel = []
        skip_slots = {i for i, m in enumerate(ob.data.materials) if m and m.name in skip_mats}
        for f in bm.faces:
            if abs(f.normal.z) >= max_nz or f.material_index in skip_slots:
                continue
            n = Vector((f.normal.x, f.normal.y, 0)).normalized()
            t = Vector((-n.y, n.x, 0))
            P = [v.co / lib.MM for v in f.verts]
            ss = [p.dot(t) for p in P]
            zz = [p.z for p in P]
            w, h = max(ss) - min(ss), max(zz) - min(zz)
            if w < 1e-3 or h < 1e-3:
                continue
            wpx, hpx = int(math.ceil(w * ppm)) + 2, int(math.ceil(h * ppm)) + 2
            if x + wpx > width_px:
                x, y, row_h = 0, y + row_h, 0
            o = t * min(ss) + n * (P[0].dot(n))
            o.z = min(zz)
            c = {"object": ob.name, "o": list(o), "t": list(t), "n": list(n), "w": w, "h": h,
                 "x0": x, "y0": y, "wpx": wpx, "hpx": hpx}
            charts.append(c)
            sel.append((f, c))
            x += wpx
            row_h = max(row_h, hpx)
        per_obj.append((ob, bm, uv, slot, sel))
    W, H = width_px, max(64, int(math.ceil((y + row_h) / 64.0)) * 64)
    for ob, bm, uv, slot, sel in per_obj:
        for f, c in sel:
            if slot is not None:
                f.material_index = slot
            t, o = Vector(c["t"]), Vector(c["o"])
            for lo in f.loops:
                p = lo.vert.co / lib.MM
                su, sz = (p - o).dot(t), p.z - o.z
                lo[uv].uv = ((c["x0"] + 1 + su * ppm) / W, 1 - (c["y0"] + 1 + (c["h"] - sz) * ppm) / H)
        bm.to_mesh(ob.data)
        bm.free()
    return charts, W, H


def _up_local() -> Vector:
    """Gravity up (scene frame Z) expressed in the package frame."""
    fr = json.loads((lib.ROOT / "config" / "frames.json").read_text())["frames"]
    Rp = Matrix(fr["package"]["R"])
    zs = Vector([fr["scene"]["R"][i][2] for i in range(3)])
    return (Rp.transposed() @ zs).normalized()


def build(P: dict, coll) -> None:
    c = P["colors"]
    M = {
        "ring": lib.mat_pbr("package_ring", color=tuple(c["ring"]), roughness=0.35, metallic=0.0),
        "sub": lib.mat_pbr("package_substrate", color=tuple(c["substrate"]), roughness=0.5),
        "field": lib.mat_pbr("package_field", color=tuple(c["field"]), roughness=0.5),
        "oe": lib.mat_pbr("package_oe", color=tuple(c["oe"]), roughness=0.4),
        "chip": lib.mat_pbr("package_chip", color=tuple(c.get("chip", c["oe"])), roughness=0.4),
        "lid": lib.mat_pbr("package_lid", color=tuple(c["lid"]), roughness=0.25),
        "stand": lib.mat_pbr("package_stand", color=tuple(c["stand"]), roughness=0.6),
    }
    for k, m in M.items():
        sp = P.get("stand_specular", P.get("specular", 0.25)) if k == "stand" else P.get("specular", 0.25)
        m.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = sp
    objs = []
    S, R, F, O, L, T = P["substrate"], P["ring"], P["field"], P["oe"], P["lid"], P["stand"]
    hx, hy = S["size_x"] / 2, S["size_y"] / 2

    # substrate (top face at z = 0)
    objs.append(_bx("package.substrate", -hx, hx, -hy, hy, -S["thickness"], 0, coll, M["sub"]))

    # stiffener ring: outer slab minus corner notches minus the plus-shaped opening, lowered inner lips
    ring = _bx("package.ring", -hx, hx, -hy, hy, 0, R["top"], coll, M["ring"])
    n, e = R["notch"], 1.0
    for sx in (-1, 1):
        for sy in (-1, 1):
            xa, xb = sorted((sx * (hx - n), sx * (hx + e)))
            ya, yb = sorted((sy * (hy - n), sy * (hy + e)))
            lib.boolean(ring, _prism("_cut", _rrect(xa, xb, ya, yb, R.get("r_notch", 0)), -e, R["top"] + e, coll))
    ix, iy, cb = R["inner_x"], R["inner_y"], R["corner_block"]
    rb, ro = R.get("r_block", 0), R.get("r_open", 0)
    plus = [(ix, -cb), (ix, cb), (cb, cb), (cb, iy), (-cb, iy), (-cb, cb), (-ix, cb), (-ix, -cb), (-cb, -cb),
            (-cb, -iy), (cb, -iy), (cb, -cb)]
    rad = [ro, ro, rb, ro, ro, rb, ro, ro, rb, ro, ro, rb]
    lib.boolean(ring, _prism("_cut", _fillet_poly(plus, rad), -e, R["top"] + e, coll))
    if "lip_top" in R:
        lt, rx, ry = R["lip_top"], R["rim_x"], R["rim_y"]
        for s_ in (-1, 1):
            xa, xb = sorted((s_ * (ix - e), s_ * rx))
            lib.boolean(ring, _bx("_cut", xa, xb, -cb, cb, lt, R["top"] + e, coll, None))
            ya, yb = sorted((s_ * (iy - e), s_ * ry))
            lib.boolean(ring, _bx("_cut", -cb, cb, ya, yb, lt, R["top"] + e, coll, None))
    objs.append(ring)

    # light-blue substrate field inside the opening (bump/cap field)
    objs.append(_boxes("package.field", [(-ix, ix, -cb, cb, 0, F["thickness"]),
                                         (-cb, cb, -iy, -cb, 0, F["thickness"]),
                                         (-cb, cb, cb, iy, 0, F["thickness"])], coll, M["field"]))

    # optical engines: count per side at pitch, centered on each side; gold bracket (outer) + gray chip (inner)
    k = O["count"]
    a = [(i - (k - 1) / 2) * O["pitch"] for i in range(k)]
    for part, wkey, hkey, rk, ck in (("gold", "gold_width", "gold_height", "rows_gold_r", "cols_gold_r"),
                                     ("chip", "chip_width", "chip_height", "rows_chip_r", "cols_chip_r")):
        w2, h = O[wkey] / 2, O[hkey]
        (r0, r1), (c0, c1) = O[rk], O[ck]
        sides = {
            "top": [(u - w2, u + w2, r0, r1, 0, h) for u in a],
            "bottom": [(u - w2, u + w2, -r1, -r0, 0, h) for u in a],
            "right": [(c0, c1, u - w2, u + w2, 0, h) for u in a],
            "left": [(-c1, -c0, u - w2, u + w2, 0, h) for u in a],
        }
        for nm, items in sides.items():
            objs.append(_boxes(f"package.oe_{part}_{nm}", items, coll, M["oe" if part == "gold" else "chip"]))

    # central lid / die
    objs.append(_bx("package.lid", L["cx"] - L["size_x"] / 2, L["cx"] + L["size_x"] / 2,
                    L["cy"] - L["size_y"] / 2, L["cy"] + L["size_y"] / 2, 0, L["height"], coll, M["lid"]))

    # discrete components on the field (relief test): rectangles detected in the top texture (measure/caps.py)
    if F.get("caps") and F.get("cap_height", 0) > 0 and (lib.ROOT / F["caps"]).exists():
        rects = json.loads((lib.ROOT / F["caps"]).read_text())["rects_mm"]
        z0 = F["thickness"]
        objs.append(_boxes("package.caps", [(a, b, c_, d, z0, z0 + F["cap_height"]) for a, b, c_, d in rects],
                           coll, M[F.get("cap_side", "chip")]))

    # photo texture on all up-facing package faces: one planar image over the substrate square
    tex = P.get("texture", {}).get("top")
    if tex and (lib.ROOT / tex).exists():
        mt = lib.mat_image("package_top_tex", tex, roughness=P["texture"].get("roughness", 0.5))
        for ob in objs:
            _planar_uv(ob, mt, hx, hy, P["texture"].get("min_nz", 0.9))

    side = P.get("texture", {}).get("sides")
    if side and (lib.ROOT / side).exists():
        ms = lib.mat_image("package_side_tex", side, roughness=P["texture"].get("roughness", 0.5))
        for ob in objs[:2]:   # substrate, ring
            _side_uv(ob, ms, hx, hy, R["top"], -S["thickness"])

    walls = P.get("texture", {}).get("walls", "")
    dump = os.environ.get("PACKAGE_WALL_CHARTS")
    if (walls and (lib.ROOT / walls).exists()) or dump:
        mw = lib.mat_image("package_walls_tex", walls, roughness=P["texture"].get("roughness", 0.5)) \
            if walls and (lib.ROOT / walls).exists() else None
        skip_obj = tuple(P["texture"].get("walls_skip", []))   # e.g. OE chips: flat color scored better (LOO)
        skip_obj += ("package.caps", "package.stand")
        charts, W, H = _wall_charts([ob for ob in objs[1:] if not any(k in ob.name for k in skip_obj)], mw,
                                    P["texture"].get("walls_ppm", 20.0))
        if dump:
            with open(dump, "w") as fh:
                json.dump({"ppm": P["texture"].get("walls_ppm", 20.0), "W": W, "H": H, "charts": charts}, fh)

    # stand: fake solid back plate under the substrate, ledge under the +Y (gravity-bottom) edge, rod, post, base
    zb = -S["thickness"]
    b2 = T["back_size"] / 2
    objs.append(_bx("package.stand_back", -b2, b2, -b2, b2, zb - T["back_thickness"], zb, coll, M["stand"]))
    lc, ls = list(T["ledge"]), list(T["ledge_size"])
    lc[1], lc[2] = T.get("ledge_y", lc[1]), T.get("ledge_z", lc[2])        # scalar overrides (for fit_params)
    ls[1], ls[2] = T.get("ledge_t", ls[1]), T.get("ledge_h", ls[2])
    lr = min(T.get("ledge_r", 0.0), ls[0] / 2 - 0.01, ls[1] / 2 - 0.01)
    if lr > 0:   # stadium-shaped bar (rounded ends in the package plane), extruded along package z
        objs.append(_prism("package.stand_ledge", _rrect(lc[0] - ls[0] / 2, lc[0] + ls[0] / 2, lc[1] - ls[1] / 2,
                                                         lc[1] + ls[1] / 2, lr), lc[2] - ls[2] / 2,
                           lc[2] + ls[2] / 2, coll, M["stand"]))
    else:
        objs.append(lib.box("package.stand_ledge", tuple(ls), tuple(lc), coll, M["stand"]))
    objs.append(_rod("package.stand_rod", T["rod_p0"], T["rod_p1"], T["rod_r"], coll, M["stand"]))
    if "panel_fl" in T:
        # inclined black back panel (easel back): front face through FL, FR, BL; thickness behind it
        FL, FR, BL = Vector(T["panel_fl"]), Vector(T["panel_fr"]), Vector(T["panel_bl"])
        ax = (FR - FL).normalized()
        ay = (FL - BL) - ax * (FL - BL).dot(ax)
        D = ay.length
        ay.normalize()
        az = ax.cross(ay)
        if az.z < 0:
            az = -az                        # az toward the cameras; thickness goes to -az
        W = (FR - FL).length + T.get("panel_dw", 0.0)
        D += T.get("panel_dd", 0.0)
        Cf = (FL + FR) / 2 + ax * T.get("panel_dx", 0.0) + ay * T.get("panel_df", 0.0) + az * T.get("panel_dz", 0.0)
        Ry = Matrix.Rotation(math.radians(T.get("panel_yaw", 0.0)), 3, az)   # in-plane rotation about the normal
        ax, ay = Ry @ ax, Ry @ ay
        pan = _prism("package.stand_base", _rrect(-W / 2, W / 2, -D, 0, T.get("panel_r", 0)),
                     -T["panel_thickness"], 0, coll, M["stand"])
        Mx = Matrix((ax, ay, az)).transposed().to_4x4()
        Mx.translation = Vector(lib.mm(*Cf))
        pan.data.transform(Mx)
        objs.append(pan)

    # frame correction (rotation about the corrected origin, then translation; provisional-frame coords)
    fc = P.get("frame_correction", {})
    rx, ry, rz = (math.radians(v) for v in fc.get("rotate_deg", [0, 0, 0]))
    C = Matrix.Translation(Vector(lib.mm(*fc.get("translate_mm", [0, 0, 0])))) @ \
        Matrix.Rotation(rz, 4, "Z") @ Matrix.Rotation(ry, 4, "Y") @ Matrix.Rotation(rx, 4, "X")
    for ob in objs:
        ob.data.transform(C)
    lib.apply_frame(objs, "package")
