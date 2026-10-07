"""Chassis group (tray shell): floor pan, side walls, rear wall, front panel (plate, lip, grip bar, MMC adapter
blocks, RJ45, USB, LED row), corner levers, NVIDIA crossbar, divider, black side ducts. All dimensions in
config/model/chassis.toml (mm, tray frame). Phase 1A block-out (chassis agent, 2026-10-05)."""

import sys
from pathlib import Path

import lib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chassis_surfaces import surfaces  # noqa: E402


def _lin(c8: float) -> float:
    c = c8 / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _mats(P):
    out = {}
    for name, rough in (("metal", 0.45), ("bezel", 0.5), ("black", 0.6), ("green", 0.6), ("port", 0.6)):
        m = lib.mat_pbr(f"chassis_{name}", color=tuple(_lin(v) for v in P["colors"][name]), roughness=rough)
        m.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.25
        out[name] = m
    return out


def _bx(name, x0, x1, y0, y1, z0, z1, coll, mat):
    return lib.box(name, (x1 - x0, y1 - y0, z1 - z0), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), coll, mat)


def _extrude_x(name, prof_yz, x0, x1, coll, mat):
    """Prism: closed polygon in the Y-Z plane (mm, either winding) extruded from X = x0 to x1."""
    import bmesh
    from mathutils import Vector
    bm = bmesh.new()
    a = [bm.verts.new(lib.mm(x0, y, z)) for y, z in prof_yz]
    bm.faces.new(a)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    r = bmesh.ops.extrude_face_region(bm, geom=list(bm.faces))
    bmesh.ops.translate(bm, vec=Vector(lib.mm(x1 - x0, 0, 0)), verts=[e for e in r["geom"] if isinstance(e, bmesh.types.BMVert)])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._obj_from_bm(name, bm, coll, mat)


def _holed_quad(name, corners, holes, coll, mat, offset_mm=0.05):
    """Front-facing (normal -Y) textured quad at constant Y from corners TL, TR, BR, BL (X right, Z down), split into grid
    cells at the hole edges; cells inside holes (x0, x1, z0, z1) are left out. UVs = position in the full quad."""
    import bmesh
    from mathutils import Vector
    (xa, y, za), (xb, _, _), _, (_, _, zc_) = corners
    y -= offset_mm
    xs = sorted({xa, xb} | {v for h in holes for v in h[:2]})
    zs = sorted({za, zc_} | {v for h in holes for v in h[2:]})
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    for i in range(len(xs) - 1):
        for j in range(len(zs) - 1):
            cx, cz = (xs[i] + xs[i + 1]) / 2, (zs[j] + zs[j + 1]) / 2
            if any(h[0] <= cx <= h[1] and h[2] <= cz <= h[3] for h in holes):
                continue
            q = [(xs[i], zs[j + 1]), (xs[i], zs[j]), (xs[i + 1], zs[j]), (xs[i + 1], zs[j + 1])]  # TL, BL, BR, TR
            f_ = bm.faces.new([bm.verts.new(Vector(lib.mm(x, y, z))) for x, z in q])
            for loop, (x, z) in zip(f_.loops, q):
                loop[uvl].uv = ((x - xa) / (xb - xa), (z - zc_) / (za - zc_))
    return lib._obj_from_bm(name, bm, coll, mat)


def _arc_shell(ye, sg, zt, r, t, zbot, n=8):
    """Y-Z polygon of a sheet (thickness t) that runs from the top plate edge (Y = ye + sg r, Z = zt) through a
    quarter arc of outer radius r down to the vertical flange face at Y = ye, then straight down to Z = zbot.
    sg = +1 for the front edge (flange at the smaller Y), -1 for the rear edge."""
    import math
    cy, cz = ye + sg * r, zt - r
    outer = [(cy - sg * r * math.sin(math.radians(90 * i / n)), cz + r * math.cos(math.radians(90 * i / n)))
             for i in range(n + 1)] + [(ye, zbot)]
    ri = max(r - t, 0.01)
    inner = [(ye + sg * t, zbot)] + [(cy - sg * ri * math.sin(math.radians(90 * i / n)),
                                       cz + ri * math.cos(math.radians(90 * i / n))) for i in range(n, -1, -1)]
    return outer + inner


def _rrect(x0, x1, y0, y1, r, n=6):
    """Rounded rectangle in the X-Y plane (mm), counter-clockwise."""
    import math
    pts = []
    for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _extrude_z(name, poly_xy, z0, z1, coll, mat):
    """Prism: closed polygon in the X-Y plane (mm) from Z = z0 to z1."""
    import bmesh
    from mathutils import Vector
    bm = bmesh.new()
    bm.faces.new([bm.verts.new(lib.mm(x, y, z0)) for x, y in poly_xy])
    r = bmesh.ops.extrude_face_region(bm, geom=list(bm.faces))
    bmesh.ops.translate(bm, vec=Vector(lib.mm(0, 0, z1 - z0)), verts=[e for e in r["geom"] if isinstance(e, bmesh.types.BMVert)])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._obj_from_bm(name, bm, coll, mat)


def build(P: dict, coll) -> None:
    M = _mats(P)
    b, w = P["body"], P["wall"]
    X, t, zb, zr, yr = b["x_outer"], b["t"], b["z_bot"], b["z_rim"], b["y_rear"]
    f = P["front"]

    # floor pan and walls
    _bx("chassis.floor", -X, X, f["y_face"], yr, zb, zb + t, coll, M["metal"])
    for s, nm in ((-1, "wall_mx"), (1, "wall_px")):
        xo, xi = s * X, s * (X - t)
        _bx(f"chassis.{nm}", min(xo, xi), max(xo, xi), w["y0"], yr, zb, zr, coll, M["metal"])
        xf = s * (X - w["flange_w"])
        _bx(f"chassis.{nm}_flange", min(xo, xf), max(xo, xf), w["y0"], yr, zr - t, zr, coll, M["metal"])
    _bx("chassis.rear_wall", -X, X, yr - t, yr, zb, P["rear"]["z_top"], coll, M["metal"])

    # front panel
    yf = f["y_face"]
    _bx("chassis.front_plate", -X, X, yf, yf + f["plate_t"], f["z_bot"], f["z_top"], coll, M["bezel"])
    _bx("chassis.front_lip", -X, X, yf + f["plate_t"], yf + f["lip_d"], f["z_top"] - f["lip_h"], f["z_top"], coll, M["metal"])
    xi = X - f["bezel_inset_x"]
    _extrude_x("chassis.front_grip", f["grip"]["profile_yz"], -xi, xi, coll, M["bezel"])

    m = f["mmc"]  # Phase 1A: one block per group (rows come with the texture pass)
    for gi, xc in enumerate(m["x_centers"]):
        _bx(f"chassis.mmc_g{gi}", xc - m["w"] / 2, xc + m["w"] / 2, yf - m["depth"], yf, m["z0"], m["z1"],
            coll, M["green"])
    holes = []
    if f.get("ports", {}).get("recess", False):  # RJ45 / USB openings cut through the plate, dark open cavities behind
        rj, u, dep = f["rj45"], f["usb"], f["ports"].get("depth", 12.0)
        holes = [(xc - rj["w"] / 2, xc + rj["w"] / 2, rj["zc"] - rj["h"] / 2, rj["zc"] + rj["h"] / 2) for xc in rj["x_centers"]]
        holes.append((u["xc"] - u["w"] / 2, u["xc"] + u["w"] / 2, u["zc"] - u["h"] / 2, u["zc"] + u["h"] / 2))
        plate = coll.objects["chassis.front_plate"]
        for i, (x0, x1, z0, z1) in enumerate(holes):
            lib.boolean(plate, _bx(f"_cut{i}", x0, x1, yf - 1, yf + f["plate_t"] + 1, z0, z1, coll, None))
            nm = f"chassis.port_{i}"
            _bx(nm, x0 - 0.6, x1 + 0.6, yf + f["plate_t"], yf + dep, z0 - 0.6, z1 + 0.6, coll, M["port"])  # back
            _bx(nm + "_t", x0, x1, yf, yf + dep, z1, z1 + 0.6, coll, M["port"])
            _bx(nm + "_b", x0, x1, yf, yf + dep, z0 - 0.6, z0, coll, M["port"])
            _bx(nm + "_l", x0 - 0.6, x0, yf, yf + dep, z0, z1, coll, M["port"])
            _bx(nm + "_r", x1, x1 + 0.6, yf, yf + dep, z0, z1, coll, M["port"])
    mmc_holes = {}
    if m.get("recess", False):  # MMC port rows as shallow recesses in each adapter block (green walls)
        seps = [m["z1"]] + list(m["row_seps"]) + [m["z0"]]
        fr, dep = m.get("frame", 1.5), m.get("recess_depth", 4.0)
        for gi, xc in enumerate(m["x_centers"]):
            ob = coll.objects[f"chassis.mmc_g{gi}"]
            hs = []
            for ri in range(len(seps) - 1):
                x0, x1, z0, z1 = xc - m["w"] / 2 + fr, xc + m["w"] / 2 - fr, seps[ri + 1] + fr, seps[ri] - fr
                hs.append((x0, x1, z0, z1))
                ya = yf - m["depth"]
                lib.boolean(ob, _bx("_mcut", x0, x1, ya - 1, ya + dep, z0, z1, coll, None))
                nm = f"chassis.mmcport_g{gi}_r{ri}"
                _bx(nm, x0, x1, ya + dep - 0.6, ya + dep, z0, z1, coll, M[m.get("recess_back", "port")])  # opening back
            mmc_holes[f"mmc_g{gi}"] = hs
    if f.get("ports", {}).get("geometry", True):
        rj = f["rj45"]
        for i, xc in enumerate(rj["x_centers"]):
            _bx(f"chassis.rj45_{i}", xc - rj["w"] / 2, xc + rj["w"] / 2, yf - 0.6, yf, rj["zc"] - rj["h"] / 2,
                rj["zc"] + rj["h"] / 2, coll, M["port"])
        u = f["usb"]
        _bx("chassis.usb", u["xc"] - u["w"] / 2, u["xc"] + u["w"] / 2, yf - 0.6, yf, u["zc"] - u["h"] / 2,
            u["zc"] + u["h"] / 2, coll, M["port"])
        led = f["led"]
        _bx("chassis.led_row", led["x0"], led["x1"], yf - 0.4, yf, led["zc"] - 1, led["zc"] + 1, coll, M["port"])

    # corner levers: barrel along Z at each front corner (hook end and arm: Phase 1B)
    L = P["lever"]
    for s, nm in ((-1, "lever_mx"), (1, "lever_px")):
        xc = s * (X - L["xc_inset"])
        lib.cylinder(f"chassis.{nm}", L["r"], L["z1"] - L["z0"], (xc, L["yc"], (L["z0"] + L["z1"]) / 2), coll,
                     axis="z", mat=M["bezel"], verts=32)
        lib.cylinder(f"chassis.{nm}_head", L["head_r"], L["head_len"], (xc, L["yc"], L["z1"] - L["head_dz"]), coll,
                     axis="x", mat=M["bezel"], verts=32)

    cl = P["clips"]
    for s, side in ((-1, "mx"), (1, "px")):
        for i, yc in enumerate(cl["y_centers"]):
            xo, xi_ = s * X, s * (X - cl["w"])
            _bx(f"chassis.clip_{side}{i}", min(xo, xi_), max(xo, xi_), yc - cl["len_y"] / 2, yc + cl["len_y"] / 2,
                zr, zr + cl["h"], coll, M["metal"])

    # NVIDIA crossbar: top plate at Z 0 plus two flanges
    c = P["crossbar"]
    zc = c.get("z_top", 0.0)
    r = c.get("edge_r", 0.0)
    if r > 0:  # rounded long edges: top plate between the arcs, each flange = arc + vertical strip (thin shells)
        _bx("chassis.crossbar_top", -c["x_half"], c["x_half"], c["y0"] + r, c["y1"] - r, zc - t, zc, coll, M["metal"])
        for nm, ye, sg in (("crossbar_fl_front", c["y0"], 1), ("crossbar_fl_rear", c["y1"], -1)):
            _extrude_x(f"chassis.{nm}", _arc_shell(ye, sg, zc, r, t, -c["flange_depth"], int(c.get("edge_n", 8))),
                       -c["x_half"], c["x_half"], coll, M["metal"])
    else:
        _bx("chassis.crossbar_top", -c["x_half"], c["x_half"], c["y0"], c["y1"], zc - t, zc, coll, M["metal"])
        ye, ze = c.get("y_ext", 0.0), c.get("z_ext", zc)
        if ye > c["y1"]:  # rear ledge behind the stamped step at y1 (Wave 5c); rear flange moves to its edge
            _bx("chassis.crossbar_ext", -c["x_half"], c["x_half"], c["y1"], ye, min(zc, ze) - t, ze, coll, M["metal"])
        yf_, zf_ = c.get("y_ext_front", c["y0"]), c.get("z_ext_front", zc)
        if yf_ < c["y0"]:  # front ledge in front of the stamped step at y0 (Wave 6); front flange moves to its edge
            _bx("chassis.crossbar_ext_front", -c["x_half"], c["x_half"], yf_, c["y0"], min(zc, zf_) - t, zf_, coll,
                M["metal"])
        yr_ = max(ye, c["y1"])
        for nm, y, ztop in (("crossbar_fl_front", min(yf_, c["y0"]), zf_ if yf_ < c["y0"] else zc),
                            ("crossbar_fl_rear", yr_ - t, ze if ye > c["y1"] else zc)):
            _bx(f"chassis.{nm}", -c["x_half"], c["x_half"], y, y + t, -c["flange_depth"], ztop, coll, M["metal"])

    d = P["divider"]
    _bx("chassis.divider", -(X - t), X - t, d["y"], d["y"] + t, zb + t, d["z_top"] - t, coll, M["metal"])
    if "flange_y0" in d:  # horizontal top flange with rounded ends
        _extrude_z("chassis.divider_flange", _rrect(-d["flange_x"], d["flange_x"], d["flange_y0"], d["flange_y1"],
                                                    d["flange_r"]), d["z_top"] - t, d["z_top"], coll, M["metal"])

    du = P["ducts_front"]
    for s, nm in ((-1, "duct_front_mx"), (1, "duct_front_px")):
        _bx(f"chassis.{nm}", min(s * du["x_in"], s * du["x_out"]), max(s * du["x_in"], s * du["x_out"]),
            du["y0"], du["y1"], zb + t, du["z_top"], coll, M["black"])

    sh = P.get("shelf_px", {})
    if sh.get("enabled", False):  # silver shelf along the +X wall in the middle bay (Wave 5b)
        _bx("chassis.shelf_px", sh["x0"], X - t, sh["y0"], sh["y1"], sh["z_top"] - t, sh["z_top"], coll, M["metal"])

    rr = P.get("rail_rear_px", {})
    if rr.get("enabled", False):  # gray rail strip along the +X wall in the rear bay (Wave 5d)
        _bx("chassis.rail_rear_px", rr["x0"], rr["x1"], rr["y0"], rr["y1"], rr["z_top"] - t, rr["z_top"], coll, M["metal"])

    r_ = P["rails_front"]
    for s, nm in ((-1, "rail_front_mx"), (1, "rail_front_px")) if r_.get("enabled", True) else ():
        _bx(f"chassis.{nm}", min(s * r_["x_in"], s * r_["x_out"]), max(s * r_["x_in"], s * r_["x_out"]),
            r_["y0"], r_["y1"], r_["z_top"] - r_["t"], r_["z_top"], coll, M["metal"])

    # photo-textured quads (assets/textures/chassis_<name>.png), only where the texture exists
    T = P.get("tex", {})
    if T.get("enabled", False):
        for nm, sf in surfaces(P).items():
            png = lib.ROOT / "assets" / "textures" / f"chassis_{nm}{T.get(nm, {}).get('suffix', '')}.png"
            if png.exists() and not T.get(nm, {}).get("skip", False):
                if nm == "front_plate" and holes:
                    _holed_quad(f"chassis.tex_{nm}", sf["corners"], holes, coll, lib.mat_image(f"chassis_tex_{nm}", png))
                elif nm in mmc_holes:
                    _holed_quad(f"chassis.tex_{nm}", sf["corners"], mmc_holes[nm], coll,
                                lib.mat_image(f"chassis_tex_{nm}", png))
                else:
                    lib.label_quad(f"chassis.tex_{nm}", sf["corners"], coll, lib.mat_image(f"chassis_tex_{nm}", png))

    # unseen faces (undersides; back of the front panel; outer face of the rear wall): photographed wall/floor color,
    # untextured, so the model reads as one material in orbit renders (Wave 6; never seen by any camera)
    U = P.get("unseen", {})
    if U.get("enabled", False):
        mu = lib.mat_pbr("chassis_unseen", color=tuple(_lin(v) for v in P["colors"]["unseen"]), roughness=0.5)
        mu.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.25
        back = set(U.get("back_parts", []))
        for ob in list(coll.objects):
            if ob.type != "MESH" or ob.name.startswith(("chassis.tex_", "chassis.port_", "chassis.mmc")):
                continue
            me = ob.data
            idx = None
            for poly in me.polygons:
                nz, ny = poly.normal.z, poly.normal.y
                if nz < -0.9 or (ob.name in back and ny > 0.9):
                    if idx is None:
                        me.materials.append(mu)
                        idx = len(me.materials) - 1
                    poly.material_index = idx

    # all parts are built level (tray frame); shared "tray" frame (config/frames.json) carries the world roll
    rd = P["body"].get("roll_deg", 0.0)  # must stay 0 (no private correction); kept only for A/B diagnostics
    if rd:
        for ob in list(coll.objects):
            lib.place(ob, rot_deg=(0.0, rd, 0.0))
    lib.apply_frame(list(coll.objects), "tray")
