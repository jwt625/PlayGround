"""Lower group: front bay contents (Y 0-330): control board with NVIDIA logo, connector, clips, 8 copper
cold-plate fingers over the optical engines, copper loops/tubes to the center manifold, fiber organizers and
grouped blue fibers, black floor; braided cables over their full length. All dimensions in
config/model/lower.toml (mm, tray frame). Phase 1A block-out (lower agent, 2026-10-05)."""

import math

import bmesh
import bpy
from mathutils import Matrix, Vector

import sys
from pathlib import Path

import lib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tubegeom  # noqa: E402


def _mat(name, color, rough=0.5, metallic=0.0, spec=0.25):
    m = lib.mat_pbr(name, color=tuple(color), roughness=rough, metallic=metallic)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Specular IOR Level"].default_value = spec
    return m


def _mats(P):
    c = P["colors"]
    return {
        "board": _mat("lower_board", c["board"], 0.45),
        "copper": _mat("lower_copper", c["copper"], 0.35, metallic=1.0),
        "oe": _mat("lower_oe", c["oe"], 0.4, metallic=1.0),
        "base": _mat("lower_base", c["base"], 0.7),
        "org": _mat("lower_org", c["org"], 0.6),
        "strip": _mat("lower_strip", c["strip"], 0.6),
        "fiber": _mat("lower_fiber", c["fiber"], 0.4),
        "braid": _mat("lower_braid", c["braid"], 0.6),
        "braid_b": _mat("lower_braid_b", c["braid_b"], 0.6),
        "ribbon": _mat("lower_ribbon", c["ribbon"], 0.5),
        "conn": _mat("lower_conn", c["conn"], 0.5),
        "white": _mat("lower_white", c["white"], 0.6),
        "metal": _mat("lower_metal", c["metal"], 0.4, metallic=1.0),
        "silver": _mat("lower_silver", c["silver"], 0.35, metallic=1.0),
        "cable_blk": _mat("lower_cable_blk", c["cable_blk"], 0.5),
        "screw": _mat("lower_screw", c.get("screw", c["silver"]), 0.5),
        "screw_cu": _mat("lower_screw_cu", c.get("screw_cu", c["copper"]), 0.4, metallic=1.0),
    }


def _bx(name, b, coll, mat):
    x0, x1, y0, y1, z0, z1 = b
    return lib.box(name, (x1 - x0, y1 - y0, z1 - z0), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), coll, mat)


def _fillet(pts, r, step=1.5):
    """Polyline (mm) with corners rounded by arcs of radius <= r (limited by half the adjacent segments)."""
    P = [Vector(p) for p in pts]
    if len(P) < 3 or r <= 0:
        return [tuple(p) for p in P]
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        u, w = (a - b), (c - b)
        la, lc = u.length, w.length
        u.normalize()
        w.normalize()
        ang = u.angle(w)  # interior angle
        if ang > math.pi - 1e-3 or ang < 1e-3:
            out.append(b)
            continue
        t = r / math.tan(ang / 2)
        t = min(t, 0.5 * la, 0.5 * lc)
        rr = t * math.tan(ang / 2)
        p0, p1 = b + u * t, b + w * t
        bis = (u + w).normalized()
        ctr = b + bis * (rr / math.sin(ang / 2))
        v0, v1 = p0 - ctr, p1 - ctr
        sweep = v0.angle(v1)
        n = max(2, int(rr * sweep / step))
        for k in range(n + 1):
            s = k / n
            # slerp between v0 and v1 around ctr
            v = (math.sin((1 - s) * sweep) * v0 + math.sin(s * sweep) * v1) / math.sin(sweep)
            out.append(ctr + v)
    out.append(P[-1])
    return [tuple(p) for p in out]


def _tubes(name, paths, r, coll, mat, fillet=0.0):
    """One curve object with a round bevel; each path is a list of mm points (POLY splines, filleted)."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = r * lib.MM
    cu.bevel_resolution = 3
    cu.use_fill_caps = True
    for pts in paths:
        q = _fillet(pts, fillet) if fillet > 0 else pts
        sp = cu.splines.new("POLY")
        sp.points.add(len(q) - 1)
        for p, c in zip(sp.points, q):
            p.co = (c[0] * lib.MM, c[1] * lib.MM, c[2] * lib.MM, 1.0)
    ob = bpy.data.objects.new(name, cu)
    coll.objects.link(ob)
    ob.data.materials.append(mat)
    return ob


def _multi_box(name, boxes, coll, mat):
    """One mesh object made of several axis-aligned boxes (x0, x1, y0, y1, z0, z1) in mm."""
    bm = bmesh.new()
    for x0, x1, y0, y1, z0, z1 in boxes:
        g = bmesh.ops.create_cube(bm, size=1.0)
        vs = g["verts"]
        bmesh.ops.scale(bm, vec=Vector(lib.mm(x1 - x0, y1 - y0, z1 - z0)), verts=vs)
        bmesh.ops.translate(bm, vec=Vector(lib.mm((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)), verts=vs)
    return lib._obj_from_bm(name, bm, coll, mat)


def _uv_quads(name, rects, z, uvs, coll, mat):
    """Horizontal quads (facing +Z) at height z (mm), one per rect (x0, x1, y0, y1) with UV rect (u0, u1, v0, v1);
    v = 1 at y1 (image top), u = 0 at x0."""
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    for (x0, x1, y0, y1), (u0, u1, v0, v1) in zip(rects, uvs):
        vs = [bm.verts.new(lib.mm(x, y, z)) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        f = bm.faces.new(vs)  # counter-clockwise from above: normal +Z
        for loop, t in zip(f.loops, ((u0, v0), (u1, v0), (u1, v1), (u0, v1))):
            loop[uvl].uv = t
    return lib._obj_from_bm(name, bm, coll, mat)


def _tex_surfaces(P, coll, M, xs):
    """Photo-textured top surfaces ([[tex]] in the TOML). Without a baked file the quads get the flat material
    of the part below (bootstrap for the ID render the bake needs)."""
    F, O = P["fingers"], P["oe"]
    for t in P.get("tex", []):
        X0, X1, Y0, Y1 = t["rect"]
        f = lib.ROOT / "assets" / "textures" / t["file"]
        mat = lib.mat_image(f"lower_tex_{t['name']}", f) if f.exists() else M[t["flat"]]
        xsel = [xs[i] for i in t.get("idx", range(len(xs)))]  # Wave 4: optional subset of fingers/OEs per entry
        if t["kind"] == "fingers":
            rects = [(xc - F["w"] / 2, xc + F["w"] / 2, a, b) for xc in xsel for a, b in F["blocks"]]
        elif t["kind"] == "oe":
            rects = [(xc - O["w"] / 2, xc + O["w"] / 2, Y0, Y1) for xc in xsel]
        else:
            rects = [(X0, X1, Y0, Y1)]
        uvs = [((a - X0) / (X1 - X0), (b - X0) / (X1 - X0), (c - Y0) / (Y1 - Y0), (d - Y0) / (Y1 - Y0))
               for a, b, c, d in rects]
        _uv_quads(f"lower.tex_{t['name']}", rects, t["z"] + t.get("lift", 0.15), uvs, coll, mat)


def _tube_mesh(name, P, key, r, coll, mat, n_around=16, tex=None):
    """Cable as a UV-mapped tube mesh on the shared centerline (tubegeom): u along the length, v around
    (v = 1 at angle 0 = image top). Photo texture if the baked file exists, else the flat material."""
    C = tubegeom.centerline(P, key)
    X, _ = tubegeom.surface(C, r, n_around)
    s = tubegeom.arclen(C)
    u = s / s[-1]
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    V = [[bm.verts.new(lib.mm(*X[k, i])) for i in range(X.shape[1])] for k in range(n_around + 1)]
    for k in range(n_around):
        for i in range(X.shape[1] - 1):
            f = bm.faces.new((V[k][i], V[k][i + 1], V[k + 1][i + 1], V[k + 1][i]))
            for loop, (kk, ii) in zip(f.loops, ((k, i), (k, i + 1), (k + 1, i + 1), (k + 1, i))):
                loop[uvl].uv = (u[ii], 1.0 - kk / n_around)
    for i in (0, X.shape[1] - 1):  # end caps
        c = bm.verts.new(lib.mm(*C[i]))
        for k in range(n_around):
            f = bm.faces.new((c, V[k][i], V[k + 1][i]) if i == 0 else (c, V[k + 1][i], V[k][i]))
            for loop in f.loops:
                loop[uvl].uv = (u[i], 0.5)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bm.normal_update()
    if tex is not None:
        f = lib.ROOT / "assets" / "textures" / tex
        if f.exists():
            mat = lib.mat_image(f"lower_tex_{Path(tex).stem}", f)
    ob = lib._obj_from_bm(name, bm, coll, mat)
    for poly in ob.data.polygons:
        poly.use_smooth = True
    return ob


def _smooth_tube(name, pts, r, coll, mat):
    """Cable: smooth Bezier through measured points (lib.tube_path)."""
    return lib.tube_path(name, [tuple(p) for p in pts], r, coll, mat, resolution=6, bevel_res=3)


def build(P: dict, coll) -> None:
    M = _mats(P)

    # control board, bracket, connector, plug, clips
    b = P["board"]
    _bx("lower.board", (b["x0"], b["x1"], b["y0"], b["y1"], b["z_top"] - b["t"], b["z_top"]), coll, M["board"])
    sc = P.get("board_screws")
    if sc:
        bm = bmesh.new()
        for x in sc["x"]:
            for y in sc["y"]:
                g = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=sc["r"] * lib.MM,
                                          radius2=sc["r"] * lib.MM, depth=sc["h"] * lib.MM)
                bmesh.ops.translate(bm, vec=Vector(lib.mm(x, y, b["z_top"] + sc["h"] / 2)), verts=g["verts"])
        lib._obj_from_bm("lower.board_screws", bm, coll, M["screw"])
    k = P["bracket"]
    _bx("lower.board_bracket", (k["x0"], k["x1"], k["y0"], k["y1"], k["z0"], k["z1"]), coll, M["metal"])
    _bx("lower.connector", P["connector"]["b"], coll, M["conn"])
    _bx("lower.plug", P["plug"]["b"], coll, M["white"])
    _bx("lower.clip_board", P["clip_board"]["b"], coll, M["white"])
    cb = P["clip_bar"]
    if cb.get("strap"):  # strap sleeve around braid A at y_c (len along the cable) + base block to the crossbar
        st = cb["strap"]
        C = tubegeom.centerline(P, "braid_a")
        s_ = tubegeom.arclen(C)
        sc = s_[int(abs(C[:, 1] - st["y_c"]).argmin())]
        seg = C[(s_ >= sc - st["len"] / 2) & (s_ <= sc + st["len"] / 2)]
        _tubes("lower.clip_bar", [[tuple(p) for p in seg]], st["r"], coll, M["white"])
        _bx("lower.clip_bar_base", st["base"], coll, M["white"])
    else:
        _bx("lower.clip_bar", cb.get("b") or [cb[k] for k in ("x0", "x1", "y0", "y1", "z0", "z1")], coll, M["white"])

    # fingers and OE frames
    F, O = P["fingers"], P["oe"]
    xs = [F["x_first"] + i * F["pitch"] for i in range(F["n"])]
    for i, xc in enumerate(xs):
        _multi_box(f"lower.finger_{i}", [(xc - F["w"] / 2, xc + F["w"] / 2, a, b, F["z_top"] - F["h"], F["z_top"])
                                         for a, b in F["blocks"]], coll, M["copper"])
        _bx(f"lower.oe_{i}", (xc - O["w"] / 2, xc + O["w"] / 2, O["y0"], O["y1"], O["z_top"] - O["h"],
                              O["z_top"]), coll, M["oe"])
    fs = F.get("screw")
    if fs:  # Wave 5: screw head at the center of the rear copper block of each finger
        bm = bmesh.new()
        for xc in xs:
            g = bmesh.ops.create_cone(bm, cap_ends=True, segments=16, radius1=fs["r"] * lib.MM, radius2=fs["r"] * lib.MM,
                                      depth=fs["h"] * lib.MM)
            bmesh.ops.translate(bm, vec=Vector(lib.mm(xc + fs.get("dx", 0.0), fs["y"], F["z_top"] + fs["h"] / 2 - fs["sink"])),
                                verts=g["verts"])
        lib._obj_from_bm("lower.finger_screws", bm, coll, M["screw_cu"])
    st = F["straps"]
    _multi_box("lower.finger_straps", [(xc - F["w"] / 2 - st["overhang"], xc + F["w"] / 2 + st["overhang"], a, b,
                                        F["z_top"] - st["drop"] - st["t"], F["z_top"] - st["drop"])
                                       for xc in xs for a, b in st["y"]], coll, M["silver"])
    B = P["base"]
    _bx("lower.oe_base", (B["x0"], B["x1"], B["y0"], B["y1"], B["z_top"] - B["t"], B["z_top"]), coll, M["base"])

    # copper loops: U-bends between neighbor fingers (legs included); long tubes to the manifold
    T = P["tubes"]
    z, ya, yb, dx = T["z"], T["y_leg0"], T["y_leg1"], T["leg_dx"]
    for gname, pairs in (("loops_l", [p for p in T["u_pairs"] if p[0] < 4]),
                         ("loops_r", [p for p in T["u_pairs"] if p[0] >= 4])):
        paths = []
        for i, j in pairs:
            xa, xb = xs[i] + dx, xs[j] - dx
            yc = yb + T["u_depth"]
            paths.append([(xa, ya, z), (xa, yb, z), (xa, yc, z), (xb, yc, z), (xb, yb, z), (xb, ya, z)])
        _tubes(f"lower.{gname}", paths, T["r"], coll, M["copper"], fillet=(xs[1] - xs[0] - 2 * dx) / 2 - 0.01)
    for lt in P["long_tube"]:
        _tubes(f"lower.{lt['name']}", [lt["pts"]], T["r"], coll, M["copper"], fillet=T["fillet"])
    Fi = P["fittings"]
    for i, x in enumerate(Fi["x"]):
        lib.cylinder(f"lower.fitting_{i}", Fi["r"], Fi["y1"] - Fi["y0"], (x, (Fi["y0"] + Fi["y1"]) / 2, Fi["z"]),
                     coll, axis="y", mat=M["metal"], verts=24)
    _bx("lower.manifold", P["manifold"]["b"], coll, M["metal"])

    # organizers, strips, fibers
    _bx("lower.org_l", P["org_l"]["b"], coll, M["org"])
    _bx("lower.org_r", P["org_r"]["b"], coll, M["org"])
    _bx("lower.strip_l", P["strip_l"]["b"], coll, M["strip"])
    _bx("lower.strip_r", P["strip_r"]["b"], coll, M["strip"])
    Fb = P["fibers"]
    for side, idx, org in (("l", range(0, 4), P["org_l"]["b"]), ("r", range(4, 8), P["org_r"]["b"])):
        ports = [xs[i] + s * Fb["port_dx"] for i in idx for s in (-1, 1)]
        ox0, ox1 = org[0] + 3, org[1] - 3
        ents = [ox0 + (ox1 - ox0) * k / (len(ports) - 1) for k in range(len(ports))]
        tr = Fb.get(f"trace_{side}")
        paths = []
        for k, (xp, xe) in enumerate(zip(ports, ents)):
            xm = xp + 0.45 * (xe - xp)
            if tr:  # traced per-fiber x: port, strip crossing, organizer entry
                xp, xm, xe = tr[k]
            paths.append([(xp, Fb["port_y"], Fb["z_port"]), (xp + 0.1 * (xe - xp), Fb["port_y"] + 10, Fb["z_floor"]),
                          (xm, Fb["strip_y"], Fb["z_floor"] + 1.5), (xe, Fb["org_y"] - 6, Fb["z_floor"]),
                          (xe, Fb["org_y"] + 2, Fb["z_org"])])
        if Fb.get("boot_len"):  # strain-relief boots: first boot_len mm of each fiber from the port
            boots = []
            for pth in paths:
                a0, a1 = Vector(pth[0]), Vector(pth[1])
                boots.append([tuple(a0), tuple(a0 + (a1 - a0).normalized() * Fb["boot_len"])])
            nr = Fb.get("boot_ribs", 0)
            if nr:  # Wave 5: ribbed boots: core tube (boot_r_core) + nr short rings of boot_r along the boot
                _tubes(f"lower.oe_boots_{side}", boots, Fb["boot_r_core"], coll, M["white"])
                rings = []
                for a0, a1 in boots:
                    a0, a1 = Vector(a0), Vector(a1)
                    for k in range(nr):
                        c = a0 + (a1 - a0) * ((k + 0.5) / nr)
                        h = (a1 - a0).normalized() * (0.5 * Fb["boot_len"] / nr * Fb.get("rib_frac", 0.55))
                        rings.append([tuple(c - h), tuple(c + h)])
                _tubes(f"lower.oe_boot_ribs_{side}", rings, Fb["boot_r"], coll, M["white"])
            else:
                _tubes(f"lower.oe_boots_{side}", boots, Fb["boot_r"], coll, M["white"])
        o = _tubes(f"lower.fibers_{side}", paths, Fb["r"], coll, M["fiber"], fillet=8.0)
        o.data.resolution_u = 1
        Xr = Fb.get("exit_ribbon")
        if Xr:  # Wave 4: flat level ribbon of n fibers from the organizer's rear face straight under the crossbar
            xc, w = Xr[f"xc_{side}"], Xr["w"]
            ex = [xc - w / 2 + w * k / (Xr["n"] - 1) for k in range(Xr["n"])]
            zz = Xr["z"]
            paths = [[(x, Xr["y0"], zz), (x, Xr["y1"], zz + Xr.get("dz_end", 0.0))] for x in ex]
            _tubes(f"lower.fibers_exit_{side}", paths, Xr["r"], coll, M["fiber"])
        else:
            ex = [ox0 + (ox1 - ox0) * k / (Fb["exit_n"] - 1) for k in range(Fb["exit_n"])]
            y0, y1, z0, z1 = Fb["exit_y0"], Fb["exit_y1"], Fb["exit_z0"], Fb["exit_z1"]
            paths = [[(x, y0 - 2, z0), (x, y0 + 0.4 * (y1 - y0), z0 - 3), (x, y1 - 3, z0 - 0.5 * (z0 - z1)),
                      (x, y1, z1)] for x in ex]
            _tubes(f"lower.fibers_exit_{side}", paths, Fb["exit_r"], coll, M["fiber"], fillet=6.0)

    # cables
    _tube_mesh("lower.braid_a", P, "braid_a", P["braid_a"]["r"], coll, M["braid"], tex=P["braid_a"].get("tex"))
    for nm in ("ribbon_a_front", "ribbon_a_rear"):
        R = P[nm]
        a, c, tip = Vector(R["line"][0]), Vector(R["line"][1]), Vector(R["tip"])
        paths = []
        for k in range(R["n"]):
            p0 = a + (c - a) * (k / (R["n"] - 1))
            mid = p0 + (tip - p0) * 0.5
            paths.append([tuple(p0), tuple(mid), tuple(tip)])
        _tubes(f"lower.{nm}", paths, R["r"], coll, M["ribbon"])
    # braid_b: fitted offsets on the visible front part are applied in tubegeom.centerline
    _tube_mesh("lower.braid_b", P, "braid_b", P["braid_b"]["r"], coll, M["braid_b"], tex=P["braid_b"].get("tex"))
    _smooth_tube("lower.cable_loop", P["cable_loop"]["pts"], P["cable_loop"]["r"], coll, M["cable_blk"])

    # left side: silver rod, metal plate, small SMD board
    rl = P["rod_l"]
    _tubes("lower.rod_l", [[(x + rl.get("dx", 0.0), y, z + rl.get("dz", 0.0)) for x, y, z in rl["pts"]]], rl["r"], coll,
           M["silver"], fillet=10.0)
    _bx("lower.plate_l", P["plate_l"]["b"], coll, M["metal"])
    _bx("lower.smd_board_l", P["smd_board_l"]["b"], coll, M["board"])

    _tex_surfaces(P, coll, M, xs)

    # tray frame: the rigid front-bay assembly is built level and moved with lib.apply_frame(objs, "tray")
    # (config/frames.json: world z = tray z + 0.0148 x, rotation about world Y through the origin). A residual
    # pitch dzdy about y0 (board plane fit in the tray frame, see devlog 5d) is applied first as a shear in the
    # level frame. Cables placed from world measurements ([tilt].skip) are not moved.
    t = P.get("tilt", {})
    skip = set(t.get("skip", []))
    rigid = [ob for ob in coll.objects if ob.name.startswith("lower.") and ob.name[6:] not in skip]
    if t.get("dzdy", 0.0):
        bb, y0 = t["dzdy"], t["y0"] * lib.MM
        S = Matrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, bb, 1, -bb * y0), (0, 0, 0, 1)))
        for ob in rigid:
            ob.data.transform(S)
    lib.apply_frame(rigid, "tray")
