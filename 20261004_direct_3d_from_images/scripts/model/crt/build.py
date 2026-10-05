"""crt group: CRT tube assembly lying along X in the electronics tray (yoke, label, holder, clamp, neck, socket)."""

import math

import bmesh
import bpy
import lib
from mathutils import Matrix, Vector


# ---------- geometry helpers (mm in, meters out) ----------

def _loft(name, rings, ay, az, coll, mat, seg=48):
    """Closed loft along X through elliptical rings [(x, ry, rz), ...] centered on the axis (ay, az)."""
    profiles = [(x, [(ry * math.cos(t), rz * math.sin(t)) for t in (2 * math.pi * i / seg for i in range(seg))])
                for x, ry, rz in rings]
    return _prism_loft(name, profiles, ay, az, coll, mat)


def _prism_loft(name, profiles, ay, az, coll, mat):
    """Closed loft along X through [(x, [(dy, dz), ...]), ...] (same vertex count per profile)."""
    bm = bmesh.new()
    loops = [[bm.verts.new(lib.mm(x, ay + dy, az + dz)) for dy, dz in pts] for x, pts in profiles]
    n = len(loops[0])
    uvl = bm.loops.layers.uv.new("UVMap")
    x0, x1 = profiles[0][0], profiles[-1][0]
    ang = [((math.atan2(dz, dy) + math.pi / 2) % (2 * math.pi)) / (2 * math.pi) for dy, dz in profiles[0][1]]
    for k, (a, b) in enumerate(zip(loops[:-1], loops[1:])):
        va, vb = 1 - (profiles[k][0] - x0) / (x1 - x0), 1 - (profiles[k + 1][0] - x0) / (x1 - x0)
        for i in range(n):
            j = (i + 1) % n
            f = bm.faces.new((a[i], a[j], b[j], b[i]))
            ui, uj = ang[i], ang[j] if ang[j] >= ang[i] - 0.5 else ang[j] + 1  # cylinder UVs (seam at bottom)
            for lp, uv in zip(f.loops, ((ui, va), (uj, va), (uj, vb), (ui, vb))):
                lp[uvl].uv = uv
    bm.faces.new(list(reversed(loops[0])))
    bm.faces.new(loops[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._obj_from_bm(name, bm, coll, mat)


def _uv_loft(name, rings, ay, az, coll, mat, cap_mat, seg=96, t0=-math.pi / 2):
    """Loft along X through elliptical rings with UVs matching texture_yoke.py: u = (t - t0) / 2pi (seam at the
    bottom), v = 1 at the first ring (-X end). Side faces smooth; end caps use separate vertices and cap_mat."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    x0, x1 = rings[0][0], rings[-1][0]
    loops = []
    for x, ry, rz in rings:
        loops.append([bm.verts.new(lib.mm(x, ay + ry * math.cos(t0 + 2 * math.pi * i / seg),
                                          az + rz * math.sin(t0 + 2 * math.pi * i / seg))) for i in range(seg + 1)])
    vs = [1 - (r[0] - x0) / (x1 - x0) for r in rings]
    for k in range(len(rings) - 1):
        a_, b_ = loops[k], loops[k + 1]
        for i in range(seg):
            f = bm.faces.new((a_[i], b_[i], b_[i + 1], a_[i + 1]))
            f.smooth = True
            for lp, (u, v) in zip(f.loops, ((i / seg, vs[k]), (i / seg, vs[k + 1]), ((i + 1) / seg, vs[k + 1]),
                                            ((i + 1) / seg, vs[k]))):
                lp[uv_layer].uv = (u, v)
    for k in (0, len(rings) - 1):
        x, ry, rz = rings[k]
        cv = [bm.verts.new(lib.mm(x, ay + ry * math.cos(t0 + 2 * math.pi * i / seg),
                                  az + rz * math.sin(t0 + 2 * math.pi * i / seg))) for i in range(seg)]
        f = bm.faces.new(cv if k else list(reversed(cv)))
        f.material_index = 1
    bm.normal_update()
    ob = lib._obj_from_bm(name, bm, coll, mat)
    ob.data.materials.append(cap_mat)
    return ob


def _xcyl(name, x0, x1, r, ay, az, coll, mat, verts=48):
    return lib.cylinder(name, r, x1 - x0, ((x0 + x1) / 2, ay, az), coll, axis="x", mat=mat, verts=verts)


def _ring_at(rings, x):
    """Linear interpolation of (ry, rz) along the yoke loft."""
    for (xa, ya, za), (xb, yb, zb) in zip(rings[:-1], rings[1:]):
        if xa <= x <= xb:
            t = (x - xa) / (xb - xa)
            return ya + t * (yb - ya), za + t * (zb - za)
    return rings[-1][1], rings[-1][2]


def _label_patch(name, rings, L, ay, az, coll, mat, nx=8, nt=16):
    """Label as a curved patch on the yoke ellipse. UV: u along +Y (y0 -> y1), v = 1 at x0 (-X, top of print)."""
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    grid = []
    for i in range(nx + 1):
        x = L["x0"] + (L["x1"] - L["x0"]) * i / nx
        ry, rz = _ring_at(rings, x)
        ry, rz = ry + L["offset"], rz + L["offset"]
        t0 = math.acos(max(-1, min(1, (L["y0"] - ay) / ry)))
        t1 = math.acos(max(-1, min(1, (L["y1"] - ay) / ry)))
        grid.append([bm.verts.new(lib.mm(x, ay + ry * math.cos(t), az + rz * math.sin(t)))
                     for t in (t0 + (t1 - t0) * k / nt for k in range(nt + 1))])
    for i in range(nx):
        for k in range(nt):
            f = bm.faces.new((grid[i][k], grid[i][k + 1], grid[i + 1][k + 1], grid[i + 1][k]))
            for lp, (a, b) in zip(f.loops, ((i, k), (i, k + 1), (i + 1, k + 1), (i + 1, k))):
                lp[uv_layer].uv = (b / nt, 1 - a / nx)
    bm.normal_update()
    if sum(f.normal.z for f in bm.faces) < 0:
        for f in bm.faces:
            f.normal_flip()
    return lib._obj_from_bm(name, bm, coll, mat)


def _radial_box(name, size, center, ay, az, coll, mat):
    """Box (along x, radial, tangential) placed at center (mm), oriented radially from the axis."""
    sx, sr, st = size
    ob = lib.box(name, (sx, st, sr), (0, 0, 0), coll, mat, bevel_mm=0.4)
    ang = math.degrees(math.atan2(center[1] - ay, center[2] - az))  # angle from +Z toward +Y
    lib.place(ob, center, (-ang, 0, 0))
    return ob


# ---------- materials ----------

# Base colors: linear medians of each part's pixels in IMG_1560 (top view, s4) under the world-only rig
# (crt_010), DevLog-002 "Colors". Specular IOR level low (0.25) so the white world adds no veil.
def _principled(name, color, roughness=0.5, metallic=0.0, spec=0.25, alpha=1.0):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m = lib.mat_pbr(name, color, roughness=roughness, metallic=metallic, alpha=alpha)
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Specular IOR Level"].default_value = spec
    if alpha < 1.0:
        m.surface_render_method = "BLENDED"
    return m


def _srgb(c):
    """sRGB 0-255 triple -> linear."""
    return tuple(((v / 255 + 0.055) / 1.055) ** 2.4 if v / 255 > 0.04045 else v / 255 / 12.92 for v in c)


def build(P: dict, coll) -> None:
    ay, az = P["axis"]["y"], P["axis"]["z"]
    C = P["colors"]
    tape_cap = _principled("crt_tape_plain", _srgb(C["tape"]), roughness=0.4)
    copper_plain = _principled("crt_copper_plain", _srgb(C["copper"]), roughness=0.4)
    yellow = lib.mat_image("crt_tape_tex", P["yoke"]["texture"], roughness=0.4)
    copper = lib.mat_image("crt_copper_tex", P["copper"]["texture"], roughness=0.4)
    white = _principled("crt_white_plastic", _srgb(C["white"]))
    black = _principled("crt_black_plastic", _srgb(C["dark_ring"]))
    metal = _principled("crt_metal", _srgb(C["clamp"]), roughness=0.35, spec=0.5)
    screw = _principled("crt_screw", _srgb(C["screw"]), roughness=0.35, spec=0.5)
    cream = _principled("crt_cream", _srgb(C["cream"]), roughness=0.8)
    glass = _principled("crt_glass", _srgb(C["glass"]), roughness=0.05, spec=0.5, alpha=P["neck"].get("alpha", 0.6))
    gun = _principled("crt_gun_metal", _srgb(C["gun"]), roughness=0.35)
    blue = _principled("crt_light_blue", _srgb(C["cap"]))
    blue_rim = _principled("crt_light_blue_rim", _srgb(C["cap_rim"]))
    board = _principled("crt_board", _srgb(C["board"]))
    L = P["label"]
    label = lib.mat_image("crt_label_tex", L["texture"], roughness=0.5)

    Y = P["yoke"]
    sy, sz = Y.get("scale_y", 1.0), Y.get("scale_z", 1.0)
    ks = [Y.get(f"s{i}", 1.0) for i in range(len(Y["rings"]))]  # per-ring scale knobs (fit)
    rings = [(x, ry * sy * k, rz * sz * k) for (x, ry, rz), k in zip(Y["rings"], ks)]
    yay, yaz = ay + Y.get("dy", 0.0), az + Y.get("dz", 0.0)
    _uv_loft("crt.yoke", rings, yay, yaz, coll, yellow, tape_cap)
    c = P["copper"]
    _uv_loft("crt.yoke_copper", [(c["x0"], c["ry"] * sy, c["rz"] * sz), (c["x1"], c["ry"] * sy, c["rz"] * sz)],
             yay, yaz, coll, copper, copper_plain)
    _label_patch("crt.label_yoke", rings, L, yay, yaz, coll, label)

    for key, name, mat in (("holder", "crt.yoke_holder", white), ("clamp", "crt.neck_clamp", metal),
                           ("white_ring", "crt.neck_white_ring", white), ("cream_ring", "crt.neck_ring", cream),
                           ("neck", "crt.neck", glass), ("cap", "crt.socket_cap", blue)):
        q = P[key]
        cy, cz = ay + q.get("dy", 0.0), az + q.get("dz", 0.0)
        tex = lib.ROOT / "assets" / "textures" / f"crt_{key}.png"  # from scripts/model/crt/texture_yoke.py
        if P["textures"].get("round_parts", True) and tex.exists():
            tm = lib.mat_image(f"crt_{key}_tex", tex, roughness=0.5)
            _uv_loft(name, [(q["x0"], q["r"], q["r"]), (q["x1"], q["r"], q["r"])], cy, cz, coll, tm, mat, seg=64)
        else:
            _xcyl(name, q["x0"], q["x1"], q["r"], cy, cz, coll, mat)

    cp = P["cap"]
    _xcyl("crt.socket_cap_flange", cp["flange_x0"], cp["x0"], cp["flange_r"], ay, az, coll, blue_rim)
    _xcyl("crt.socket_cap_step", cp["x1"], cp["step_x1"], cp["step_r"], ay, az, coll, blue)

    # notched dark ring: gear profile extruded along x
    g = P["black_ring"]
    n = int(g["teeth"])
    prof = []
    for i in range(2 * n):
        t = math.pi * i / n
        r = g["r"] if i % 2 == 0 else g["r_root"]
        for dt in (-0.25, 0.25):
            tt = t + dt * math.pi / n
            prof.append((r * math.cos(tt), r * math.sin(tt)))
    ring_tex = lib.ROOT / "assets" / "textures" / "crt_dark_ring.png"
    ring_mat = lib.mat_image("crt_dark_ring_tex", ring_tex, roughness=0.5) if ring_tex.exists() else black
    _prism_loft("crt.yoke_ring", [(g["x0"], prof), (g["x1"], prof)], ay, az, coll, ring_mat)

    T = P["tabs"]
    for i, ctr in enumerate(T["centers"]):
        _radial_box(f"crt.yoke_tab_{i}", T["size"], ctr, ay, az, coll, white)

    e = P["clamp_ear"]
    lib.box("crt.clamp_ear", (e["x1"] - e["x0"], e["y1"] - e["y0"], e["z1"] - e["z0"]),
            ((e["x0"] + e["x1"]) / 2, (e["y0"] + e["y1"]) / 2, (e["z0"] + e["z1"]) / 2), coll, metal)
    lib.cylinder("crt.clamp_screw", e["screw_r"], e["screw_h"],
                 ((e["x0"] + e["x1"]) / 2, (e["y0"] + e["y1"]) / 2, e["z1"] + e["screw_h"] / 2), coll,
                 axis="z", mat=screw, verts=24)

    G = P["gun"]
    _xcyl("crt.neck_gun", G["x0"], G["x1"], G["r"], ay, az, coll, gun, verts=24)
    bm = bmesh.new()
    for i in range(int(G["pins"])):
        t = 2 * math.pi * (i + 0.5) / G["pins"]
        M = Matrix.Translation(Vector(lib.mm((G["pin_x0"] + G["pin_x1"]) / 2, ay + G["pin_ring"] * math.cos(t),
                                             az + G["pin_ring"] * math.sin(t)))) @ Matrix.Rotation(math.pi / 2, 4, "Y")
        bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=lib.mm(G["pin_r"]), radius2=lib.mm(G["pin_r"]),
                              depth=lib.mm(G["pin_x1"] - G["pin_x0"]), matrix=M)
    lib._obj_from_bm("crt.neck_pins", bm, coll, gun)

    b = P["board"]
    lib.box("crt.socket_board", (b["x1"] - b["x0"], b["y1"] - b["y0"], b["z1"] - b["z0"]),
            ((b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2, (b["z0"] + b["z1"]) / 2), coll, board)
    board_tex = lib.ROOT / "assets" / "textures" / "crt_board.png"
    if board_tex.exists():  # +X face photo texture (TL, TR, BR, BL seen from +X; right = +Y)
        lib.label_quad("crt.socket_board_face", [(b["x1"], b["y0"], b["z1"]), (b["x1"], b["y1"], b["z1"]),
                                                 (b["x1"], b["y1"], b["z0"]), (b["x1"], b["y0"], b["z0"])],
                       coll, lib.mat_image("crt_board_tex", board_tex, roughness=0.6), offset_mm=0.05)
