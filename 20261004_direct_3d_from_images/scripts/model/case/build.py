"""Case group: black plastic housing with a taller screen section (-X) and a lower electronics tray (+X),
long-wall ramps at the junction, a clear window insert with flange clips and a white card in the screen section,
and a black bracket bar with screw pads, two screws and two holes at the junction.
All dimensions from config/model/case.toml (case-local mm, placed by [frame])."""

import math
import sys
from pathlib import Path

import bmesh
import lib
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling helper (resolves to the LKG copy when used)
import case_panels  # noqa: E402

TEX_DIR = lib.ROOT / "assets" / "textures"


def _tex_mat(name, P):
    """Photo-texture material for a panel, or None if the texture has not been baked yet."""
    path = TEX_DIR / f"case_{name}.png"
    if not path.exists():
        return None
    m = lib.mat_image(f"case_tex_{name}", path, roughness=P["material"]["tex_rough"])
    m.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = P["material"]["tex_spec"]
    for nd in m.node_tree.nodes:
        if nd.type == "TEX_IMAGE":
            nd.extension = "EXTEND"
    return m


def _apply_panels(objs: dict, P: dict) -> None:
    """Assign panel image materials and planar-projection UVs to the faces of each panel's object (case-local
    coordinates, before placement). A face belongs to a panel if its normal is within ~35 deg of the panel normal,
    its centroid lies within tol mm of the panel plane and inside the quad."""
    tol = P["material"]["panel_tol"]
    for pn in case_panels.panels(P):
        ob = objs.get(pn["obj"])
        mat = _tex_mat(pn["name"], P)
        if ob is None or mat is None:
            continue
        c = [Vector(lib.mm(*q)) for q in pn["c"]]
        eu, ev = c[1] - c[0], c[3] - c[0]
        nrm = Vector(pn["n"]).normalized()
        n_pl = eu.cross(ev).normalized()
        me = ob.data
        if mat.name not in [m.name for m in me.materials if m]:
            me.materials.append(mat)
        mi = [m.name if m else "" for m in me.materials].index(mat.name)
        bm = bmesh.new()
        bm.from_mesh(me)
        uvl = bm.loops.layers.uv.get("UVMap") or bm.loops.layers.uv.new("UVMap")
        for fa in bm.faces:
            if fa.normal.dot(nrm.normalized()) < 0.8:
                continue
            cen = fa.calc_center_median()
            if abs((cen - c[0]).dot(n_pl)) > tol * lib.MM:
                continue
            u = (cen - c[0]).dot(eu) / eu.length_squared
            v = (cen - c[0]).dot(ev) / ev.length_squared
            if not (-0.02 <= u <= 1.02 and -0.02 <= v <= 1.02):
                continue
            fa.material_index = mi
            for lp in fa.loops:
                d = lp.vert.co - c[0]
                lp[uvl].uv = (d.dot(eu) / eu.length_squared, 1.0 - d.dot(ev) / ev.length_squared)
        bm.to_mesh(me)
        bm.free()


def _rounded_rect(x0, x1, y0, y1, radii, seg=6):
    """2D outline (CCW) with per-corner radii in order (-x-y, +x-y, +x+y, -x+y)."""
    corners = [((x0, y0), (1, 1), 180), ((x1, y0), (-1, 1), 270), ((x1, y1), (-1, -1), 0), ((x0, y1), (1, -1), 90)]
    pts = []
    for ((cx, cy), (sx, sy), a0), r in zip(corners, radii):
        if r <= 0:
            pts.append((cx, cy))
            continue
        ox, oy = cx + sx * r, cy + sy * r
        for k in range(seg + 1):
            a = math.radians(a0 + 90 * k / seg)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    return pts


def _prism(name, outline, z0, z1, coll, mat=None, outline_top=None):
    """Prism (or loft when outline_top is given, same vertex count) between z0 and z1."""
    bm = bmesh.new()
    lo = [bm.verts.new(lib.mm(x, y, z0)) for x, y in outline]
    hi = [bm.verts.new(lib.mm(x, y, z1)) for x, y in (outline_top or outline)]
    n = len(outline)
    bm.faces.new(lo[::-1])
    bm.faces.new(hi)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((lo[i], lo[j], hi[j], hi[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._obj_from_bm(name, bm, coll, mat)


def _xz_prism(name, pts_xz, y0, y1, coll, mat=None):
    """Prism from a polygon in the XZ plane, extruded along Y from y0 to y1."""
    bm = bmesh.new()
    a = [bm.verts.new(lib.mm(x, y0, z)) for x, z in pts_xz]
    b = [bm.verts.new(lib.mm(x, y1, z)) for x, z in pts_xz]
    n = len(pts_xz)
    bm.faces.new(a)
    bm.faces.new(b[::-1])
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((a[i], b[i], b[j], a[j]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._obj_from_bm(name, bm, coll, mat)


def _cut(target, cutter_ob):
    lib.boolean(target, cutter_ob)


def _window(name, W, coll, mat):
    """Open-top clear tray: top ring at z_top, inset bottom ring with z varying linearly along X, flange ring at
    the top; long walls split into segments (case_panels.window_rings); shell via solidify."""
    top, bot, flg = case_panels.window_rings(W)
    bm = bmesh.new()
    vt = [bm.verts.new(lib.mm(*p)) for p in top]
    vb = [bm.verts.new(lib.mm(*p)) for p in bot]
    vf = [bm.verts.new(lib.mm(*p)) for p in flg]
    m = len(top)
    for i in range(m):
        j = (i + 1) % m
        bm.faces.new((vt[i], vt[j], vb[j], vb[i]))
        bm.faces.new((vf[i], vf[j], vt[j], vt[i]))
    bm.faces.new(vb[::-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = lib._obj_from_bm(name, bm, coll, mat)
    md = ob.modifiers.new("solid", "SOLIDIFY")
    md.thickness = W["thick"] * lib.MM
    md.offset = -1.0
    lib.apply_modifiers(ob)
    return ob


def build(P: dict, coll) -> None:
    b, s, t, f, rp = P["body"], P["screen"], P["tray"], P["frame"], P["ramp"]
    br, W, C, E, K = P["bracket"], P["window"], P["card"], P["window_edge"], P["clips"]
    # glossy black plastic: the eval world is uniform white, so full specular renders it medium gray; a low
    # specular level (coordinator 2026-10-05) keeps it near black
    black = lib.mat_pbr("case_black", tuple(P["material"]["black_rgb"]), roughness=P["material"]["black_rough"])
    black.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = P["material"]["black_spec"]
    # EEVEE Next without scene raytracing refracts only the world probe (opaque), so the clear insert is
    # alpha-blended instead of transmissive
    G = P["material"]
    glass = lib.mat_pbr("case_glass", tuple(G["glass_rgb"]), roughness=G["glass_rough"], alpha=G["glass_alpha"])
    glass.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = G["glass_spec"]
    if hasattr(glass, "surface_render_method"):
        glass.surface_render_method = "BLENDED"
    white = lib.mat_pbr("case_card_white", (0.80, 0.80, 0.78), roughness=0.7)
    bar_black = lib.mat_pbr("case_bracket_black", (0.015, 0.015, 0.015), roughness=0.3)
    clip_black = lib.mat_pbr("case_clip_black", (0.02, 0.02, 0.02), roughness=0.4)
    steel = lib.mat_pbr("case_screw_steel", (0.45, 0.45, 0.46), roughness=0.35, metallic=1.0)
    hole_mat = lib.mat_pbr("case_hole_dark", (0.0, 0.0, 0.0), roughness=0.9)

    L, Wd, w, fl, r = b["length"], b["width"], b["wall"], b["floor"], b["corner_r"]
    hw = Wd / 2
    x0, x2 = -L / 2, L / 2
    xj = x0 + s["length"]  # end of the full-height long walls (local)
    obs = []

    # screen section: rounded -X corners, square at the junction, closed junction wall
    H = s["height"]
    fs = b["flare"] / b["flare_ref_z"]  # outward flare of the outer walls per mm of height (molding draft)
    fH = fs * H
    scr = _prism("case.screen_box", _rounded_rect(x0, xj, -hw, hw, (r, 0, 0, r)), 0, H, coll, black,
                 _rounded_rect(x0 - fH, xj, -hw - fH, hw + fH, (r + fH, 0, 0, r + fH)))
    _cut(scr, _prism("_cut", _rounded_rect(x0 + w, xj - w, -hw + w, hw - w, (max(r - w, 0.3), 0, 0, max(r - w, 0.3))),
                     fl, H + 1, coll))
    cu = P["cuts"]
    (sy0, sy1), (sz0, sz1) = cu["slot_y"], cu["slot_z"]
    _cut(scr, lib.box("_cut", (w + 2, sy1 - sy0, sz1 - sz0), (x0 + w / 2, (sy0 + sy1) / 2, (sz0 + sz1) / 2), coll))
    nx0, nx1 = cu["notch_x"]
    _cut(scr, lib.box("_cut", (nx1 - nx0, w + 2, cu["notch_h"] + 1),
                      ((nx0 + nx1) / 2, -hw + w / 2, (cu["notch_h"] - 1) / 2), coll))
    obs.append(scr)

    # tray: rounded +X corners, open toward the screen section; near long wall lower than far and end walls
    ht = max(t["far_height"], t["end_height"])
    fT = fs * ht
    tray = _prism("case.tray", _rounded_rect(xj, x2, -hw, hw, (0, r, r, 0)), 0, ht, coll, black,
                  _rounded_rect(xj, x2 + fT, -hw - fT, hw + fT, (0, r + fT, r + fT, 0)))
    _cut(tray, _prism("_cut", _rounded_rect(xj - 1, x2 - w, -hw + w, hw - w, (0, max(r - w, 0.3), max(r - w, 0.3), 0)),
                      fl, ht + 1, coll))
    _cut(tray, lib.box("_cut", (x2 - xj + 2 + 2 * fT, w + 2 + 2 * fT, ht), ((xj + x2) / 2, -hw + w / 2 - 0.5 - fT,
                                                          t["near_height"] + ht / 2), coll))
    if t["end_height"] < ht:
        _cut(tray, lib.box("_cut", (w + 2 + 2 * fT, Wd + 2 + 2 * fT, ht), (x2 - w / 2 + 0.5 + fT, 0,
                                                                      t["end_height"] + ht / 2), coll))
    if t["far_height"] < ht:
        _cut(tray, lib.box("_cut", (x2 - xj + 2 + 2 * fT, w + 2 + 2 * fT, ht), ((xj + x2) / 2, hw - w / 2 + 0.5 + fT,
                                                              t["far_height"] + ht / 2), coll))
    obs.append(tray)

    # junction ramps on both long walls (screen rim down to each tray rim)
    for k, sy, hz in (("near", -1, t["near_height"]), ("far", 1, t["far_height"])):
        fo = fs * (hz + H) / 2  # follow the flared outer face at mid ramp height
        ya, yb = sorted((sy * (hw + fo), sy * (hw - w)))
        obs.append(_xz_prism(f"case.ramp_{k}", [(xj, hz - 1), (rp["x1"], hz - 1), (rp["x1"], hz), (xj, H)],
                             ya, yb, coll, black))

    lg = P["lugs"]
    for k, sy in (("near", -1), ("far", 1)):
        y_in = hw - 0.5
        obs.append(lib.box(f"case.lug_{k}", (lg["x"][1] - lg["x"][0], lg["y_out"] - y_in, lg["z"][1] - lg["z"][0]),
                           ((lg["x"][0] + lg["x"][1]) / 2, sy * (lg["y_out"] + y_in) / 2, (lg["z"][0] + lg["z"][1]) / 2),
                           coll, black, bevel_mm=0.5))

    # bracket bar with screw pads, screws and holes
    bt, bw = br["z_top"], br["width"]
    bar = lib.box("case.bracket", (bw, 2 * br["tab_y0"], br["thick"]), (br["x"], 0, bt - br["thick"] / 2),
                  coll, bar_black, bevel_mm=0.6)
    obs.append(bar)
    for hy in br["hole_y"]:
        obs.append(lib.cylinder(f"case.bracket_hole_{'n' if hy < 0 else 'f'}{abs(round(hy))}", br["hole_r"], 0.2,
                                (br["hole_x"], hy, bt + 0.05), coll, "z", hole_mat, verts=16))
    for k, sy in (("near", -1), ("far", 1)):
        ty = hw - br["tab_y0"]
        obs.append(lib.box(f"case.bracket_tab_{k}", (bw, ty, br["thick"]),
                           (br["x"], sy * (br["tab_y0"] + ty / 2), br["tab_z_top"] - br["thick"] / 2),
                           coll, bar_black, bevel_mm=0.5))
        obs.append(lib.cylinder(f"case.screw_{k}", br["screw_r"], br["screw_h"],
                                (br["screw_x"], sy * br["screw_y"], br["tab_z_top"] + br["screw_h"] / 2), coll, "z",
                                steel, verts=24))

    # window insert, its dark +X edge band, flange clips, card
    obs.append(_window("case.window", W, coll, glass))
    obs.append(lib.box("case.window_edge_px", (E["x1"] - E["x0"], 2 * E["half_w"], E["thick"]),
                       ((E["x0"] + E["x1"]) / 2, 0, E["z_top"] - E["thick"] / 2), coll, black, bevel_mm=0.5))
    zc = K["z0"] + K["height"] / 2
    for k, sy in (("near", -1), ("far", 1)):
        obs.append(lib.box(f"case.clip_end_{k}", (K["width"], K["length"], K["height"]), (K["end_x"], sy * K["end_y"], zc),
                           coll, clip_black, bevel_mm=0.3))
        if sy > 0:
            obs.append(lib.box(f"case.clip_side_{k}", (K["length"], K["width"], K["height"]),
                               (K["side_x"], sy * K["side_y"], zc), coll, clip_black, bevel_mm=0.3))
    R = P["rod"]
    obs.append(lib.cylinder("case.window_rod", R["r"], 2 * R["half_len"], (R["x"], 0, R["z"]), coll, "y", glass, verts=12))
    dx, dz = C["x1"] - C["x0"], C["z_px"] - C["z_mx"]
    ang = math.degrees(math.atan2(-dz, dx))  # +rot about Y lowers +X
    card = lib.box("case.card", (math.hypot(dx, dz), C["y1"] - C["y0"], C["thick"]), (0, 0, 0), coll, white)
    lib.place(card, ((C["x0"] + C["x1"]) / 2, (C["y0"] + C["y1"]) / 2, (C["z_mx"] + C["z_px"]) / 2 - C["thick"] / 2),
              (0, ang, 0))
    obs.append(card)

    # window appearance quad (photo texture of glass + card seen from above); clear glass until it is baked
    wq = [p for p in case_panels.panels(P) if p["name"] == "window_top"]
    if wq:
        obs.append(lib.label_quad("case.window_top", wq[0]["c"], coll, _tex_mat("window_top", P) or glass, offset_mm=0.0))

    _apply_panels({ob.name: ob for ob in obs}, P)
    for ob in obs:
        lib.place(ob, (f["cx"], f["cy"], 0), (0, 0, f["yaw_deg"]))
