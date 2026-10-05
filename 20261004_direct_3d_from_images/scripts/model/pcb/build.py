"""pcb group: main board in the electronics tray (+X) and the parts on it (flyback, heat-sink plate, caps, chokes,
trimmers, connectors, regulators). All dimensions from config/model/pcb.toml (mm, world frame)."""

import bmesh
import bpy
import lib
from mathutils import Vector


def _mats(P):
    metal_keys = set(P.get("metal_keys", {}).get("keys", []))
    spec = P.get("metal_keys", {}).get("dielectric_specular", 0.25)
    out = {}
    for k, v in P["mats"].items():
        m = lib.mat_pbr(f"pcb_{k}", tuple(v[:3]), roughness=v[3], metallic=1.0 if k in metal_keys else 0.0)
        if k not in metal_keys:
            m.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = spec
        out[k] = m
    return out


def _bbox(name, x0, x1, y0, y1, z0, z1, coll, mat, bevel=0.0):
    return lib.box(name, (x1 - x0, y1 - y0, z1 - z0), ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), coll, mat,
                   bevel_mm=bevel)


def _join(name, objs, coll):
    """Merge mesh objects into one object (one part for eval), keeping per-face materials."""
    bm = bmesh.new()
    mats = []
    for o in objs:
        me = o.data
        idx = []
        for m in me.materials:
            if m not in mats:
                mats.append(m)
            idx.append(mats.index(m))
        tmp = bmesh.new()
        tmp.from_mesh(me)
        for f in tmp.faces:
            f.material_index = idx[f.material_index] if idx else 0
        me2 = bpy.data.meshes.new("_tmp")
        tmp.to_mesh(me2)
        tmp.free()
        bm.from_mesh(me2)
        bpy.data.meshes.remove(me2)
        bpy.data.objects.remove(o, do_unlink=True)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    coll.objects.link(ob)
    return ob


def _prism_xz(name, pts_xz, y0, y1, coll, mat):
    """Extrude a convex polygon in the X-Z plane (mm) between y0 and y1."""
    bm = bmesh.new()
    a = [bm.verts.new(Vector(lib.mm(x, y0, z))) for x, z in pts_xz]
    b = [bm.verts.new(Vector(lib.mm(x, y1, z))) for x, z in pts_xz]
    bm.faces.new(a)
    bm.faces.new(list(reversed(b)))
    n = len(pts_xz)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new([a[i], b[i], b[j], a[j]])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return lib._obj_from_bm(name, bm, coll, mat)


def _tex(P, key, name):
    return lib.mat_image(f"pcb_tex_{name}", P["textures"][key])


def build(P: dict, coll) -> None:
    M = _mats(P)
    b = P["board"]
    zb = b["top"]
    _bbox("pcb.board", b["x0"], b["x1"], b["y0"], b["y1"], zb - b["thick"], zb, coll, M["board"])
    if "board_top" in P.get("textures", {}):
        lib.label_quad("pcb.label_board_top", [[b["x0"], b["y1"], zb], [b["x1"], b["y1"], zb],
                                               [b["x1"], b["y0"], zb], [b["x0"], b["y0"], zb]],
                       coll, _tex(P, "board_top", "board_top"), offset_mm=0.03)

    f = P["flyback"]
    _bbox("pcb.flyback_base", f["x0"], f["x1"], f["y0"], f["y1"], zb, zb + f["base_h"], coll, M["black"])
    e = f.get("end_t", 0.0)  # black end faces (+-X): IMG_1613 crop shows the -X face dark
    z0 = zb + f["base_h"]
    fb = [_bbox("_fb", f["x0"] + e, f["x1"] - e, f["y0"], f["y1"], z0, f["top"], coll, M[f.get("body_mat", "beige")],
                bevel=0.5)]
    if e > 0:
        fb += [_bbox("_fe", f["x0"], f["x0"] + e, f["y0"] + 0.3, f["y1"] - 0.3, z0, f["top"] - 0.3, coll, M["black"]),
               _bbox("_fe", f["x1"] - e, f["x1"], f["y0"] + 0.3, f["y1"] - 0.3, z0, f["top"] - 0.3, coll, M["black"])]
    _join("pcb.flyback", fb, coll)
    top_mat = _tex(P, "flyback_top", "flyback_top") if "flyback_top" in P.get("textures", {}) else M["beige"]
    lib.label_quad("pcb.label_flyback_warning", P["flyback_label"]["corners"], coll, top_mat, offset_mm=0.1)
    if "flyback_side" in P.get("textures", {}):
        lib.label_quad("pcb.label_flyback_side", P["flyback_side_label"]["corners"], coll,
                       _tex(P, "flyback_side", "flyback_side"), offset_mm=0.1)

    p = P["plate"]
    cdx, cdz = p.get("chamfer_dx", 0.0), p.get("chamfer_dz", 0.0)
    pts = [(p["x0"], zb), (p["x1"], zb), (p["x1"], p["top"]), (p["x0"] + cdx, p["top"]), (p["x0"], p["top"] - cdz)]
    plate = _prism_xz("_plate", pts, p["y0"], p["y1"], coll, M["metal"])
    if "flange_y0" in p:
        fl = _bbox("_flange", p["x0"] + cdx, p["x1"], p["flange_y0"], p["y0"], p["top"] - p["flange_t"], p["top"], coll,
                   M["metal"])
        _join("pcb.heatsink_plate", [plate, fl], coll)
    else:
        plate.name = "pcb.heatsink_plate"

    F = P.get("fit", {})

    def off(name):
        return F.get(f"{name}_dx", 0.0), F.get(f"{name}_dy", 0.0)

    for blk in P.get("block", []):
        x0, x1, y0, y1 = blk["xy"]
        dx, dy = off(blk["name"])
        x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
        _bbox(f"pcb.{blk['name']}", x0, x1, y0, y1, zb, blk["top"], coll, M[blk["mat"]], bevel=0.3)

    for sl in P.get("slab", []):
        ob = lib.box(f"pcb.{sl['name']}", tuple(sl["size"]), (0, 0, 0), coll, M[sl["mat"]])
        lib.place(ob, tuple(sl["center"]), (sl.get("rot_x_deg", 0.0), 0, sl.get("rot_z_deg", 0.0)))

    for i, lg in enumerate(P.get("legs", [])):
        x0, x1 = lg["x"]
        parts = [_bbox("_leg", x0, x1, y - lg["w"] / 2, y + lg["w"] / 2, lg["z"] - lg["t"] / 2, lg["z"] + lg["t"] / 2,
                       coll, M["leg"]) for y in lg["y"]]
        _join(f"pcb.legs_{i}", parts, coll)

    t = P["trimmer"]
    s = t["side"]
    for i, (x, y) in enumerate(t["xy"]):
        dx, dy = off(f"trim_{i}")
        x, y = x + dx, y + dy
        tex = f"assets/textures/pcb_trim_{i}_top.png"
        if (lib.ROOT / tex).exists():  # photo-textured top on a box up to the rotor top (bake_id_owned.py)
            zt = t["rotor_top"]
            ob = _bbox(f"pcb.trim_{i}", x - s / 2, x + s / 2, y - s / 2, y + s / 2, zb, zt, coll, M["trim_base"])
            lib.label_quad(f"pcb.label_trim_{i}_top", [[x - s / 2, y + s / 2, zt], [x + s / 2, y + s / 2, zt],
                                                      [x + s / 2, y - s / 2, zt], [x - s / 2, y - s / 2, zt]],
                           coll, lib.mat_image(f"pcb_tex_trim_{i}_top", tex), offset_mm=0.05)
            continue
        base = _bbox("_tb", x - s / 2, x + s / 2, y - s / 2, y + s / 2, zb, t["top"], coll, M["trim_base"])
        hr = t["rotor_top"] - t["top"]
        rot = lib.cylinder("_tr", t["rotor_d"] / 2, hr, (x, y, t["top"] + hr / 2), coll, mat=M["trim_rotor"],
                           verts=20)
        _join(f"pcb.trim_{i}", [base, rot], coll)

    v = P.get("cap_vent", {"inset": 0.5, "h": 0.3})
    for c in P.get("cap", []):
        dx, dy = off(c["name"])
        c = dict(c, x=c["x"] + dx, y=c["y"] + dy)
        h = c["top"] - zb
        objs = [lib.cylinder("_cs", c["d"] / 2, h, (c["x"], c["y"], zb + h / 2), coll, mat=M[c["mat"]], verts=24)]
        if c.get("vent", c["mat"] != "ferrite"):
            objs.append(lib.cylinder("_cv", c["d"] / 2 - v["inset"], v["h"], (c["x"], c["y"], c["top"] + v["h"] / 2),
                                     coll, mat=M["cap_top"], verts=24))
        _join(f"pcb.{c['name']}", objs, coll)

    for a in P.get("axial", []):
        ax = a["axis"]
        dx, dy = off(a["name"])
        cen = (a["center"][0] + dx, a["center"][1] + dy, a["center"][2] + F.get(f"{a['name']}_dz", 0.0))
        objs = [lib.cylinder("_ab", a["r"], a["len"], cen, coll, axis=ax, mat=M[a["mat"]], verts=16)]
        if a.get("lead_len", 0) > 0:
            objs.append(lib.cylinder("_al", 0.3, a["lead_len"], cen, coll, axis=ax, mat=M["leg"], verts=8))
        _join(f"pcb.{a['name']}", objs, coll)

    for q in P.get("tex_quad", []):
        if (lib.ROOT / q["texture"]).exists():
            lib.label_quad(f"pcb.label_{q['name']}", q["corners"], coll,
                           lib.mat_image(f"pcb_tex_{q['name']}", q["texture"]), offset_mm=q.get("offset", 0.05))

