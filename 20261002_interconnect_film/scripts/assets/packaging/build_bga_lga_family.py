"""Build bga_lga_family: nine BGA/LGA packages (3 FCBGA 1.0 mm, 3 fine-pitch 0.8 mm BGA, 3 LGA 1.0 mm) for the S3 NPO scene.

Run: /Applications/Blender.app/Contents/MacOS/Blender -b --python scripts/assets/packaging/build_bga_lga_family.py -- assets/components/packaging
One .blend holds nine ASSET_<id> collections; each root sits at x = index * 60 mm in the file (move the root to the origin when appending).
z = 0 is the ball-tip / land plane; die side is +Z; A1 corner at (-x, +y).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
import pk  # noqa: E402
import parts  # noqa: E402
from pk import MB, MM, PI  # noqa: E402

C = pk.C
OUT = C.argv_after_dashes()[0]
mat = pk.mat
C.reset()
pk.reset_mats()

# id, kind, body, pitch, ball D, array n, center void (cells), substrate t, top style, die size, accuracy
SPECS = [
    dict(id="fcbga_1p0_27", kind="BGA", body=27.0, pitch=1.0, D=0.60, n=25, cv=5, sub=1.0, top="bare", die=14.0, desc="FCBGA 27x27, 1.0 mm pitch, bare die + underfill fillet"),
    dict(id="fcbga_1p0_35", kind="BGA", body=35.0, pitch=1.0, D=0.60, n=33, cv=9, sub=1.2, top="lid", die=20.0, desc="FCBGA 35x35, 1.0 mm pitch, metal lid"),
    dict(id="fcbga_1p0_45", kind="BGA", body=45.0, pitch=1.0, D=0.60, n=43, cv=15, sub=1.5, top="stiff", die=26.0, desc="FCBGA 45x45, 1.0 mm pitch, stiffener ring + die + caps, land-side caps in the center void"),
    dict(id="fbga_0p8_15", kind="BGA", body=15.0, pitch=0.8, D=0.45, n=17, cv=5, sub=0.5, top="bare", die=7.0, desc="fine-pitch flip-chip BGA 15x15, 0.8 mm pitch, bare die"),
    dict(id="fbga_0p8_19", kind="BGA", body=19.0, pitch=0.8, D=0.45, n=21, cv=7, sub=0.6, top="mold", die=0.0, desc="fine-pitch BGA 19x19, 0.8 mm pitch, overmolded"),
    dict(id="fbga_0p8_23", kind="BGA", body=23.0, pitch=0.8, D=0.45, n=26, cv=8, sub=0.5, top="bare", die=11.0, desc="fine-pitch flip-chip BGA 23x23, 0.8 mm pitch, bare die + 0201 caps"),
    dict(id="lga_1p0_23", kind="LGA", body=23.0, pitch=1.0, D=0.65, n=21, cv=7, sub=0.8, top="mold", die=0.0, desc="LGA module 23x23, 1.0 mm pitch lands, overmolded"),
    dict(id="lga_1p0_31", kind="LGA", body=31.0, pitch=1.0, D=0.65, n=29, cv=9, sub=1.0, top="lid", die=16.0, desc="LGA 31x31, 1.0 mm pitch lands, aluminum heat spreader"),
    dict(id="lga_1p0_45", kind="LGA", body=45.0, pitch=1.0, D=0.65, n=43, cv=13, sub=1.2, top="stiff", die=24.0, desc="LGA 45x45, 1.0 mm pitch lands, stiffener ring + die + caps"),
]

records = []
all_pngs = {}
ASSETS = {}


def sub_poly(w, d, ch=1.5, ch2=0.3):
    x, y = w / 2, d / 2
    return [(-x + ch, y), (x - ch2, y), (x, y - ch2), (x, -y + ch2), (x - ch2, -y), (-x + ch2, -y), (-x, -y + ch2), (-x, y - ch)]


def build(idx, S):
    aid = S["id"]
    coll, root = C.new_asset(aid, accuracy="B")
    root.location = (idx * 60.0 * MM, 0, 0)
    src = C.sub_collection(coll, "SOURCES")
    P = aid + "_"
    dims = pk.Dims()
    body, pitch, D, n, cv, st = S["body"], S["pitch"], S["D"], S["n"], S["cv"], S["sub"]
    kind = S["kind"]
    hb = 0.68 * D if kind == "BGA" else 0.04       # standoff (collapsed ball) or land plate height
    z_sub0 = hb                                      # substrate underside
    z_sub1 = z_sub0 + st                             # substrate top

    def obj(name, mb, mats, loc=(0, 0, 0)):
        return mb.to_obj(P + name, [mat(m) for m in mats], coll, root, loc)

    pts = sub_poly(body, body)
    mb = MB()
    mb.prism(pts, z_sub0, z_sub1, mi=(0, 1, 2))
    obj("substrate", mb, ["substrate", "fr4", "mask_green"])

    # ---- underside array
    cells = []
    nvoid = 0
    for j in range(n):
        for i in range(n):
            ci, cj = i - (n - 1) / 2, j - (n - 1) / 2
            if abs(ci) <= (cv - 1) / 2 and abs(cj) <= (cv - 1) / 2:
                nvoid += 1
                continue
            if i in (0, n - 1) and j in (0, n - 1):
                nvoid += 1          # corner voiding
                continue
            if i == 0 and j == n - 1:
                continue
            cells.append((ci * pitch, cj * pitch, 0.0))
    if kind == "BGA":
        mb = MB()
        r = D / 2
        mb.lathe([(0.26 * D, 0.0), (0.485 * D, 0.28 * hb), (r, 0.5 * hb), (0.44 * D, 0.78 * hb), (0.34 * D, hb)], seg=16, mi=0)
        s = pk.source(mb.to_obj(P + "src_ball", [mat("solder")], src, root))
        pk.scatter(P + "balls", cells, s, coll, root)
        # pads under the balls would be hidden; add NSMD copper ring on the mask around each ball base
    else:
        mb = MB()
        r = D / 2
        mb.lathe([(0.0, 0.0), (r, 0.0), (r, 0.03), (0.0, 0.03)], seg=20, mi=0, smooth=False)
        mb.lathe([(r, 0.0), (r + 0.06, 0.0), (r + 0.06, 0.012), (r, 0.012)], seg=20, mi=1, smooth=False)
        s = pk.source(mb.to_obj(P + "src_land", [mat("gold_enig"), mat("copper")], src, root))
        pk.scatter(P + "lands", cells, s, coll, root)
    # fiducials, A1 marker
    mb = MB()
    mbf = MB()
    for (fx, fy) in ((body / 2 - 1.2, -body / 2 + 1.2), (body / 2 - 1.2, body / 2 - 1.2), (-body / 2 + 1.2, -body / 2 + 1.2)):
        mb.lathe([(0.0, z_sub0 - 0.0005), (0.9, z_sub0 - 0.0005), (0.9, z_sub0 - 0.0015), (0.0, z_sub0 - 0.0015)], c=(fx, fy, 0), seg=20, mi=0, smooth=False)
        mbf.lathe([(0.0, z_sub0 - 0.0015), (0.5, z_sub0 - 0.0015), (0.5, z_sub0 - 0.03), (0.0, z_sub0 - 0.03)], c=(fx, fy, 0), seg=20, mi=0, smooth=False)
    obj("fiducial_openings", mb, ["substrate_land"])
    obj("fiducials", mbf, ["gold_enig"])
    mb = MB()
    a = -body / 2 + 0.4
    b2 = body / 2 - 0.4
    mb.prism([(a, b2), (a + 1.6, b2), (a, b2 - 1.6)], z_sub0 - 0.03, z_sub0 - 0.0015, mi=0)
    obj("pin1_underside", mb, ["silkscreen"])
    pk.text(P + "underside_text", "A1", 0.9, (a + 1.2, b2 - 2.2, z_sub0 - 0.0015), mat("silkscreen"), coll, root, rot=(PI, 0, 0), extrude_mm=0.01)

    # ---- top side
    zt = z_sub1
    top_info = ""
    if S["top"] in ("bare", "stiff"):
        dz = 0.775 if S["top"] == "stiff" else 0.4
        dw = S["die"]
        mb = MB()
        mb.box((0, 0, zt + dz / 2 + 0.06), (dw, dw, dz), mi=0, bevel=0.05, seg=1)
        obj("die", mb, ["silicon"])
        mb = MB()
        f = 0.9
        mb.loft_rects((dw + 2 * f, dw + 2 * f, zt), (dw, dw, zt + 0.06 + dz * 0.45), mi=0)
        mb.box((0, 0, zt + 0.03), (dw + 0.2, dw + 0.2, 0.06), mi=0)
        obj("underfill_fillet", mb, ["underfill"])
        if S["top"] == "stiff":
            # stiffener ring (copper alloy, nickel plated) around the die, small caps inside the ring
            wo = body - 4.0
            wi = dw + 2 * 4.5
            mb = MB()
            mb.ring_rect(wo, wo, wi, wi, zt + 0.08, zt + 0.08 + 1.0, rad_o=1.0, chamfer=0.2, seg=3, mi=(0, 0, 0))
            obj("stiffener_ring", mb, ["nickel"])
            mb = MB()
            mb.ring_rect(wo - 0.4, wo - 0.4, wi + 0.4, wi + 0.4, zt, zt + 0.08, rad_o=1.0, seg=3, mi=(0, 0, 0))
            obj("stiffener_adhesive", mb, ["plastic_gray"])
        # 0402/0201 caps around the die
        csz = "0201" if body < 25 else "0402"
        mbc, cm = parts.mlcc(csz, t=None)
        cs = pk.source(mbc.to_obj(P + "src_cap", [mat(m_) for m_ in cm], src, root))
        gap = dw / 2 + (2.6 if S["top"] == "stiff" else 1.8)
        cp, rz = [], []
        spacing = 1.9 if csz == "0402" else 1.2
        k = -(dw / 2 + 1.0)
        while k <= dw / 2 + 1.0:
            for sgn in (-1, 1):
                if abs(gap) < body / 2 - 2.0:
                    cp.append((k, sgn * gap, zt))
                    rz.append(0.0)
                    cp.append((sgn * gap, k, zt))
                    rz.append(PI / 2)
            k += spacing
        pk.scatter(P + "top_caps", cp, cs, coll, root, rz=rz)
        top_info = "die %.0f x %.0f mm, %d caps" % (dw, dw, len(cp))
        mark_z, mark_pos = zt + 0.06 + dz, (0, 0)
        txt_size = max(dw / 14, 0.5)
        pk.text(P + "die_mark", "PKG-%s" % aid.upper().replace("_", "-"), txt_size, (0, 0, mark_z), mat("steel_dark"), coll, root, extrude_mm=0.005)
    elif S["top"] == "lid":
        dw = S["die"]
        hl = 1.6 if kind == "BGA" else 2.0
        lw = body - 3.0
        mb = MB()
        mb.ring_rect(lw + 0.4, lw + 0.4, lw - 3.0, lw - 3.0, zt, zt + 0.12, rad_o=1.0, seg=3, mi=(0, 0, 0))
        obj("lid_adhesive", mb, ["plastic_gray"])
        mb = MB()
        mb.box((0, 0, zt + 0.12 + 1.0 + hl / 2 - 0.5), (lw, lw, hl + 1.0 - 0.0), mi=0, bevel=0.35, seg=2)
        lid_mat = "nickel" if kind == "BGA" else "aluminum"
        obj("lid", mb, [lid_mat])
        top_info = "lid %.0f x %.0f mm" % (lw, lw)
        zl = zt + 0.12 + 1.0 + hl + 0.5 - 0.5
        pk.text(P + "lid_mark_1", "PKG-%s" % aid.upper().replace("_", "-"), 1.6, (0, 2.2, zl), mat("steel_dark"), coll, root, extrude_mm=0.005)
        pk.text(P + "lid_mark_2", "%dx%d  %.1fP  R1" % (n, n, pitch), 1.3, (0, 0, zl), mat("steel_dark"), coll, root, extrude_mm=0.005)
        pk.text(P + "lid_mark_3", "LOT 000001  0001", 1.1, (0, -2.0, zl), mat("steel_dark"), coll, root, extrude_mm=0.005)
        mb = MB()
        mb.prism([(-lw / 2 + 1.4, lw / 2 - 1.4), (-lw / 2 + 3.4, lw / 2 - 1.4), (-lw / 2 + 1.4, lw / 2 - 3.4)], zl - 0.02, zl + 0.0, mi=0)
        obj("lid_pin1", mb, ["steel_dark"])
    elif S["top"] == "mold":
        mh = 0.6 if kind == "BGA" else 1.0
        mb = MB()
        mb.box((0, 0, zt + mh / 2), (body - 0.2, body - 0.2, mh), mi=0, bevel=0.15, seg=2)
        obj("mold_cap", mb, ["mold_black"])
        zl = zt + mh
        mb = MB()
        mb.cyl((-body / 2 + 2.0, body / 2 - 2.0, zl - 0.04), 0.5, 0.1, seg=16, mi=0)
        obj("pin1_dimple", mb, ["mold_black"])
        pk.text(P + "mold_mark_1", "PKG-%s" % aid.upper().replace("_", "-"), body / 14, (0, body * 0.12, zl), mat("laser_gold"), coll, root, extrude_mm=0.004)
        pk.text(P + "mold_mark_2", "%dx%d  %.1fP  R1" % (n, n, pitch), body / 18, (0, 0, zl), mat("laser_gold"), coll, root, extrude_mm=0.004)
        pk.text(P + "mold_mark_3", "LOT 000001  0001", body / 20, (0, -body * 0.1, zl), mat("laser_gold"), coll, root, extrude_mm=0.004)
        top_info = "overmold %.1f mm" % mh
    # land-side caps in the center void (45 mm types)
    if S["top"] == "stiff":
        mbl, cml = parts.mlcc("0402", t=0.30)
        mbl.mirror_z()
        ls = pk.source(mbl.to_obj(P + "src_lsc", [mat(m_) for m_ in cml], src, root))
        lp = []
        for j in range(-int(cv * pitch / 2 / 1.0) + 1, int(cv * pitch / 2 / 1.0)):
            for i in range(-int(cv * pitch / 2 / 1.7) + 1, int(cv * pitch / 2 / 1.7)):
                lp.append((i * 1.7, j * 1.0, z_sub0))
        pk.scatter(P + "land_side_caps", lp, ls, coll, root)
        top_info += ", %d land-side caps" % len(lp)

    C.hook("underside_center", coll, root, loc=(0, 0, 0))
    C.hook("top_center", coll, root, loc=(0, 0, zt * MM + 0.0))
    C.hook("pin1", coll, root, loc=((-body / 2 + 0.8) * MM, (body / 2 - 0.8) * MM, zt * MM))
    root["p_ball_count"] = len(cells)

    dims.add("body size", "%.0f x %.0f" % (body, body), "mm", "standard JEDEC-style square body size (MO-275 family: 15/19/23/27/35/45 mm are common bodies); not tied to one part", "B")
    dims.add("ball/land pitch", pitch, "mm", "JEDEC fine-pitch BGA pitches 1.0 and 0.8 mm (MO-275 family, via Analog Devices package outline PDFs and NXP AN10778)", "A")
    dims.add("ball diameter" if kind == "BGA" else "land diameter", D, "mm", "typical ball diameters by pitch: 0.8 mm -> 0.45, 1.0 mm -> 0.50-0.60 (NXP AN10778 footprint table, TI MicroStar guide); LGA land 0.65 is a typical gold land for 1.0 mm pitch (not a spec value)", "B")
    if kind == "BGA":
        dims.add("collapsed ball standoff", round(hb, 3), "mm", "0.68 x ball diameter (typical reflowed height ratio); range 0.55-0.75", "C")
    else:
        dims.add("land height above substrate", hb, "mm", "typical plated land stack 30-40 um", "C")
    dims.add("array", "%d x %d" % (n, n), "cells", "chosen to leave about 1.5 mm edge margin", "C")
    dims.add("center void", "%d x %d cells" % (cv, cv), "cells", "thermal/ground center depopulation pattern (illustrative)", "C")
    dims.add("ball/land count", len(cells), "count", "full array minus center void minus 4 corner cells minus the A1 corner cell", "A")
    dims.add("substrate thickness", st, "mm", "typical laminate thickness for the class (0.5-0.6 mm FBGA, 0.8-1.5 mm FCBGA/LGA)", "C")
    dims.add("top style", S["top"], "", top_info, "C")
    meta = {
        "title": S["desc"], "accuracy_level": "B (pitch/ball/body from standards-style conventions; heights and top details C)",
        "origin": "z = 0 at the ball tip / land plane, X/Y centered; die side +Z; A1 corner at (-X, +Y). The root sits at x = %d mm in the file (family file); move to the origin when appending" % (idx * 60),
        "usage": "S3: stand the package up with its underside to camera (rotate about X by 90 deg); front = -Y",
        "sources": [
            {"what": "ball diameter by pitch", "url": "https://www.nxp.com/docs/en/application-note/AN10778.pdf", "accessed": "2026-10-02"},
            {"what": "JEDEC MO-275 fine pitch BGA outlines", "url": "https://www.analog.com/media/en/package-pcb-resources/package/pkg_pdf/sbga-heatsink/BP_196_3.pdf", "accessed": "2026-10-02"},
            {"what": "BGA packaging reference", "url": "https://www.ti.com/lit/pdf/ssyz015", "accessed": "2026-10-02"}],
        "instancing": "balls/lands, caps are Geometry Nodes instances (NG_pk_scatter) with sources in SOURCES",
        "simplifications": ["no solder mask openings on the BGA side (ball pads hidden behind the balls)", "no top-side routing", "marking text is generic"],
        "family_note": "family file bga_lga_family.blend holds 9 ASSET_ collections; this JSON describes one; see bga_lga_family.json for the table",
    }
    return coll, root, dims, meta, len(cells)


colls = []
for idx, S in enumerate(SPECS):
    colls.append((S,) + build(idx, S))

blend = os.path.join(OUT, "bga_lga_family.blend")
C.save(blend)
pv = os.path.join(OUT, "previews")
family = {"family": "bga_lga_family", "assets": []}
for (S, coll, root, dims, meta, nb) in colls:
    aid = S["id"]
    bb, tris = pk.eval_stats(coll)
    ctr = [(bb[0][k] + bb[1][k]) / 2 for k in range(3)]
    body = S["body"]
    r = body
    cx, cy = ctr[0], 0.0
    views = [dict(name="front", loc=(cx, -2.6 * r, 0.55 * r), tgt=(cx, 0, ctr[2]), lens=50, floor=True),
             dict(name="three_quarter", loc=(cx + 1.4 * r, -1.9 * r, 1.4 * r), tgt=(cx, 0, ctr[2]), lens=50, floor=True),
             dict(name="top", loc=(cx, -0.01, 2.4 * r), tgt=(cx, 0, ctr[2]), lens=50),
             dict(name="underside", loc=(cx, -0.01, -2.4 * r), tgt=(cx, 0, 0), lens=50),
             dict(name="closeup_balls", loc=(cx - body * 0.28, -body * 0.5, -body * 0.18), tgt=(cx - body * 0.28, -body * 0.2, 0.2), lens=70)]
    for o in bpy.data.objects:
        pass
    pngs = pk.render_views(coll, pv, aid, views)
    hooks, mnames, props = pk.collect_auto_meta(coll, root)
    meta.update({"asset_id": aid, "bbox_mm": bb, "size_mm": [round(bb[1][k] - bb[0][k], 3) for k in range(3)],
                 "triangles_including_instances": tris, "dimension_table": dims.rows, "hooks": hooks, "material_slots": mnames,
                 "custom_properties": props, "previews": [os.path.relpath(p, OUT) for p in pngs], "ball_or_land_count": nb})
    pk.write_json(os.path.join(OUT, aid + ".blend"), meta)       # per-asset JSON next to the family blend
    family["assets"].append({"id": aid, "desc": S["desc"], "count": nb, "size_mm": meta["size_mm"], "tris": tris})
    print("PKG", aid, nb, tris, meta["size_mm"])
C.save(blend)
pk.write_json(blend, family)
